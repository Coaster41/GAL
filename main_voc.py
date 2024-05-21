import os
import numpy as np
import utils.common as utils
from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from model import resnet_voc

from torchvision import datasets, transforms
import numpy as np
import torch
from torchvision import datasets
from xml.etree.ElementTree import Element as ET_Element
from typing import Any, Dict
import collections
from sklearn.metrics import average_precision_score, f1_score

from fista import FISTA
from model import Discriminator

from data import cifar10
import pdb 

device = torch.device(f"cuda:0")
checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')

def compute_mAP(labels,outputs):
    AP = []
    for i in range(labels.shape[0]):
        AP.append(average_precision_score(labels[i],outputs[i]))
    return np.mean(AP)

def compute_f1(labels, outputs):
    outputs = outputs > 0.5
    return f1_score(labels, outputs, average="samples")

class VOCnew(datasets.VOCDetection):
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, range(len(classes))))   

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        

        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(datasets.VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
                objs = [def_dic["object"]]
                lbl = np.zeros(len(VOCnew.classes))
                for ix, obj in enumerate(objs[0][0]):        
                    obj_class = VOCnew.class_to_ind[obj['name']]
                    lbl[obj_class] = 1
                return lbl
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

class Data:
    def __init__(self):
        self.loader_train = VOCnew(root=r'/tmp/public_dataset/pytorch/pascalVOC-data', image_set='train', download=False,
                        transform=transforms.Compose([
                            transforms.Resize(330),
                            transforms.Pad(30),
                            transforms.RandomCrop(300),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))

        self.loader_test = VOCnew(root=r'/tmp/public_dataset/pytorch/pascalVOC-data', image_set='val', download=False,
                        transform=transforms.Compose([
                            transforms.Resize(330), 
                            transforms.CenterCrop(300),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))

class VocModel(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        # Use a pretrained model
        self.network = resnet_voc.resnet34(weights=weights, mask=True)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

def main():

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    # Data loading
    print('=> Preparing data..')
    
    loader = Data()

    # Create model
    print('=> Building model...')
    model_t = VocModel(20).to(device)

    # Load teacher model
    ckpt_t = torch.load(args.teacher_dir, map_location=device)
    # pdb.set_trace()

    state_dict_t = ckpt_t


    model_t.load_state_dict(state_dict_t, strict=False)
    model_t = model_t.to(device)

    for para in list(model_t.parameters())[:-2]:
        para.requires_grad = False

    model_s = VocModel(20).to(device)

    # model_dict_s = model_s.state_dict()
    # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]

    
    # new_state_dict = {}
    # for layer_name, arr in state_dict_t.items():
    #     if 'feature' in layer_name:
    #         cor_layer_num = -1
    #         cnt_down = 0
    #         cfg_iter = 0
    #         split_name = layer_name.split('.')
    #         num = int(split_name[1])+1
    #         for i in range(num):
    #             if cnt_down == 0:
    #                 cnt_down = 3 if cfg[cfg_iter] != 'M' else 1
    #                 cor_layer_num += 1 if cfg[cfg_iter] != 'M' else 0
    #                 cfg_iter += 1
    #             cor_layer_num += 1
    #             cnt_down -= 1
    #         split_name[1] = str(cor_layer_num-1)
    #         new_state_dict['.'.join(split_name)] = arr


    # model_dict_s.update(new_state_dict)
    model_s.load_state_dict(state_dict_t, strict=False)

    if len(args.gpus) != 1:
        model_s = nn.DataParallel(model_s, device_ids=args.gpus)

    model_d = Discriminator().to(device) 

    models = [model_t, model_s, model_d]

    optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # mask_index = []
    # cor_layer_num = -1
    # cnt_down = 3
    # cfg_iter = 0
    # while True:
    #     mask_index.append(0)
    #     cor_layer_num += 1
    #     cnt_down -= 1
    #     if cnt_down == 0:
    #         print(cnt_down)
    #         if cfg[cfg_iter] != 'M':
    #             mask_index.append(1)
    #         cfg_iter += 1
    #         if cfg_iter == len(cfg):
    #             break
    #         cnt_down = 3 if cfg[cfg_iter] != 'M' else 1
    #         cor_layer_num += 1 if cfg[cfg_iter] != 'M' else 0
    # print(mask_index)

    param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]
    param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]

    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_m = FISTA(param_m, lr=args.lr, gamma=args.sparse_lambda)

    scheduler_d = StepLR(optimizer_d, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_s = StepLR(optimizer_s, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_m = StepLR(optimizer_m, step_size=args.lr_decay_step, gamma=0.1)

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_s.load_state_dict(ckpt['state_dict_s'])
        model_d.load_state_dict(ckpt['state_dict_d'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        optimizer_s.load_state_dict(ckpt['optimizer_s'])
        optimizer_m.load_state_dict(ckpt['optimizer_m'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        scheduler_s.load_state_dict(ckpt['scheduler_s'])
        scheduler_m.load_state_dict(ckpt['scheduler_m'])
        print('=> Continue from epoch {}...'.format(start_epoch))


    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return

    optimizers = [optimizer_d, optimizer_s, optimizer_m]
    schedulers = [scheduler_d, scheduler_s, scheduler_m]
    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)

        train(args, loader.loader_train, models, optimizers, epoch)
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        model_state_dict = model_s.module.state_dict() if len(args.gpus) > 1 else model_s.state_dict()

        state = {
            'state_dict_s': model_state_dict,
            'state_dict_d': model_d.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_s': optimizer_s.state_dict(),
            'optimizer_m': optimizer_m.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'scheduler_s': scheduler_s.state_dict(),
            'scheduler_m': scheduler_m.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    print_logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

    best_model = torch.load(f'{args.job_dir}/checkpoint/model_best.pt', map_location=device)

    model = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, best_model['state_dict_s'])

def train(args, loader_train, models, optimizers, epoch):
    losses_d = utils.AverageMeter()
    losses_data = utils.AverageMeter()
    losses_g = utils.AverageMeter()
    losses_sparse = utils.AverageMeter()
    mAP = utils.AverageMeter(':6.3f')
    f1 = utils.AverageMeter(':.4e')

    model_t = models[0]
    model_s = models[1]
    model_d = models[2]

    bce_logits = nn.BCEWithLogitsLoss()

    optimizer_d = optimizers[0]
    optimizer_s = optimizers[1]
    optimizer_m = optimizers[2]

    # switch to train mode
    model_d.train()
    model_s.train()
        
    num_iterations = len(loader_train)

    real_label = 1
    fake_label = 0

    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)

        features_t = model_t(inputs)
        features_s = model_s(inputs)

        ############################
        # (1) Update D network
        ###########################

        for p in model_d.parameters():  
            p.requires_grad = True  

        optimizer_d.zero_grad()

        output_t = model_d(features_t.detach())
        # print('output_t',output_t)
        labels_real = torch.full_like(output_t, real_label, device=device)
        error_real = bce_logits(output_t, labels_real)

        output_s = model_d(features_s.to(device).detach())
        # print('output_s',output_t)
        labels_fake = torch.full_like(output_s, fake_label, device=device)
        error_fake = bce_logits(output_s, labels_fake)

        error_d = error_real + error_fake

        labels = torch.full_like(output_s, real_label, device=device)
        error_d += bce_logits(output_s, labels)

        error_d.backward()
        losses_d.update(error_d.item(), inputs.size(0))

        writer_train.add_scalar(
            'discriminator_loss', error_d.item(), num_iters)

        optimizer_d.step()

        ############################
        # (2) Update student network
        ###########################

        for p in model_d.parameters():  
            p.requires_grad = False  

        optimizer_s.zero_grad()
        optimizer_m.zero_grad()

        error_data = args.miu * F.mse_loss(features_t, features_s.to(device))

        losses_data.update(error_data.item(), inputs.size(0))
        writer_train.add_scalar(
            'data_loss', error_data.item(), num_iters)
        error_data.backward(retain_graph=True)

        # fool discriminator
        output_s = model_d(features_s.to(device))
        
        labels = torch.full_like(output_s, real_label, device=device)
        error_g = bce_logits(output_s, labels)

        losses_g.update(error_g.item(), inputs.size(0))
        writer_train.add_scalar(
            'generator_loss', error_g.item(), num_iters)
        error_g.backward(retain_graph=True)

        # train mask
        mask = []
        for name, param in model_s.named_parameters():
            if 'mask' in name:
                mask.append(param.view(-1))
        mask = torch.cat(mask)
        error_sparse = args.sparse_lambda * torch.norm(mask, 1)
        error_sparse.backward()

        losses_sparse.update(error_sparse.item(), inputs.size(0))
        writer_train.add_scalar(
        'sparse_loss', error_sparse.item(), num_iters)

        optimizer_s.step()

        decay = (epoch % args.lr_decay_step == 0 and i == 1)
        if i % args.mask_step == 0:
            optimizer_m.step(decay)
        labels_cpu = targets.cpu().detach().numpy()
        outputs_cpu = features_s.cpu().detach().numpy()
        mAP.update(compute_mAP(labels_cpu, outputs_cpu), inputs.size(0))
        f1.update(compute_f1(labels_cpu, outputs_cpu), inputs.size(0))
        # prec1, prec5 = utils.accuracy(features_s, targets, topk=(1, 5))
        # top1.update(prec1[0], inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))

        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}):\t'
                'Loss_sparse {loss_sparse.val:.4f} ({loss_sparse.avg:.4f})\t'
                'Loss_data {loss_data.val:.4f} ({loss_data.avg:.4f})\t'
                'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                'Loss_g {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                'mAP {mAP.val*100:.3f} ({mAP.avg*100:.3f})\t'
                'f1_score {f1.val:.3f} ({f1.avg:.3f})'.format(
                epoch, i, num_iterations, loss_sparse=losses_sparse, loss_data=losses_data, loss_g=losses_g, loss_d=losses_d, mAP=mAP, f1=f1))
            
            mask = []
            pruned = 0
            num = 0
            
            for name, param in model_s.named_parameters():
                if 'mask' in name:
                    weight_copy = param.clone()
                    param_array = np.array(weight_copy.detach().cpu())
                    pruned += sum(w == 0 for w in param_array)
                    num += len(param_array)
                    
            print_logger.info("Pruned {} / {}".format(pruned, num))

def test(args, loader_test, model_s):
    losses = utils.AverageMeter()
    # top1 = utils.AverageMeter()
    # top5 = utils.AverageMeter()
    
    mAP = utils.AverageMeter(':6.3f')
    f1 = utils.AverageMeter(':.4e')

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_s.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_s(inputs).to(device)
            loss = cross_entropy(logits, targets)

            labels_cpu = targets.cpu().detach().numpy()
            outputs_cpu = logits.cpu().detach().numpy()
            mAP.update(compute_mAP(labels_cpu, outputs_cpu), inputs.size(0))
            f1.update(compute_f1(labels_cpu, outputs_cpu), inputs.size(0))
            # prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            # top1.update(prec1[0], inputs.size(0))
            # top5.update(prec5[0], inputs.size(0))
        
        print_logger.info('mAP {mAP.avg*100:.3f} f1_score {f1.avg:.3f}'
        .format(mAP=mAP, f1=f1))

    return mAP.avg, f1.avg
    
if __name__ == '__main__':
    main()


