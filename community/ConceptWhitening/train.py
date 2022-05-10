import argparse
import os
import sys
sys.path.append("/opt/app/prototype-signal-game")
import gc
import shutil
import time
import random
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import yaml

from ConceptWhitening.model import construct_CW

from plot_functions import *
from PIL import ImageFile, Image

import util


ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    

def main(state, log):
    torch.manual_seed(state['seed'])
    torch.cuda.manual_seed_all(state['seed'])
    random.seed(state['seed'])
    
    # state['prefix'] += '_'+'_'.join(state['whitened_layers'].split(','))
    
    # set up concepts
    state['concepts'] = os.listdir(state['concept_train_dir'])

    #create model
    model = construct_CW(state)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # param_list = get_param_list_bn(model)
    optimizer = torch.optim.SGD(model.parameters(), state['lr'],
                           momentum=state['momentum'],
                           weight_decay=state['weight_decay'])
                            
    model_multi = torch.nn.DataParallel(model)
    model_multi = model_multi.cuda()
    # log.info ("model")
    # log.info (model)

    # get the number of model parameters
    log.info('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    cudnn.benchmark = True

    # Data loading code
    traindir = state['train_dir']
    valdir = state['val_dir']
    testdir = state['test_dir']
    conceptdir_train = state['concept_train_dir']
    conceptdir_test = state['concept_test_dir']
    mean = state['train_mean']
    std = state['train_std']
    normalize = transforms.Normalize(mean=mean, std=std)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=state['train_batch_size'], shuffle=True,
        num_workers=state['workers'], pin_memory=False)

    concept_loaders = [
        torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(conceptdir_train, concept), transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=state['train_batch_size'], shuffle=True,
        num_workers=state['workers'], pin_memory=False)
        for concept in state['concepts']
    ]

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=state['test_batch_size'], shuffle=True,
        num_workers=state['workers'], pin_memory=False)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=state['test_batch_size'], shuffle=False,
        num_workers=state['workers'], pin_memory=False)

    test_loader_with_path = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=state['test_batch_size'], shuffle=True,
        num_workers=state['workers'], pin_memory=False)

    log.info("Start training")
    best_prec1 = 0
    for epoch in range(state['start_epoch'], state['start_epoch'] + 4):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if 'cw' in state['architecture']:
            train(train_loader, concept_loaders, model_multi, criterion, optimizer, epoch)
        elif 'baseline' in state['architecture']:
            train_baseline(train_loader, concept_loaders, model_multi, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = test(val_loader, model_multi, log=log.info)
        prec1 = validate(val_loader, model_multi, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        ckpt_dir = os.path.join(state['save_dir'], f"{state['dataset']}_epoch-{epoch + 1}_seed-{state['seed']}.pth")
        model_state = {
            'accu': prec1.cpu().numpy(),
            'epoch': epoch + 1,
            'architecture': state['architecture'],
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'concepts': state['concepts'],
            'act_mode': state['act_mode'],
            'whitened_layers': state['whitened_layers'],
            'num_classes': state['num_classes'],
        }
        torch.save(model_state, ckpt_dir)
        log.info(f"Saved to {ckpt_dir}")
        if is_best:
            ckpt_dir = os.path.join(state['save_dir'], f"best.pth")
            torch.save(model_state, ckpt_dir)
            log.info(f"Saved best so far to {ckpt_dir}")
            


def train(train_loader, concept_loaders, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # if (i + 1) % 800 == 0:
        #     break
        if (i + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                # update the gradient matrix G
                for concept_index, concept_loader in enumerate(concept_loaders):
                    model.module.change_mode(concept_index)
                    for j, (X, _) in enumerate(concept_loader):
                        X_var = torch.autograd.Variable(X).cuda()
                        _ = model(X_var)
                        break
                model.module.update_rotation_matrix()
                # change to ordinary mode
                model.module.change_mode(-1)
            model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % state['print_freq'] == 0:
            log.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                     f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                     f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')


def test(test_loader, model, log=print):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for (images, labels) in test_loader:
        images, labels = images.cuda(), labels.cuda()

        # images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    
    accuracy = (100.0 * correct / total)
    if type(accuracy) is torch.Tensor:
        accuracy = accuracy.cpu().numpy()
        
    log(f'Test Accuracy of the model on test images: {accuracy:.3f} %%')
    return accuracy


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    from sklearn.metrics import accuracy_score
    preds = []
    actuals = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            #target = target.cuda(async=True)
            # input_var = torch.autograd.Variable(input)
            # target_var = torch.autograd.Variable(target)
            
            # compute output
            output = model(input)
            # loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            _, pred = torch.max(output.data, 1)
            pred = pred.cpu().numpy()
            preds.extend(pred)
            # print('pred', pred)
            # print('target', target.cpu().numpy())
            actuals.extend(target.cpu().numpy())
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % state['print_freq'] == 0:
                log.info(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')
    
    acc = accuracy_score(actuals, preds) * 100
    log.info(f' * Accuracy {acc:.3f} ')
    return top1.avg


'''
This function train a baseline with auxiliary concept loss jointly
train with main objective
'''
def train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch, activation_mode = 'pool_max'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_aux = AverageMeter()
    top1_cpt = AverageMeter()

    n_cpt = len(concept_loaders)

    # switch to train mode
    model.train()
    
    end = time.time()

    inter_feature = []
    def hookf(module, input, output):
        inter_feature.append(output[:,:n_cpt,:,:])
    for i, (input, target) in enumerate(train_loader):
        if (i + 1) % 20 == 0:

            #model.eval()
            
            layer = int(state['whitened_layers'][0])
            layers = model.module.layers
            if layer <= layers[0]:
                hook = model.module.model.layer1[layer-1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1]:
                hook = model.module.model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2]:
                hook = model.module.model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                hook = model.module.model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hookf)
            
            y = []
            inter_feature = []
            for concept_index, concept_loader in enumerate(concept_loaders):
                for j, (X, _) in enumerate(concept_loader):
                    y += [concept_index] * X.size(0)
                    X_var = torch.autograd.Variable(X).cuda()
                    model(X_var)
                    break
            
            inter_feature = torch.cat(inter_feature,0)
            y_var = torch.Tensor(y).long().cuda()
            f_size = inter_feature.size()
            if activation_mode == 'mean':
                y_pred = F.avg_pool2d(inter_feature,f_size[2:]).squeeze()
            elif activation_mode == 'max':
                y_pred = F.max_pool2d(inter_feature,f_size[2:]).squeeze()
            elif activation_mode == 'pos_mean':
                y_pred = F.avg_pool2d(F.relu(inter_feature),f_size[2:]).squeeze()
            elif activation_mode == 'pool_max':
                kernel_size = 3
                y_pred = F.max_pool2d(inter_feature, kernel_size)
                y_pred = F.avg_pool2d(y_pred,y_pred.size()[2:]).squeeze()
            
            loss_cpt = 10*criterion(y_pred, y_var)
            # measure accuracy and record loss
            [prec1_cpt] = accuracy(y_pred.data, y_var, topk=(1,))
            loss_aux.update(loss_cpt.data, f_size[0])
            top1_cpt.update(prec1_cpt[0], f_size[0])
            
            optimizer.zero_grad()
            loss_cpt.backward()
            optimizer.step()

            hook.remove()
            #model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % state['print_freq'] == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_aux {loss_a.val:.4f} ({loss_a.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Prec_cpt@1 {top1_cpt.val:.3f} ({top1_cpt.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_a=loss_aux, top1=top1, top5=top5, top1_cpt=top1_cpt))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = state['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    # ==========
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to configuration')
    parser.add_argument('--gpuid', nargs='+', type=str, default="0")
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args['gpuid']) if type(args['gpuid']) is list else f"{args['gpuid']}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = util.load_yaml(args['config'])
    state['device'] = device

    if "run_id" in list(state.keys()):
        run_id = state["run_id"] if state["run_id"] != "" else util.get_time_stamp()
        run_id = str(run_id)
    else:
        run_id = util.get_time_stamp()
    
    state['save_dir'] = os.path.join(state['save_dir'], state['architecture'], run_id)
    if not os.path.exists(state['save_dir']):
        os.makedirs(state['save_dir'])

    # ============ start logging
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        handlers=[
                            logging.FileHandler(os.path.join(state['save_dir'], 'train_CW.log')),
                            logging.StreamHandler()
                                 ],
                        level=logging.DEBUG)
    log = logging.getLogger('train CW')
    log.info(f"Using device={device}:{os.environ['CUDA_VISIBLE_DEVICES']}")

    config_basename = args['config'].split('/')[-1]
    shutil.copyfile(args['config'], os.path.join(state['save_dir'], config_basename))
    # ==========

    main(state, log)
