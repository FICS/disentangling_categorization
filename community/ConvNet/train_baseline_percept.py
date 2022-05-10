import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import os
import re
import shutil
import logging
import sys
sys.path.append("/opt/app/prototype-signal-game")

import torchvision.models as models
import community.jalal_models as jalal_models

from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

import util
from dataloader_dali import normalized_train_loader, normalized_test_loader


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to configuration')
parser.add_argument('--gpuid', nargs='+', type=str, default="0")
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args['gpuid']) if type(args['gpuid']) is list else f"{args['gpuid']}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device={device}:{os.environ['CUDA_VISIBLE_DEVICES']}")

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
                        logging.FileHandler(os.path.join(state['save_dir'], 'train_baseline_percept.log')),
                        logging.StreamHandler()
                             ],
                    level=logging.DEBUG)
log = logging.getLogger('train_baseline_percept')

config_basename = args['config'].split('/')[-1]
shutil.copyfile(args['config'], os.path.join(state['save_dir'], config_basename))
            
            
# From Nidia DALI sample code
def adjust_learning_rate(initial_lr, optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = initial_lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def train(state, model, model_multi, train_loader, test_loader, start_epoch=0):
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
            
    # Loss and Optimizer
    model_multi.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_update, 
                                 lr=state["learning_rate"], 
                                 weight_decay=state['weight_decay'])
    n_batches = train_loader.n_batches_per_epoch
    best_acc = 0
    # Train the Model
    for epoch in range(start_epoch, state['epochs']):
        with tqdm(total=n_batches) as pb:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(state['device']), labels.to(state['device'])
                
                # adjust_learning_rate(state["learning_rate"], optimizer, epoch, i, n_batches)
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model_multi(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                str_loss = f"{loss.cpu().data.numpy():.4f}"
                pb.update(1)
                pb.set_postfix(epoch=epoch, loss=str_loss)

            train_loader.reset()

        if (epoch + 1) % state["save_freq"] == 0:
            ckpt_path = os.path.join(state['save_dir'], f"{state['dataset']}_epoch-{epoch + 1}_seed-{state['seed']}.pth")
            model_state = {
                'epoch': epoch + 1,
                'architecture': state['architecture'],
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'num_classes': state['num_classes'],
            }
            torch.save(model_state, ckpt_path)
            log.info(f"Saved to {ckpt_path}")
        if (epoch + 1) % state["test_freq"] == 0:
            model_multi.eval()
            accu = test(state, model, test_loader)
            if accu > best_acc:
                ckpt_path = os.path.join(state['save_dir'], f"best.pth")
                model_state = {
                    'accu': accu,
                    'epoch': epoch + 1,
                    'architecture': state['architecture'],
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'num_classes': state['num_classes'],
                }
                torch.save(model_state, ckpt_path)
                log.info(f"Saved best so far to {ckpt_path}")
                best_acc = accu
                
            model_multi.train()

    return model


def test(state, model, test_loader):
    # Test the Model
    actuals = []
    preds = []
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    n_batches = test_loader.n_batches_per_epoch
    with tqdm(total=n_batches) as pb:
        for (images, labels) in test_loader:
            images, labels = images.to(state['device']), labels.to(state['device'])

            # images = Variable(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            actuals.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
            
            accu = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
            str_acc = f"{accu:.4f}"
            pb.update(1)
            pb.set_postfix(accuracy=str_acc)

        test_loader.reset()
    
    accu = accuracy_score(actuals, preds)
    log.info(f'Test Accuracy of the model on test images: {accu:.3f}')
    return accu


if state['dataset'].lower() != 'mnist':
    for d in [state['train_dir'], state['test_dir']]:
        if not os.path.exists(d):
            raise IOError(f"Dataset folder does not exist: {d}")


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
            
            
if 'resnet' in state['architecture']:
    net = models.__dict__[state['architecture']](pretrained=True, num_classes=1000)
    set_parameter_requires_grad(net)
    num_feats = net.fc.in_features
    net.fc = nn.Linear(num_feats, state['num_classes'])
elif 'densenet' in state['architecture']:
    net = models.__dict__[state['architecture']](pretrained=True, num_classes=1000)
    set_parameter_requires_grad(net)
    num_feats = net.classifier.in_features
    net.classifier = nn.Linear(num_feats, state['num_classes'])
elif 'vgg' in state['architecture']:
    net = models.__dict__[state['architecture']](pretrained=True, num_classes=1000)
    set_parameter_requires_grad(net)
    num_feats = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_feats, state['num_classes'])
elif 'jalal' in state['architecture']:
    # Mnist always start from no pretrain
    net = jalal_models.__dict__[state['architecture']](pretrained=False)
else:
    print(f"{state['architecture']} not supported, exiting...")
    exit()

start_epoch = 0

if state['reload_path']:
    model_state = torch.load(state['reload_path'])
    net.load_state_dict(model_state['state_dict'], strict=False)
    start_epoch = model_state['epoch']
    log.info(f"Reloaded at epoch {start_epoch} from {state['reload_path']}.")


mean = state['train_mean']
std = state['train_std']
seed = state['seed']
img_size = state['img_size']

net_multi = torch.nn.DataParallel(net).cuda()
train_loader = normalized_train_loader(state, img_size, seed, mean=mean, std=std)
test_loader = normalized_test_loader(state, img_size, seed, mean=mean, std=std)

train(state, net, net_multi, train_loader, test_loader, start_epoch=start_epoch)
test(state, net_multi, test_loader)

# else:
#     raise ValueError("Unsupported dataset")
