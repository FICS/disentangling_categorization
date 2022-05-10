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

from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F

import util
from dataloader_dali import normalized_test_loader
from ConvNet.test_and_train import test


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to configuration')
parser.add_argument('--reload_path', required=True, help='path to checkpoint')
parser.add_argument('--override_batch_size', required=False, help='override batch size to a different value for VRAM.', type=int)
parser.add_argument('--gpuid', nargs='+', type=str, default="0")
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args['gpuid']) if type(args['gpuid']) is list else f"{args['gpuid']}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device={device}:{os.environ['CUDA_VISIBLE_DEVICES']}")

state = util.load_yaml(args['config'])
state['device'] = device

# ============ start logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler(os.path.join(state['save_dir'], 'test_baseline_percept.log')),
                        logging.StreamHandler()
                             ],
                    level=logging.DEBUG)
log = logging.getLogger('test_baseline_percept')


net = models.__dict__[state['architecture']](pretrained=False, num_classes=state['num_classes'])
model_state = torch.load(args['reload_path'])['state_dict']
# model_state = {k.replace('module.', ''): v for k,v in model_state.items()}
net.load_state_dict(model_state)
log.info(f"Reloaded  from {args['reload_path']}.")

mean = state['train_mean']
std = state['train_std']
seed = state['seed']
img_size = state['img_size']

if args['override_batch_size']:
    state['test_batch_size'] = args['override_batch_size']

net = torch.nn.DataParallel(net).cuda()
test_loader = normalized_test_loader(state, img_size, seed, mean=mean, std=std)

_ = test(state, net, test_loader, log.info)

