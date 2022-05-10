import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

import argparse
import re
import random
import logging

from helpers import makedir
import ProtoPNet.model as model
import ProtoPNet.push as push
import ProtoPNet.prune as prune
import ProtoPNet.train_and_test as tnt
import ProtoPNet.save as save
from ProtoPNet.preprocess import preprocess_input_function

import util

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to configuration')
parser.add_argument('--gpuid', nargs='+', type=str, default=0) # python3 main.py -gpuid=0,1,2,3
args = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args['gpuid']) if type(args['gpuid']) is list else f"{args['gpuid']}"
print(f"GPU ID list: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
                        logging.FileHandler(os.path.join(state['save_dir'], 'train_ProtoPNet.log')),
                        logging.StreamHandler()
                             ],
                    level=logging.DEBUG)
log = logging.getLogger('train ProtoPNet')
log.info(f"Using device={device}:{os.environ['CUDA_VISIBLE_DEVICES']}")

config_basename = args['config'].split('/')[-1]
shutil.copyfile(args['config'], os.path.join(state['save_dir'], config_basename))
# ==========

cudnn.benchmark = True
torch.manual_seed(state['seed'])
torch.cuda.manual_seed_all(state['seed'])
random.seed(state['seed'])


# book keeping namings and code
dataset = state['dataset']
base_architecture = state['architecture']
# base_architecture, img_size, prototype_shape, num_classes, \
#                      prototype_activation_function, add_on_layers_type, experiment_run, \
#                      num_data_workers, ckpt_dir, dataset
print(f"dataset is {dataset}")
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

# case for when launched as ProtoPNet/main_modular.py
if '/' in __file__:
    filedir = __file__.split('/')[:-1]
    filedir = '/'.join(filedir)
    cwd = os.path.join(os.getcwd(), filedir)
else:
    cwd = os.getcwd()
    
# model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
# model_dir = os.path.join(ckpt_dir, base_architecture, experiment_run)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=state['save_dir'])
# shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=state['save_dir'])
shutil.copy(src=os.path.join(cwd, base_architecture_type + '_features.py'), dst=state['save_dir'])
shutil.copy(src=os.path.join(cwd, 'model.py'), dst=state['save_dir'])
shutil.copy(src=os.path.join(cwd, 'train_and_test.py'), dst=state['save_dir'])

img_dir = os.path.join(state['save_dir'], 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
train_dir = state['train_dir']
test_dir = state['test_dir']
train_push_dir = state['train_push_dir']
train_batch_size = state['train_batch_size']
test_batch_size = state['test_batch_size']
train_push_batch_size = state['train_push_batch_size']
img_size = state['img_size']
num_data_workers = state['workers']
num_classes = state['num_classes']
proto_per_class = state['proto_per_class']
prototype_activation_function = state['prototype_activation_function']
add_on_layers_type = state['add_on_layers_type']
proto_channels = state.get('proto_channels', 128)
prototype_shape = (num_classes * proto_per_class, proto_channels, 1, 1)

normalize = transforms.Normalize(mean=state['train_mean'],
                                 std=state['train_std'])

# all datasets
# train set
if 'MNIST' in dataset:
    from torchvision.datasets import MNIST

    def toRGBtorch(x):
        # C x H x W
        return x.repeat_interleave(3, 0)

    preprocess_input_function = None
    train_dataset = MNIST(root='./data/mnist', 
                          train=True, 
                          transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                          download=True)
    train_push_dataset = MNIST(root='./data/mnist', 
                               train=True, 
                               transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]), 
                               download=True)
    test_dataset = MNIST(root='./data/mnist', 
                         train=False, 
                         transform=transforms.Compose([transforms.ToTensor(), toRGBtorch]),
                         download=True)
elif 'CIFAR10' in dataset:
    from torchvision.datasets import CIFAR10

    preprocess_input_function = None
    train_dataset = CIFAR10(root='./data/cifar10', 
                          train=True, 
                          transform=transforms.Compose([transforms.ToTensor()]), 
                          download=True)
    train_push_dataset = CIFAR10(root='./data/cifar10', 
                               train=True, 
                               transform=transforms.Compose([transforms.ToTensor()]), 
                               download=True)
    test_dataset = CIFAR10(root='./data/cifar10', 
                         train=False, 
                         transform=transforms.Compose([transforms.ToTensor()]),
                         download=True)

else:
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ])
    )
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=num_data_workers, pin_memory=False
)
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=num_data_workers, pin_memory=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_data_workers, pin_memory=False
)
    
# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log.info('training set size: {0}'.format(len(train_loader.dataset)))
log.info('push set size: {0}'.format(len(train_push_loader.dataset)))
log.info('test set size: {0}'.format(len(test_loader.dataset)))
log.info('batch size: {0}'.format(train_batch_size))

if state['base_cnn_ckpt'] is not None:
    log.info(f"Load base CNN checkpoint from {state['base_cnn_ckpt']}")
    
# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              pretrain_ckpt=state['base_cnn_ckpt'])

#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
# from ProtoPNet.settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_lrs = state['joint_optimizer_lrs']
joint_lr_step_size = state['joint_lr_step_size']

joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

# from ProtoPNet.settings import warm_optimizer_lrs
warm_optimizer_lrs = state['warm_optimizer_lrs']
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

# from ProtoPNet.settings import last_layer_optimizer_lr
last_layer_optimizer_lr = state['last_layer_optimizer_lr']
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
# from ProtoPNet.settings import coefs
coefs = state['coefs']

# number of training epochs, number of warm epochs, push start epoch, push epochs
# from ProtoPNet.settings import num_train_epochs, num_warm_epochs, push_start, push_epochs
num_train_epochs = state['train_epochs']
num_warm_epochs = state['warm_epochs']
push_start = state['push_start']
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

# train the model
log.info('start training')
import copy

best_acc = 0

for epoch in range(num_train_epochs):
    log.info('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log.info)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log.info)
    else:
        tnt.joint(model=ppnet_multi, log=log.info)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log.info)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log.info)
    save.save_model_w_condition(model=ppnet, model_dir=state['save_dir'], model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log.info)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log.info)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log.info)
        save.save_model_w_condition(model=ppnet, model_dir=state['save_dir'], model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log.info)
        if accu > best_acc:
            save.save_best(model=ppnet, model_dir=state['save_dir'], accu=accu, epoch=epoch, log=log.info)
            best_acc = accu
            
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log.info)
            for i in range(20):
                log.info('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log.info)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log.info)
                save.save_model_w_condition(model=ppnet, model_dir=state['save_dir'], model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log.info)
                if accu > best_acc:
                    save.save_best(model=ppnet, model_dir=state['save_dir'], accu=accu, epoch=epoch, log=log.info)
                    best_acc = accu

