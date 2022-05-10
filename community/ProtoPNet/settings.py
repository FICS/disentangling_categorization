import os

from .helpers import get_time_stamp

# choices=['cub200', 'fc100', 'c100', 'miniImagenet']
dataset = 'cub200'

cwd = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.environ.get('DATA_ROOT', None)
if DATA_ROOT is None:
    raise IOError("Please set DATA_ROOT environment variable in your shell: $ export DATA_ROOT=path")
    

# base_architecture = 'vgg16'
# base_architecture = 'vgg16_bn'
# base_architecture = 'vgg19'
# base_architecture = 'vgg19_bn'
# base_architecture = 'resnet152'
base_architecture = 'resnet50'
# base_architecture = 'densenet121'
# base_architecture = 'densenet161'
# print(f"settings: using base architecture {base_architecture}")

img_size = 224

# num_classes = 10
proto_per_class = 10
experiment_run = get_time_stamp()

# always use ImageNet mean/std since we recycle pretrained VGG
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
    
    
if dataset == 'cub200':
    num_classes = 10
    
    if num_classes != 200:
        ckpt_dir = os.path.join(DATA_ROOT, f"ckpt/proto-{num_classes}c")
        data_path = os.path.join(DATA_ROOT, f'dataset/cub200_cropped_{num_classes}c/')
    else:
        ckpt_dir = os.path.join(DATA_ROOT, "ckpt/proto")
        data_path = os.path.join(DATA_ROOT, 'dataset/cub200_cropped/')

    train_dir = os.path.join(data_path, 'train_cropped_augmented/')
    test_dir = os.path.join(data_path, 'test_cropped/')
    train_push_dir = os.path.join(data_path, 'train_cropped/')
    
elif dataset == 'fc100':
    num_classes = 60  # 100 total, 60 for meta traininng
    
    ckpt_dir = os.path.join(DATA_ROOT, f"ckpt/proto-fc100")
    data_path = os.path.join(DATA_ROOT, 'dataset/Fewshot-CIFAR100-224px/')
    
    train_dir = os.path.join(data_path, 'train/')
    test_dir = os.path.join(data_path, 'train/') # no supervised test set
    train_push_dir = os.path.join(data_path, 'train/')
#     mean = [0.4413784,  0.48683313, 0.50770426]
#     std = [0.19826244, 0.19426456, 0.19649212]

elif dataset == 'miniImagenet':
    num_classes = 64  # 100 total, 64 for meta traininng
    
    ckpt_dir = os.path.join(DATA_ROOT, f"ckpt/proto-miniImageNet")
    data_path = os.path.join(DATA_ROOT, 'dataset/miniImageNet-custom/')
    
    train_dir = os.path.join(data_path, 'train/')
    test_dir = os.path.join(data_path, 'train/')  # no supervised test set
    train_push_dir = os.path.join(data_path, 'train/')
    
#     mean = [0.4732501,  0.44906873, 0.40405995]
#     std = [0.21175125, 0.20798777, 0.20849589]
    
elif dataset == 'mnist':
    num_classes = 10
    
    ckpt_dir = os.path.join(DATA_ROOT, "ckpt/proto-mnist")
    data_path = os.path.join(DATA_ROOT, 'dataset/MNIST')

    train_dir = os.path.join(data_path, 'train_cropped_augmented/')
    test_dir = os.path.join(data_path, 'test_cropped/')
    train_push_dir = os.path.join(data_path, 'train_cropped/')

    mean = []
    std = []
    

prototype_shape = (num_classes * proto_per_class, 128, 1, 1)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

# train_batch_size = 80
# train_batch_size = 320
train_batch_size = 256
test_batch_size = 256
train_push_batch_size = 256

# train_batch_size = 240
# test_batch_size = 100
# train_push_batch_size = 75

num_data_workers = 64

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

