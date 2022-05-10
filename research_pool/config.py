import numpy as np
import os

SEED = 9
# Seed any frameworks you use during cascading
np.random.seed(SEED)
# experiments will have their own seeds

# Absolute path to project and data root on every node. 
project_root = os.environ.get('DISENT_ROOT', None)
DATA_ROOT = os.environ.get('DATA_ROOT', None)  # added to paths in scripts. 
SAVE_ROOT = os.environ.get('SAVE_ROOT', None)  # added to paths in scripts. 
if DATA_ROOT is None:
    raise IOError("Please set DATA_ROOT environment variable in your shell using installation directions.")
if SAVE_ROOT is None:
    raise IOError("Please set SAVE_ROOT environment variable in your shell using installation directions.")
if project_root is None:
    raise IOError("Please set DISENT_ROOT environment variable in your shell using installation directions.")
   
    
print(f"env:DISENT_ROOT={project_root}")
print(f"env:DATA_ROOT={DATA_ROOT}")
print(f"env:SAVE_ROOT={SAVE_ROOT}")

conda_environ = "disent"

# Commands to run before starting experiments
global_shell_commands_before = [
    f"cd {project_root}",
    f"conda activate {conda_environ}",
    f"export DISENT_ROOT={project_root}",
    f"export DATA_ROOT={DATA_ROOT}",
    f"export SAVE_ROOT={SAVE_ROOT}",
    f"export PYTHONPATH={project_root}:{project_root}/ProtoPNet/",
]

experiment_groups = ["SEMIOSIS"]

# ============== Experiment groups definitions ==============

# =================== Params for training experiments ===================
dataset_choices=['cub10', 'miniImagenet']
# num classes =.   10           64
dataset_k = {
    'cub10': 10, 
    'cub200': 200, 
    'fc100': 60, 
    'miniImagenet': 64,
}

tuple_to_train_dir = {
    'cub2': "dataset/cub200_cropped_2c/train_cropped_augmented",
    'cub10': "dataset/cub200_cropped_10c/train_cropped_augmented",
    'cub200': "dataset/cub200_cropped/train_cropped_augmented",
    'fc100': "dataset/Fewshot-CIFAR100-224px/train",
    'miniImagenet': "dataset/miniImageNet-custom/train",
}

tuple_to_train_push_dir = {
    'cub2': "dataset/cub200_cropped_2c/train_cropped",
    'cub10': "dataset/cub200_cropped_10c/train_cropped",
    'cub200': "dataset/cub200_cropped/train_cropped",
    'fc100': "dataset/Fewshot-CIFAR100-224px/train",
    'miniImagenet': "dataset/miniImageNet-custom/train",
}

tuple_to_test_dir = {
    'cub2': "dataset/cub200_cropped_2c/test_cropped",
    'cub10': "dataset/cub200_cropped_10c/test_cropped",
    'cub200': "dataset/cub200_cropped/test_cropped",
    'fc100': "dataset/Fewshot-CIFAR100-224px/test",
    'miniImagenet': "dataset/miniImageNet-custom/test",
}

# CW folders (clone_dataset.ipynb)
tuple_to_concept_train = {
    'cub10': "dataset/cub200_cropped_10c/concept_train",
    'miniImagenet': "dataset/miniImageNet-custom/concept_train",
    'fc100': "dataset/Fewshot-CIFAR100-224px/concept_train",
}

tuple_to_concept_test = {
    'cub10': "dataset/cub200_cropped_10c/concept_test",
    'miniImagenet': "dataset/miniImageNet-custom/concept_test",
    'fc100': "dataset/Fewshot-CIFAR100-224px/concept_test",
}

seed_cub10_lookup = {}

# generator seed lookup logic, it tells us which directory (dataset subset) for each seed. 
cub10_base = "dataset/cub200_cropped_10c"
for seed in list(range(0, 5)):
    seed_cub10_lookup[(seed, 'train')] = f"{cub10_base}_seed={seed}/train_cropped_augmented"
    seed_cub10_lookup[(seed, 'train_push')] = f"{cub10_base}_seed={seed}/train_cropped"
    seed_cub10_lookup[(seed, 'test')] = f"{cub10_base}_seed={seed}/test_cropped"
    seed_cub10_lookup[(seed, 'concept_train')] = f"{cub10_base}_seed={seed}/concept_train"
    seed_cub10_lookup[(seed, 'concept_test')] = f"{cub10_base}_seed={seed}/concept_test"

# dataset_statistics.py
tuple_to_mean = {
    'cub2': [0.47827587, 0.49928078, 0.5191471 ],
    'cub10': [0.45519164, 0.45833948, 0.41331828],
    'cub200': "",
    'fc100': [0.45672116, 0.4959486,  0.51666486],
    'miniImagenet': "",
    'Imagenet': [0.485, 0.456, 0.406],
}
tuple_to_std = {
    'cub2': [0.2657977,  0.25875443, 0.28014705],
    'cub10': [0.23767576, 0.23529641, 0.26368567],
    'cub200': "",
    'fc100': [0.2843761,  0.25997803, 0.27078542],
    'miniImagenet': "",
    'Imagenet': [0.229, 0.224, 0.225],
}

cnn_percept_input_dims = {
    'vgg16': 512,
    'vgg16_bn': 512,
    'vgg19': 512,
    'vgg19_bn': 512,
    'resnet18': 512,
    'resnet50': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet161': 2208,
    'densenet201': 1920,
}


# autogen tuple db (research_pool/ckpt_wrangler.ipynb)
# each loaded object will have indices 
#     proto class models: [semiosis seed][(sender|receiver, proto per class, arch)]
#     other models: [semiosis seed][(sender|receiver, arch)]
percept_to_db = {}
percept_to_db['cub10'] = {
    'ProtoPNet' : 'ckpt/autotrain/ptdb/cub10/011322-150705/ProtoPNet.pickle',
    'ConvNet'   : 'ckpt/autotrain/ptdb/cub10/011322-150705/ConvNet.pickle',
    'CW'        : 'ckpt/autotrain/ptdb/cub10/011322-150705/CW.pickle',
}
percept_to_db['miniImagenet'] = {
    # resnet50 10x
    'ProtoPNet' : 'ckpt/autotrain/ptdb/miniImagenet/111221-014223/ProtoPNet.pickle',
    'ConvNet'   : 'ckpt/autotrain/ptdb/miniImagenet/111221-014223/ConvNet.pickle',
    'CW'        : 'ckpt/autotrain/ptdb/miniImagenet/111221-014223/CW.pickle',
}


class SEMIOSIS(object):
    def __init__(self):
        self.experiment_group = str(self.__class__.__name__)
        # disent: semiotic_social_1game
        # reification AE: reification_1game
        self.experiment = ['semiotic_social_1game']
        self.seed = list(range(0, 5))  # CHECK
        self.epochs = 12  # CHECK
        self.save_root = os.path.join("analysis/proto", 
                                      str(self.__class__.__name__))
        self.dataset = ["cub10"]  # choices=['cub10', 'miniImagenet']
        self.num_classes = [dataset_k[d] for d in self.dataset]
        self.reload_agents = None
        # Restart will try to restart from the save directory
        self.restart = 0
        self.num_data_workers = 4

        # approach agnostic params
        # self.num_classes = [2]  # CHECK
        self.separated = [True]
        
        self.distractors = [5]  # CHECK
        self.hidden_dims = [256]
        # self.hidden_dims = [64]
        self.max_lens = [4]  # >= K * pK
        # self.max_lens = [10, 20]
        self.embed_dims = [64]
        # self.embed_dims = [32, 128, 256, 512]
        self.length_costs = [0.0]
        self.topk = [11] # CHECK topk > max_len
        self.semiosis_start = [1]  # CHECK
        self.semiotic_sgd_epochs = [
            # One-to-one with semiotic_push_epochs
            [],
            # Examples:
            # [3, 4, 5, 6],
            # [*list(range(10, 20))],
            # [5, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
            # [5, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
            # [*list(range(10, 20)), *list(range(40, 50))],
        ]
        assert len(self.semiotic_sgd_epochs) >= 1
        self.features_lr = [1e-4]
        self.add_on_layers_lr = [3e-3] 
        self.prototype_vectors_lr = [3e-3]
        self.last_layer_lr = [0.01]  
        # epochs to perform ProtoPNet push operation or update CW rotation
        self.semiotic_push_epochs = [
            [],
            # Examples (following semiotic_sgd_epochs):
            # [],
            # [15, 20],
            # [10, 13, 16, 19, 22, 25, 28, 31, 34, 37],
            # [],
            # [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        ]
        assert len(self.semiotic_push_epochs) >= 1
        
        self.class_specific = [True]
        self.sign_coef =  [0.9]
        self.social_coef = [0.1]
        self.concept_classifier_epochs = [5]  # CHECK epochs for classifier convex optimization 
        
        self.gs_st = [1]
        self.learnable_temperature = [1]
        # self.learnable_temperature = [1]
        
        self.sender_cells = ["lstm"]
        # Q1/Q2: ["CwWrapper", "ProtoWrapper", "CnnBWrapper"] 
        # Q3 Proto: ["ProtoWrapper", "CnnBWrapper"] 
        # Q3 CW: ["CwWrapper", "CnnBWrapper"] 
        # Q4: ["ProtoWrapper"]
        self.sender_percept_arch = ["ProtoWrapper"]#, "CnnBWrapper"] 
        # Q1/Q2: ["RnnSenderGS"]
        # Q3: ['MultiHeadRnnSenderGS2', 'FLRnnSenderGS', 'OLRnnSenderGS']
        # Q4: ["FLRnnSenderGS"]
        self.sender_arch = ['RnnSenderGS']
        # Q1/Q2: ["CwWrapper", "ProtoWrapper", "CnnBWrapper"] 
        # Q3 Proto: ["ProtoWrapper", "CnnBWrapper"] 
        # Q3 CW: ["CwWrapper", "CnnBWrapper"] 
        # Q4: ["ProtoWrapper"]
        self.recv_percept_arch =  ["ProtoWrapper"]#, "CnnBWrapper"] 
        # Q1/Q2: ["RnnReceiverGS"]
        # Q3: ['FLRnnReceiverGS']
        # Q4: ["FLRnnReceiverGS"]
        self.recv_arch = ["RnnReceiverGS"]
        # Q1/Q2: [100]
        # Q3 CW: [10]
        # Q3 Proto: [100]
        # Q4: [100]
        self.vocab_size = [10] # CHECKING
        
        # self.sender_lr = [0.005]  # CHECK
        self.sender_lr = [0.001]
        self.receiver_cells = ["lstm"]
        # self.receiver_lr = [0.0001]
        # self.receiver_lr = [0.01]  # CHECK
        self.receiver_lr = [0.001]  
        self.tuple_to_train_dir = tuple_to_train_dir
        self.tuple_to_train_push_dir = tuple_to_train_push_dir
        self.tuple_to_test_dir = tuple_to_test_dir
        # CW
        self.tuple_to_concept_train = tuple_to_concept_train
        self.tuple_to_concept_test = tuple_to_concept_test
        self.tuple_to_train_batch_size = {
            'cub2': 8,
            'cub10': 128,
            'cub200': 128,
            'fc100': 64,
            'miniImagenet': 64,
            'MNIST': 256,
        }
        
        self.tuple_to_test_batch_size = {
            'cub2': 8,
            'cub10': 8,
            'cub200': 32,
            'fc100': 64,
            'miniImagenet': 64,
            'MNIST': 256,
        }
        self.image_sizes = {
            'cub2': 224,
            'cub10': 224,
            'cub200': 224,
            'fc100': 224,
            'miniImagenet': 224,
            'MNIST': 28,
        }
        
        # baseline
        self.aux_losses_baseline = [
            [],
            # ['least_effort'],
            # ['least_effort'],
            # ['least_effort'],
        ]
        self.aux_weights_baseline = [
            [],
            # [0.01],
            # [0.01],
            # [0.5],
        ]
        # self.baseline_vocab_sizes = [101, 1000]
        self.sender_base_cnn = ['jalal_bn']
        self.recv_base_cnn = ['jalal_bn']
        self.cnn_percept_input_dims = cnn_percept_input_dims
        self.cnn_pretrained = [True]
        
        # Proto
        # self.prototypes_per_class = 2  # CHECK
        self.sender_prototypes_per_class = [1, 10, 100]
        self.recv_prototypes_per_class = [1, 10, 100]
        self.aux_losses_proto = [
            [],
            # ['least_effort', 'bosdis'],
            # ['least_effort'],
            # ['least_effort'],
            # ['least_effort'],
        ]
        self.aux_weights_proto = [
            [],
            # [0.5, (10, 0.5)],
            # [0.01],
            # [0.01],
            # [0.5],
        ]
        self.tuple_to_mean = tuple_to_mean
        self.tuple_to_std = tuple_to_std
        self.percept_to_db = percept_to_db
        self.seed_cub10_lookup = seed_cub10_lookup
    
        
percpt_seed = list(range(0, 2*5))


class PROTOPNET(object):
    def __init__(self):
        self.experiment_group = str(self.__class__.__name__)
        self.seed = percpt_seed  # CHECK
        self.save_root = os.path.join("ckpt/autotrain/", 
                                      str(self.__class__.__name__))
        self.dataset = ["MNIST"]  # choices=['cub200', 'fc100', 'c100', 'miniImagenet']
        self.num_classes = [dataset_k[d] for d in self.dataset]
        self.base_architecture = ["jalal_bn"]
        self.proto_per_class =  [1, 3, 5, 10]
        self.tuple_to_train_dir = tuple_to_train_dir
        self.tuple_to_train_push_dir = tuple_to_train_push_dir
        self.tuple_to_test_dir = tuple_to_test_dir
        self.tuple_to_train_batch_size = {
            'cub2': 8,
            'cub10': 256,
            'cub200': 256,
            'fc100': 256,
            'miniImagenet': 256,
            'MNIST': 256,
            'CIFAR10': 256,
        }
        self.tuple_to_test_batch_size = {
            'cub2': 8,
            'cub10': 256,
            'cub200': 256,
            'fc100': 256,
            'miniImagenet': 256,
            'MNIST': 256,
            'CIFAR10': 256,
        }
        self.tuple_to_mean = tuple_to_mean
        self.tuple_to_std = tuple_to_std
        self.seed_cub10_lookup = seed_cub10_lookup
        self.proto_channels = [64] # MNIST only for now


# proto net with baseline CNN 
class PROTOPNETB(PROTOPNET):
    def __init__(self):
        super(PROTOPNETB, self).__init__()
        self.experiment_group = str(self.__class__.__name__)
        self.save_root = os.path.join("ckpt/autotrain/", 
                                      str(self.__class__.__name__))
        self.tuple_to_CnnB_model = tuple_to_CnnB_model
        self.seed_cub10_lookup = seed_cub10_lookup
        

class CONVNET(object):
    def __init__(self):
        self.experiment_group = str(self.__class__.__name__)
        self.seed = percpt_seed  # CHECK
        self.save_root = os.path.join("ckpt/autotrain/", 
                                      str(self.__class__.__name__))
        self.dataset = ["MNIST"]  # choices=['cub200', 'fc100', 'c100', 'miniImagenet']
        self.num_classes = [dataset_k[d] for d in self.dataset]
        self.base_architecture = ["jalal_bn"]
        self.tuple_to_train_dir = tuple_to_train_dir
        self.tuple_to_test_dir = tuple_to_test_dir
        self.tuple_to_train_batch_size = {
            'cub2': 8,
            'cub10': 64,
            'cub200': 64,
            'fc100': 64,
            'miniImagenet': 64,
            'MNIST': 256,
            'CIFAR10': 256,
        }
        self.tuple_to_test_batch_size = {
            'cub2': 8,
            'cub10': 64,
            'cub200': 64,
            'fc100': 64,
            'miniImagenet': 32,
            'MNIST': 256,
            'CIFAR10': 256,
        }
        self.tuple_to_mean = tuple_to_mean
        self.tuple_to_std = tuple_to_std
        self.seed_cub10_lookup = seed_cub10_lookup
        
        
tuple_to_base_cnn_ckpt = {
    ('cub10', 'resnet50'): "ckpt/autotrain/CONVNET/cub10/102921-232752/seed-0/resnet50/0/best.pth",
    ('cub10', 'vgg16_bn'): "ckpt/autotrain/CONVNET/cub10/103021-032401/seed-0/vgg16_bn/0/best.pth",
    ('cub10', 'resnet18'): "ckpt/autotrain/CONVNET/cub10/102821-204643/seed-0/resnet18/0/best.pth",
    ('cub10', 'densenet161'): "ckpt/autotrain/CONVNET/cub10/103021-120453/seed-0/densenet161/0/best.pth",
    ('miniImagenet', 'vgg16_bn'): "ckpt/autotrain/CONVNET/miniImagenet/103121-194812/seed-1/vgg16_bn/0/best.pth",
    ('miniImagenet', 'resnet50'): "",
    ('fc100', 'vgg16_bn'): "ckpt/autotrain/CONVNET/fc100/110121-155023/seed-0/vgg16_bn/0/best.pth",
}


for k in list(tuple_to_base_cnn_ckpt.keys()):
    ds, arch = k
    tuple_to_base_cnn_ckpt[(ds, arch + '_cw')] = tuple_to_base_cnn_ckpt[(ds, arch)]
    tuple_to_base_cnn_ckpt[(ds, arch + '_baseline')] = tuple_to_base_cnn_ckpt[(ds, arch)]

cw_arch_to_wl = {
    'resnet50_cw': [16],
    'resnet18_cw': [8],
    'vgg16_bn_cw': [13], #13
    'densenet161_cw': [2], # 2
}
cw_arch_to_lr = {
    'resnet50_cw': 0.01,
    'resnet18_cw': 0.01,
    'vgg16_bn_cw': 0.001,
    'densenet161_cw': 0.01,
}



class CW(object):
    def __init__(self):
        self.experiment_group = str(self.__class__.__name__)
        self.seed = percpt_seed  # CHECK
        self.save_root = os.path.join("ckpt/autotrain/", 
                                      str(self.__class__.__name__))
        self.dataset = ["miniImagenet"]  # choices=['cub200', 'fc100', 'c100', 'miniImagenet']
        self.num_classes = [dataset_k[d] for d in self.dataset]
        self.base_architecture = ["vgg16_bn"]
        self.tuple_to_train_dir = tuple_to_train_dir
        self.tuple_to_concept_train = tuple_to_concept_train
        self.tuple_to_test_dir = tuple_to_test_dir
        self.tuple_to_val_dir = tuple_to_test_dir
        self.tuple_to_concept_test = tuple_to_concept_test
        self.tuple_to_train_batch_size = {
            'cub2': 8,
            'cub10': 64,
            'cub200': 64,
            'fc100': 64,
            'miniImagenet': 64,
        }
        self.tuple_to_test_batch_size = {
            'cub2': 8,
            'cub10': 64,
            'cub200': 64,
            'fc100': 64,
            'miniImagenet': 64,
        }
        self.whitened_layers = [
            cw_arch_to_wl[self.base_architecture[0]]
        ]
        self.act_mode = ["pool_max"]
        self.tuple_to_mean = tuple_to_mean
        self.tuple_to_std = tuple_to_std
        self.tuple_to_base_cnn_ckpt = tuple_to_base_cnn_ckpt
        self.arch_to_lr = cw_arch_to_lr
        self.arch_to_wl = cw_arch_to_wl
        self.seed_cub10_lookup = seed_cub10_lookup
        self.percept_to_db = percept_to_db
