import os
import sys
sys.path.append(os.environ.get('DISENT_ROOT'))

import argparse
import shutil
import random
from tqdm import tqdm
from distutils.dir_util import copy_tree

import concurrent.futures

opt = argparse.ArgumentParser()
opt.add_argument('--seed', 
                 default=100, type=int,
                 help='Seed from random augmentation.')
opt = vars(opt.parse_args())

opt['from_dir'] = os.path.join(os.environ.get('DATA_ROOT'), f"dataset/cub200_cropped_seed={opt['seed']}/")
opt['base_dir'] = os.path.join(os.environ.get('DATA_ROOT'), "dataset/")
            
            
for seed in range(0, 5):
    def copy_job(num_classes):
        data_path = os.path.join(opt['base_dir'], f'cub200_cropped_{num_classes}c_seed={seed}')
        nc_klass = klass[:num_classes]

        for sub_dir in ['test_cropped', 'train_cropped', 'train_cropped_augmented']:
            k_root = os.path.join(data_path, sub_dir)
            if not os.path.isdir(k_root):
                os.makedirs(k_root)
            from_k_root = os.path.join(opt['from_dir'], sub_dir)
            assert os.path.isdir(from_k_root), f"Nothing found at {from_k_root}"
            nc_klass = klass[:num_classes]
            for kd in nc_klass:
                from_k_root_klass = os.path.join(from_k_root, kd)
                k_root_klass = os.path.join(k_root, kd)
                shutil.copytree(from_k_root_klass, k_root_klass)
            
            
    from_k_root = os.path.join(opt['from_dir'], 'train_cropped')
    assert os.path.isdir(from_k_root), f"Nothing found at {from_k_root}"
    klass = sorted(os.listdir(from_k_root))
    random.Random(seed).shuffle(klass)
    # print(klass[:10])
    
    print("Start regular transfers...")
    for num_classes in [10]:
        copy_job(num_classes)

    # CW
    def copy_job(num_classes):
        data_path = os.path.join(opt['base_dir'], f'cub200_cropped_{num_classes}c_seed={seed}')
        nc_klass = klass[:num_classes]

        for sub_dir in ['test_cropped', 'train_cropped', 'train_cropped_augmented']:
            k_root = os.path.join(data_path, sub_dir)
            if not os.path.isdir(k_root):
                os.makedirs(k_root)
            from_k_root = os.path.join(opt['from_dir'], sub_dir)
            assert os.path.isdir(from_k_root), f"Nothing found at {from_k_root}"
            nc_klass = klass[:num_classes]
            for kd in nc_klass:
                from_k_root_klass = os.path.join(from_k_root, kd)
                k_root_klass = os.path.join(k_root, kd)
                if not os.path.isdir(k_root_klass):
                    print(f"{from_k_root_klass}->{k_root_klass}")
                    shutil.copytree(from_k_root_klass, k_root_klass)
                
        '''
        ├── concept_train
        │   ├── airplane
        │   │   ├── airplane
        │   ├── bed
        │   │   ├── bed
        │   └── ......
        '''
        original_lookup = {
            'concept_train': 'train_cropped_augmented',
            'concept_test': 'test_cropped'
        }
        for sub_dir in ['concept_train', 'concept_test']:
            k_root = os.path.join(data_path, sub_dir)
            if not os.path.isdir(k_root):
                os.makedirs(k_root)
            from_k_root = os.path.join(opt['from_dir'], original_lookup[sub_dir])
            assert os.path.isdir(from_k_root), f"Nothing found at {from_k_root}"

            nc_klass = klass[:num_classes]
            for kd in nc_klass:
                concept_dir = os.path.join(k_root, kd)
                if not os.path.isdir(concept_dir):
                    os.makedirs(concept_dir)

                from_k_root_klass = os.path.join(from_k_root, kd)
                k_root_klass = os.path.join(concept_dir, kd)
                if not os.path.isdir(k_root_klass):
                    # print(f"{from_k_root_klass}->{k_root_klass}")
                    shutil.copytree(from_k_root_klass, k_root_klass)
                
    print("Start CW transfers...")
    for num_classes in [10]:
        copy_job(num_classes)

    print("Done.")