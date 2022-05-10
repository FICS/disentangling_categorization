import os
import sys
sys.path.append(os.environ.get('DISENT_ROOT'))

import shutil
import random

from distutils.dir_util import copy_tree



opt = {
    'from_dir': os.path.join(os.environ.get("DATA_ROOT"), "dataset/miniImageNet-custom/")
}


"""
create CW structure for miniImagenet or FC100
sets that play nicely with CW rotation matrix loaders. 
"""
print("Start transfers...")
data_path = opt['from_dir']

'''
├── concept_train
│   ├── airplane
│   │   ├── airplane
│   ├── bed
│   │   ├── bed
│   └── ......
'''
original_lookup = {
    'concept_train': 'train',
    'concept_test': 'test'
}
for sub_dir in ['concept_train', 'concept_test']:
    from_k_root = os.path.join(opt['from_dir'], original_lookup[sub_dir])
    assert os.path.isdir(from_k_root), f"Nothing found at {from_k_root}"

    klass = sorted(os.listdir(from_k_root))
    k_root = os.path.join(data_path, sub_dir)
    if not os.path.isdir(k_root):
        os.makedirs(k_root)
    from_k_root = os.path.join(opt['from_dir'], original_lookup[sub_dir])
    assert os.path.isdir(from_k_root), f"Nothing found at {from_k_root}"

    for kd in klass:
        concept_dir = os.path.join(k_root, kd)
        if not os.path.isdir(concept_dir):
            os.makedirs(concept_dir)

        from_k_root_klass = os.path.join(from_k_root, kd)
        k_root_klass = os.path.join(concept_dir, kd)
        if not os.path.isdir(k_root_klass):
            print(f"{from_k_root_klass} -> {k_root_klass}")
            shutil.copytree(from_k_root_klass, k_root_klass)


print("Done.")