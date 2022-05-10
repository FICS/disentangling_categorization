import os
import argparse
import shutil
import numpy as np
import logging
import pickle

from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
a_logger = logging.getLogger(__name__)


def pickle_write(fpath, obj):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)

    return obj


def augment(opt: dict, return_stats=False):
    if not os.path.isdir(opt['out_dir']):
        os.makedirs(opt['out_dir'])

    resize = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(336)
    ])
    jobs = []

    for sub_dir in ["train", "test", "val"]:
        if not os.path.isdir(os.path.join(opt['base_dir'], sub_dir)):
            continue
        
        out_split_dir = os.path.join(opt['out_dir'], sub_dir)
        
        k = os.listdir(os.path.join(opt['base_dir'], sub_dir))
        for from_class_dir in k:
            jobs.append((opt, out_split_dir, sub_dir, from_class_dir, None, resize))
            # augment_class_(opt, out_split_dir, sub_dir, from_class_dir, transform, resize)

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(augment_class_, *pack) for pack in jobs]

        for future in tqdm(as_completed(futures), total=len(jobs)):
            _, _ = future.result()


def augment_class_(opt, out_split_dir, sub_dir, from_class_dir, transform, resize):
    out_class_dir = os.path.join(out_split_dir, from_class_dir)
    xi_k = os.listdir(os.path.join(opt['base_dir'], sub_dir, from_class_dir))

    to_tensor = transforms.ToTensor()

    if not os.path.isdir(out_class_dir):
        os.makedirs(out_class_dir)

    for xi_name in xi_k:
        from_path = os.path.join(opt['base_dir'], sub_dir, from_class_dir, xi_name)
        if 'ipynb_checkpoints' in xi_name:
            a_logger.warning(f"Found bad file at {from_path}")
            continue
            
        xi = Image.open(from_path)

        to_path = os.path.join(out_class_dir, xi_name)

        if os.path.exists(to_path):
            try:
                xi_prime = Image.open(to_path)
            except UnidentifiedImageError as e:
                xi_prime = resize(xi)
                xi_prime.save(to_path)    
        else:
            xi_prime = resize(xi)
            xi_prime.save(to_path)

        # [0, 255] -> [0, 1]
        xi_prime_t = to_tensor(xi_prime).float()
        # Make sure we have 3 channels since we have to normalize later with other multi domain data. 

        if xi_prime_t.shape[0] == 1:
            xi_prime_t = xi_prime_t.repeat((3, 1, 1))

        D, H, W = xi_prime_t.shape
        assert D == 3, f"Inconsistent channels detected when writing {to_path}"

        
    return None, None


if __name__ == '__main__':
    opt = {}
    opt['base_dir'] = os.path.join(os.environ.get("DATA_ROOT"), f"dataset/miniImageNet/")
    opt['out_dir'] = os.path.join(os.environ.get("DATA_ROOT"), f"dataset/miniImageNet-custom/")
    
    augment(opt)
