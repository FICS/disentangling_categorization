import os
import sys
import argparse
import shutil
import numpy as np
import logging
import torch
import concurrent.futures
import Augmentor

from tqdm import tqdm

project_root = os.environ.get('DISENT_ROOT', '')
sys.path.append(project_root)

from data_helpers import quiet_command, PipelineError

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
a_logger = logging.getLogger(__name__)
    
    

if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--seed', 
                     default=100, type=int,
                     help='Seed for random augmentation.')
    opt = vars(opt.parse_args())
    
    datasets_root_dir = os.path.join(os.environ.get("DATA_ROOT"), f"dataset/cub200_cropped_seed={opt['seed']}/")
    
    if not os.path.exists(datasets_root_dir):
        raise IOError()
    
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    
    # dir = datasets_root_dir + 'train_cropped/'
    dir = os.path.join(datasets_root_dir, 'train_cropped')
    # target_dir = datasets_root_dir + 'train_cropped_augmented/'
    target_dir = os.path.join(datasets_root_dir, 'train_cropped_augmented')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        raise PipelineError(f"To avoid augmentation issues, please move or delete the existing folder: {target_dir}")
        
    folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

    for i in tqdm(range(len(folders))):
        fd = folders[i]
        tfd = target_folders[i]
        
        ret = quiet_command(f"conda run --no-capture-output -n disent python $DISENT_ROOT/data/augment_folder.py {fd} {tfd} --seed {opt['seed']}")
        if ret != 0:
            raise PipelineError("Got nono-zero exit code from augment_folder.py! Check traceback or try running manually to debug the issue."
                                f"Params: fd={fd}, tfd={tfd}")