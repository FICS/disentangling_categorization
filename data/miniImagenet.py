import os
import sys
import socket
import threading
import glob
import pandas as pd
import argparse
import logging
import torch

from datetime import datetime
from tqdm import tqdm

project_root = os.environ.get('DISENT_ROOT', '')
sys.path.append(project_root)

from data_helpers import command, gdown_maybe, check_hash, HashError, PipelineError

opt = argparse.ArgumentParser()
opt.add_argument('--start-over', 
                 default=False, required=False, action='store_true',
                 help='Ignore progress so far and start the data generation from first step.')
opt.add_argument('--seed', 
                 default=100, type=int,
                 help='Random seed for this data pipeline.')
opt = vars(opt.parse_args())


DATA_ROOT = os.environ.get('DATA_ROOT', '')
conda_env = 'disent'
dataset = 'miniImagenet'
seed = opt['seed']
################################

dataset_root = os.path.join(DATA_ROOT, 'dataset')
if not os.path.exists(dataset_root):
    os.makedirs(dataset_root)


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler(os.path.join(dataset_root, f'{dataset}.log')),
                        logging.StreamHandler()
                             ],
                    level=logging.DEBUG)
log = logging.getLogger(f'{dataset}')

log.info(f"env:DISENT_ROOT={project_root}")
log.info(f"env:DATA_ROOT={DATA_ROOT}")


progress_filep = os.path.join(dataset_root, f'{dataset}_stages_done.pth')

if opt.get('start-over', False):
    log.debug(f"User option: start over from scratch.")
    progress = 0
else:
    try:
        progress = torch.load(progress_filep)
        log.debug(f"Found previous script progress at {progress_filep}")
        log.debug(f"Starting with {progress} stage(s) done.")
    except FileNotFoundError:
        log.debug(f"No progress found at {progress_filep}, starting from scratch.")
        progress = 0
        
    
################################

def command_wrapper(execute_string):
    log.debug("> " + execute_string)
    ret = command(execute_string, log_path=os.path.join(dataset_root, f'{dataset}.log'))
    if ret != 0:
        raise PipelineError
        
    return ret


try:
    log.info(f"\tPreparing miniImagenet with seed={seed}")
    
    if progress < 1:
        log.info(f"\tDownload val...")
        gdown_maybe("https://drive.google.com/uc?id=1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl", 
                     os.path.join(dataset_root, 'miniImageNet/val.tar'), log=log.info)
        check_hash("72ada64fab64c26a00250bfa622d608efd2f5cf7720b11bc936103498dbaf314", 
                   os.path.join(dataset_root, 'miniImageNet/val.tar'), log=log.info)
        progress +=1 
    
    if progress < 2:
        log.info(f"\tDownload train...")
        gdown_maybe("https://drive.google.com/uc?id=107FTosYIeBn5QbynR46YG91nHcJ70whs", 
                     os.path.join(dataset_root, 'miniImageNet/train.tar'), log=log.info)
        check_hash("376419f799ffe00216622461ac08355f6175d1873ca495eb1d876dc2b8aad429", 
                   os.path.join(dataset_root, 'miniImageNet/train.tar'), log=log.info)
        progress +=1 
        
    if progress < 3:
        log.info(f"\tDownload test...")
        gdown_maybe("https://drive.google.com/uc?id=1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v", 
                     os.path.join(dataset_root, 'miniImageNet/test.tar'), log=log.info)
        check_hash("58cc4c9f68afe15be87de75928c2d0b7d93d00b337a4472d3d9aad2ea671fa4d", 
                   os.path.join(dataset_root, 'miniImageNet/test.tar'), log=log.info)
        progress +=1 
        
    
    if progress < 4:
        log.info(f"\tUnpack...")
        ret = command_wrapper("for f in $(ls $DATA_ROOT/dataset/miniImageNet ); do tar -xf $DATA_ROOT/dataset/miniImageNet/$f -C $DATA_ROOT/dataset/miniImageNet/; done")
        progress +=1 
    
    
    if progress < 5:
        # unpack
        log.info(f"\tPreparing resize for miniImagenet")
        ret = command_wrapper(f"conda run --no-capture-output -n {conda_env} python $DISENT_ROOT/data/resize_miniImagenet.py")
        progress +=1 

    if progress < 6:
        # statistic
        log.info(f"\tCalculate statistic")
        ret = command_wrapper(f"conda run --no-capture-output -n {conda_env} python $DISENT_ROOT/data/dataset_statistics.py --seed {seed} --dataset {dataset}")
        progress +=1 
    
    if progress < 7:
        # CW class sets
        log.info(f"\tGenerate CW sets")
        ret = command_wrapper(f"conda run --no-capture-output -n {conda_env} python $DISENT_ROOT/data/subsets_miniImagenet.py")
        progress +=1 
    
    torch.save(progress, progress_filep)
    log.info(f"Done with data generation at {DATA_ROOT}/dataset/miniImageNet")
    
except KeyboardInterrupt:
    log.info(f"Early exit")
except PipelineError:
    log.info(f"An error occured in the data generation pipeline, stopping early.")
except HashError as e:
    log.exception(e)
    log.info(f"Aborting program.")
except Exception as e:
    log.exception(e)
finally:
    torch.save(progress, progress_filep)
