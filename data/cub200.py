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

from data_helpers import command, urllib_maybe, check_hash, HashError, PipelineError

opt = argparse.ArgumentParser()
opt.add_argument('--start_over', 
                 action='store_true',
                 help='Ignore progress so far and start the data generation from first step.')
opt.add_argument('--seed', 
                 default=100, type=int,
                 help='Random seed for this data pipeline.')
opt = vars(opt.parse_args())


DATA_ROOT = os.environ.get('DATA_ROOT', '')
conda_env = 'disent'
dataset = 'cub200'
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

if opt.get('start_over', False):
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
    log.info(f"\tPreparing augmented cub200 with seed={seed}")
    
    if progress < 1:
        # Download cub200
        log.info(f"\tDownload cub200...")
        urllib_maybe("https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/", 
                     os.path.join(dataset_root, 'CUB_200_2011.tgz'), log=log.info)
        check_hash("0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081", 
                   os.path.join(dataset_root, 'CUB_200_2011.tgz'), log=log.info)
        progress +=1 
    
    if progress < 2:
        # unpack
        log.info(f"\tPreparing cub200 with seed={seed}")
        ret = command_wrapper(r"tar -xzf $DATA_ROOT/dataset/CUB_200_2011.tgz -C $DATA_ROOT/dataset")
        progress +=1 
    
    if progress < 3:
        # crop
        log.info(f"\tcrop...")
        ret = command_wrapper(f"conda run --no-capture-output -n {conda_env} python $DISENT_ROOT/data/crop_cub200.py")
        progress +=1 
    
    if progress < 4:
        # prepare
        log.info(f"\tPrepare custom folder...")
        ret = command_wrapper(f"mkdir -p $DATA_ROOT/dataset/cub200_cropped_seed={seed}")
        progress +=1 
    
    if progress < 5:
        # move
        log.info(f"\tMove data...")
        ret = command_wrapper(f"rsync -avh --info=progress2 $DATA_ROOT/dataset/CUB_200_2011/cropped_split/train/ $DATA_ROOT/dataset/cub200_cropped_seed={seed}/train_cropped")
        ret = command_wrapper(f"rsync -avh --info=progress2 $DATA_ROOT/dataset/CUB_200_2011/cropped_split/test/ $DATA_ROOT/dataset/cub200_cropped_seed={seed}/test_cropped")
        progress +=1 
    
    if progress < 6:
        # augment
        log.info(f"\tStart augmentation, it may take up to 2 hours depending on storage medium and CPU.")
        ret = command_wrapper(f"conda run --no-capture-output -n {conda_env} python $DISENT_ROOT/data/augment_cub200.py --seed {seed}")
        progress +=1 

    if progress < 7:
        # statistic
        log.info(f"\tCalculate statistic")
        ret = command_wrapper(f"conda run --no-capture-output -n {conda_env} python $DISENT_ROOT/data/dataset_statistics.py --seed {seed} --dataset {dataset}")
        progress +=1 
    
    if progress < 8:
        # 10 class subsets
        log.info(f"\tGenerate subsets")
        ret = command_wrapper(f"conda run --no-capture-output -n {conda_env} python $DISENT_ROOT/data/subsets_cub200.py --seed {seed}")
        progress +=1 
    
    torch.save(progress, progress_filep)
    log.info(f"Done with data generation at {DATA_ROOT}/dataset/cub200_cropped_seed={seed}")
    
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
