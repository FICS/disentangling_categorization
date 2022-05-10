import os
import sys
sys.path.append(os.environ.get('DISENT_ROOT'))
import argparse
import shutil
import numpy as np
import logging
import torch

from tqdm import tqdm
from dataloader_dali import statistics_train_loader


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
log = logging.getLogger(__name__)


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--dataset',
                     required=True,
                     choices=['miniImagenet', 'cub200'])
    opt.add_argument('--seed', 
                     default=100, type=int,
                     help='Seed for random augmentation.')
    opt.add_argument('--train_batch_size',
                     default=10)
    opt = vars(opt.parse_args())
    
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.deterministic = True
    
    if opt['dataset'] == 'cub200':
        opt['out_dir'] = os.path.join(os.environ.get("DATA_ROOT"), f"dataset/cub200_cropped_seed={opt['seed']}/")
        opt['train_dir'] = os.path.join(os.environ.get("DATA_ROOT"), f"dataset/cub200_cropped_seed={opt['seed']}/train_cropped_augmented")
    elif opt['dataset'] == 'miniImagenet':
        opt['out_dir'] = os.path.join(os.environ.get("DATA_ROOT"), f"dataset/miniImageNet-custom/")
        opt['train_dir'] = os.path.join(os.environ.get("DATA_ROOT"), f"dataset/miniImageNet-custom/train")
    else:
        raise NotImplementedError(f"Unsupported dataset: {opt['dataset']}")
        
    
    from dataloader_dali import push_train_loader
    
    state = {
        'train_dir': opt['train_dir'],
        'train_batch_size': opt['train_batch_size'],
        'seed': opt['seed'],
    }
    
    # no normalize
    train_loader = statistics_train_loader(state, img_size=224, seed=state['seed'])
    nb = train_loader.n_batches_per_epoch
    channels_sum, channels_squared_sum = 0, 0
    
    with tqdm(total=nb) as pb:
        for x, _ in train_loader:
            # mean over NHW
            channels_sum += torch.mean(x, dim=[0,2,3])
            channels_squared_sum += torch.mean(x ** 2, dim=[0,2,3])
            
            pb.update(1)
            
        mean = channels_sum / nb
        
        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / nb - mean ** 2) ** 0.5
    
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()
    
    mean_path = os.path.join(opt['out_dir'], 'mean.pth')
    std_path = os.path.join(opt['out_dir'], 'std.pth')
    log.info(f"mean: {mean}, std: {std}")
    
    torch.save(mean, mean_path)
    torch.save(std, std_path)
