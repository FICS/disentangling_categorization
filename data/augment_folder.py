import os
import argparse
import shutil
import numpy as np
import logging
import torch
import concurrent.futures
import Augmentor


opt = argparse.ArgumentParser()
opt.add_argument('source_directory',
                 help='Source class image folder')
opt.add_argument('output_directory',
                 help='Output class image folder (augmented images)')
opt.add_argument('--seed', 
                 default=100, type=int,
                 help='Seed for random augmentation.')
opt = vars(opt.parse_args())

np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])
torch.backends.cudnn.deterministic = True

fd = opt['source_directory']
tfd = opt['output_directory']

# rotation
p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
p.flip_left_right(probability=0.5)
for i in range(10):
    p.process()
# skew
s = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
s.skew(probability=1, magnitude=0.2)  # max 45 degrees
s.flip_left_right(probability=0.5)
for i in range(10):
    s.process()
# shear
q = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
q.shear(probability=1, max_shear_left=10, max_shear_right=10)
q.flip_left_right(probability=0.5)
for i in range(10):
    q.process()
# random_distortion
#p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
#p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
#p.flip_left_right(probability=0.5)
#for i in range(10):
#    p.process()
#del p