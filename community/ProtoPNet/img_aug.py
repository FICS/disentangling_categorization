import Augmentor
import os
from tqdm import tqdm

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

cwd = os.getcwd()
datasets_root_dir = os.path.join(cwd, 'datasets/cub200_cropped/')
# dir = datasets_root_dir + 'train_cropped/'
dir = os.path.join(datasets_root_dir, 'train_cropped')
# target_dir = datasets_root_dir + 'train_cropped_augmented/'
target_dir = os.path.join(datasets_root_dir, 'train_cropped_augmented')

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in tqdm(range(len(folders))):
    fd = folders[i]
    tfd = target_folders[i]
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
