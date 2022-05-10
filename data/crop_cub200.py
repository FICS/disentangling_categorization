import os
import argparse
from matplotlib import pyplot as plt

from PIL import Image

import numpy as np


if __name__ == '__main__':
    # Precondition: CUB200 data is located in the folder named $DATA_ROOT/dataset/CUB_200_2011
    # Postcondition: The images are cropped and organized into train and test splits
    dataset_base_dir = os.path.join(os.environ.get("DATA_ROOT"), "dataset/CUB_200_2011/")
    dataset_images_dir = os.path.join(dataset_base_dir, "images/")
    images_text_path = os.path.join(dataset_base_dir, "images.txt")
    bounding_boxes_path = os.path.join(dataset_base_dir, "bounding_boxes.txt")
    splits_text_path = os.path.join(dataset_base_dir, "train_test_split.txt")

    cropped_dir = os.path.join(dataset_base_dir, 'cropped_split')
    if not os.path.isdir(cropped_dir):
        os.makedirs(cropped_dir)

    with open(images_text_path, 'r') as im_f, open(bounding_boxes_path, 'r') as bb_f, open(splits_text_path, 'r') as s_f:
        for im_line, bb_line, s_line in zip(im_f, bb_f, s_f):
            _, x, y, width, height = bb_line.rstrip('\n').split(' ')
            x, y, width, height = int(float(x)), int(float(y)), int(float(width)), int(float(height))
            _, is_training_image = s_line.rstrip('\n').split(' ')
            is_training_image = int(is_training_image)

            liner = im_line.rstrip('\n').split(' ')[1]
            class_dir, fname = liner.split('/')

            if is_training_image:
                cropped_dir_name = os.path.join(cropped_dir, 'train', class_dir)
            else:
                cropped_dir_name = os.path.join(cropped_dir, 'test', class_dir)

            if not os.path.isdir(cropped_dir_name):
                os.makedirs(cropped_dir_name)

            cropped_fname = os.path.join(cropped_dir_name, fname)

            if os.path.exists(cropped_fname):
                continue

            from_fname = os.path.join(dataset_images_dir, class_dir, fname)
            im = np.asarray(Image.open(from_fname))
            if len(im.shape) == 3:
                cropped_im = im[y:y+height, x:x+width, :]
            elif len(im.shape) == 2:
                cropped_im = im[y:y+height, x:x+width]
            else:
                raise ValueError('Bad input dimensions.')

            # plt.imshow(cropped_im)
            # plt.show()
            cropped_pil = Image.fromarray(cropped_im)
            cropped_pil.save(cropped_fname)
