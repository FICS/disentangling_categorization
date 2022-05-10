import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import os
import re
import shutil
import logging
import sys
sys.path.append("/opt/app/prototype-signal-game")

import torchvision.models as models

from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F

import util


def test(state, model, test_loader, log=print):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    n_batches = test_loader.n_batches_per_epoch
    with tqdm(total=n_batches) as pb:
        for (images, labels) in test_loader:
            images, labels = images.to(state['device']), labels.to(state['device'])

            # images = Variable(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print('pred', predicted)
            # print('targ', labels)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            str_acc = f"{100.0 * (correct / total):.4f}"
            pb.update(1)
            pb.set_postfix(accuracy=str_acc)

        test_loader.reset()
    
    accuracy = (100.0 * correct / total)
    if type(accuracy) is torch.Tensor:
        accuracy = accuracy.cpu().numpy()
        
    log(f'Test Accuracy of the model on test images: {accuracy:.3f} %%')
    return accuracy