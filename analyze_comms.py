import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import os
import re
import json
import shutil
import pandas as pd
import sys
import logging

import util
from util import get_time_stamp, load_json, save_json, project_root, DATA_ROOT, SAVE_ROOT, loader_to_message_data, pickle_write
sys.path.append(project_root)

from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F

from community.topsim import topographic_similarity
from community.info_theory import gap_mi_first_second, calc_entropy
from model_builder import build_complete_sender, build_complete_receiver, full_system_from_row
from optim import disable_parameter_requires_grad
from dataloader_dali import distractor_train_loader, distractor_test_loader

# =====================
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to configuration')
parser.add_argument('--gpuid', nargs='+', type=str, default="0")
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args['gpuid'][0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device={device}:{os.environ['CUDA_VISIBLE_DEVICES']}")

state = util.load_json(args['config'])
state = util.apply_env(state, log=print)


state['seed'] = int(state['seed'])
np.random.seed(state['seed'])
torch.manual_seed(state['seed'])
torch.backends.cudnn.deterministic = True

if "run_id" in list(state.keys()):
    run_id = state["run_id"] if state["run_id"] != "" else util.get_time_stamp()
    run_id = str(run_id)
else:
    run_id = util.get_time_stamp()

state['analysis_dir'] = os.path.join(state['analysis_dir'], run_id)
if not os.path.exists(state['analysis_dir']):
    os.makedirs(state['analysis_dir'])

    
    
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler(os.path.join(state['analysis_dir'], 'analyze_comms.log')),
                        logging.StreamHandler()
                             ],
                    level=logging.DEBUG)
log = logging.getLogger('analyze_comms')

    
config_basename = args['config'].split('/')[-1]
shutil.copyfile(args['config'], os.path.join(state['analysis_dir'], config_basename))

# always remove
canary_path = os.path.join(state['analysis_dir'], "canary.txt")
if os.path.exists(canary_path):
    os.remove(canary_path)


name = state['name']
# =====================
db = pd.DataFrame()
db = db.append(pd.Series(data={
    'name': name
}, name=name))

curr_epoch = state['best_epoch']
approach = state['approach']
split = state['split']

state['device'] = device
state['train_dir'] = state['train_push_dir']
state['seed'] = int(state['seed'])

log.info(f"Split: {state['split']}")

# returns eval() models
sender_wrapper, sender, receiver_wrapper, receiver = full_system_from_row(state, from_epoch=curr_epoch, test=True)

for model in [sender_wrapper, sender, receiver_wrapper, receiver]:
    disable_parameter_requires_grad(model)

try:
    if curr_epoch > state['apply_symbols_epoch']:
        receiver.apply_symbols = True
except KeyError:
    print('Not a reified architecture.')
    
    
# Use the joined loader to avoid caching for receiver images that aren't used anyway
if state['split'] == 'train':
    curr_loader = distractor_train_loader(state, 224, None, None)
else:
    curr_loader = distractor_test_loader(state, 224, None, None)
    
    
# ==========================
in_vectors = []
messages = []
actuals = []
preds = []

# play signal game with preprocessed feats
# n = len(curr_loader.cache)

md = loader_to_message_data(curr_loader, (sender_wrapper, receiver_wrapper), (sender, receiver), max_msgs=state['max_msgs'])
messages = md['messages']
in_vectors = md['in_vectors']
in_structs = md['in_structs']

# print("acc: ", state['recv_acc'])
ts = topographic_similarity(messages, in_vectors)
print('ts: ', ts)
db.at[name, f'{split}_top_sim'] = ts[0]
db.at[name, f'{split}_top_sim_pvalue'] = ts[1]

if state['sender_percept_arch'] != 'CnnBWrapper':
    ds = topographic_similarity(messages, in_structs)
    print('ds: ', ds)
    db.at[name, f'{split}_dis_sim'] = ds[0]
    db.at[name, f'{split}_dis_sim_pvalue'] = ds[1]

H = calc_entropy(messages)
db.at[name, 'entropy'] = H
print('entropy: ', H)

if approach == 'proto':
    m = state['prototypes_per_class']
else:
    m = 10


db.at[name, 'needs_update'] = False

db_path = os.path.join(state['analysis_dir'], f"{name}.pkl")
db.to_pickle(db_path)

cts_path = os.path.join(state['analysis_dir'], f"{name}-cts.pkl")
pickle_write(cts_path, md['class_to_symbols'])

vectors_path = os.path.join(state['analysis_dir'], f"{name}-vectors.pkl")
pickle_write(vectors_path, in_vectors)

structs_path = os.path.join(state['analysis_dir'], f"{name}-structures.pkl")
pickle_write(structs_path, in_structs)

messages_path = os.path.join(state['analysis_dir'], f"{name}-messages.pkl")
pickle_write(messages_path, messages)

md_path = os.path.join(state['analysis_dir'], f"{name}-full_message_data.pkl")
torch.save(md, md_path)

# md_path = os.path.join(state['analysis_dir'], f"{name}-full_message_data.pkl")
# torch.save(md, md_path)


with open(canary_path, 'w') as cf:
    cf.write("We made it.\n")
