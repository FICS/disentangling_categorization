import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import os
import re
import json
import shutil
import logging
import sys

import util
from util import *
from dataloader_dali import push_train_loader, normalized_train_loader, normalized_test_loader, \
                            separated_cached_distractor_train_loader, separated_cached_distractor_test_loader
from games import SignalGameGS, ReificationAEGameGS
import losses
from losses import loss_nll, loss_xent, least_effort
import train
from community import train_and_test as tnt
from model_builder import build_complete_sender, build_complete_receiver
from optim import set_eval, set_train, semiotic_social_optimizers
from tasks import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to configuration')
parser.add_argument('--gpuid', nargs='+', type=str, default="0")
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args['gpuid'][0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device={device}:{os.environ['CUDA_VISIBLE_DEVICES']}")

state = util.load_json(args['config'])
state['device'] = device

state = util.apply_env(state, log=print)


if "run_id" in list(state.keys()):
    run_id = state["run_id"] if state["run_id"] != "" else util.get_time_stamp()
    run_id = str(run_id)
else:
    run_id = util.get_time_stamp()


state['save_dir'] = os.path.join(state['save_dir'], run_id)
if not os.path.exists(state['save_dir']):
    os.makedirs(state['save_dir'])
            

# ============ start logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[
                        logging.FileHandler(os.path.join(state['save_dir'], 'train_full.log')),
                        logging.StreamHandler()
                             ],
                    level=logging.DEBUG)
log = logging.getLogger('train_full')


if "Proto" in state['sender_percept_arch']:
    approach = "proto"
elif "Cw" in state['sender_percept_arch']:
    approach = "cw"
elif "CnnB" in state['sender_percept_arch']:
    approach = "cnnb"
else:
    approach = "cnn"

state['approach'] = approach

try:
    np.random.seed(state['seed'])
    torch.manual_seed(state['seed'])
    torch.backends.cudnn.deterministic = True
    
    config_basename = args['config'].split('/')[-1]
    shutil.copyfile(args['config'], os.path.join(state['save_dir'], config_basename))

    history_path = os.path.join(state['save_dir'], 'history.pkl')
    state['history_path'] = history_path
    epoch_histories = []
    start_epoch = 0
    
    log.info(f"Sender percept mean: {state['sender_mean']}, std: {state['sender_std']}")
    log.info(f"Receiver percept mean: {state['recv_mean']}, std: {state['recv_std']}")
    
    ## ========================= manage restarts, invariant of perceptual wrapper
    cond1 = state['restart']
    cond2 = util.any_matching_prefix(state['save_dir'], 'sender_e') and util.any_matching_prefix(state['save_dir'], 'receiver_e')
    if cond1 and cond2:
        state['sender_ckpt'] = util.get_latest_checkpoint(state['save_dir'], 'sender_e')
        state['recv_ckpt'] = util.get_latest_checkpoint(state['save_dir'], 'receiver_e')
        log.info(f"\Found sender at {state['sender_ckpt']}")
        log.info(f"\Found receiver at {state['recv_ckpt']}")

        start_epoch = int(re.search(r'\d+', state['sender_ckpt'].split('/')[-1]).group(0))
        log.info(f"\tAdvance to epoch {start_epoch}")

        # update signs model to the latest available epoch (only sender is trained)
        latest_ckpt_file = get_last_semiotic_model_file(state['save_dir'], by_epoch=start_epoch)
        if latest_ckpt_file:
            state['sender_percept_ckpt'] = os.path.join(state['save_dir'], latest_ckpt_file)

        epoch_histories = pickle_load(history_path)
    
    num_distractors = state['distractors']

    # Initialize models
    # ==================================
    sender_percept, _sender = build_complete_sender(state)
    recv_percept, _receiver = build_complete_receiver(state)
    
    # disable grad and only enable later
    for model in [sender_percept, _sender, recv_percept, _receiver]:
        disable_parameter_requires_grad(model)
        
    set_eval([sender_percept, _sender, recv_percept, _receiver])
    
    log.info('\t== system summary ==')
    log.info('sender: ' + ' -> '.join([
        str(state['sender_base_cnn']),
        str(sender_percept.__class__.__name__),
        str(_sender.__class__.__name__)]))
    log.info('receiver: ' + ' -> '.join([
        str(state['recv_base_cnn']),
        str(recv_percept.__class__.__name__),
        str(_receiver.__class__.__name__),
    ]))
    
    # dataloaders
    # ==================================
    # receiver and sender models are separate
    # normalization happens inside percept wrappers
    log.debug("Initiate training loader")
    img_size = state.get("img_size", 224)
    semiotic_train_loader = separated_cached_distractor_train_loader(state, 
                                                                     sender_percept.prelinguistic, 
                                                                     recv_percept.prelinguistic, 
                                                                     img_size, mean=None, std=None)
    log.debug("Initiate test loader")
    semiotic_test_loader = separated_cached_distractor_test_loader(state, 
                                                                   sender_percept.prelinguistic, 
                                                                   recv_percept.prelinguistic, 
                                                                   img_size, mean=None, std=None)

    # loader for true conv projection operation
    train_push_loader = push_train_loader(state, img_size, seed=state['seed'])

    # loader for last layer convex optimization (of sender)
    ll_train_loader = normalized_train_loader(state, img_size, state['seed'], 
                                              mean=state['sender_mean'], 
                                              std=state['sender_std'])
    ll_test_loader = normalized_test_loader(state, img_size, state['seed'], 
                                            mean=state['sender_mean'], 
                                            std=state['sender_std'])

    log.info(f'training set size: {semiotic_train_loader.dataset_size}')
    log.info(f'test set size: {semiotic_test_loader.dataset_size}')
    log.info(f"batch size: {state['train_batch_size']}/{state['test_batch_size']}")
    log.info(f"push batch size: {state['train_push_batch_size']}")


    assert len(state['aux_losses']) == len(state['aux_weights']), "Each aux loss should have a weight!"
    aux_losses, system_losses = losses.unpack_losses(state['aux_losses'], state['aux_weights'])

    # Always remove
    canary_path = os.path.join(state['save_dir'], "canary.txt")
    if os.path.exists(canary_path):
        os.remove(canary_path)

    experiment = state.get('experiment', 'semiotic_social_1game')

    if experiment == 'semiotic_social_1game':
        # game define
        # ==================================
        signal_game = SignalGameGS(_sender, _receiver, loss_xent, 
                            length_cost=state['length_cost'], 
                            aux_losses=aux_losses,
                            sys_losses=system_losses)

        # optimizers
        static_optimizer, semiotic_optimizer, classifier_optimizer = \
            semiotic_social_optimizers(state, sender_percept, signal_game.sender, 
                                       recv_percept, signal_game.receiver)    

        # tasks: semiotic signaling, classification
        semiotic_signal_task(state, start_epoch, epoch_histories,
                             semiotic_train_loader, semiotic_test_loader, 
                             train_push_loader, ll_train_loader, ll_test_loader,
                             sender_percept, recv_percept, signal_game,
                             static_optimizer, semiotic_optimizer, classifier_optimizer,
                             device)
    else:
        raise NotImplementedError(f"Unknown experiment: {experiment}")

    with open(canary_path, 'w') as cf:
        cf.write("We made it.\n")
        
except Exception as e:
    log.exception(e)
