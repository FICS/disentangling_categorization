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

from util import project_root

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'community/'))

from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from collections import defaultdict

import torchvision.datasets as datasets

import util
import agents2
import models
from util import *
from community.ConceptWhitening.features import construct_CW_features
from community.ConvNet.features import construct_Cnn_features
from community.ProtoPNet.model import construct_PPNet

# map of which agent is compatible with which perceptual module. 
agent_compat_lists = {
    'RnnSenderGS': ["ProtoWrapper", "ProtoBWrapper", 
                    "CwWrapper", 
                    "CnnWrapper", "CnnBWrapper"],
    'FLRnnSenderGS': ["ProtoWrapper", "ProtoBWrapper", 
                      "CwWrapper", 
                      "CnnWrapper", "CnnBWrapper"],
    'OLRnnSenderGS': ["ProtoWrapper", "CwWrapper"],
    'MultiHeadRnnSenderGS': ["ProtoWrapper", "CwWrapper"],
    'MultiHeadRnnSenderGS2': ["ProtoWrapper", "CwWrapper"],
    'ProtoSenderGS': ["ProtoWrapper", "ProtoBWrapper"],
    'ProtoSender2GS': ["ProtoWrapper2"],
    'ProtoSender3GS': ["ProtoWrapper2"],
    'Top1ReifiedRnnSenderGS': ["ProtoWrapper"],
    'RnnReceiverGS': ["ProtoWrapper", "ProtoBWrapper", "MultiHeadProtoWrapper",
                      "CwWrapper", 
                      "CnnWrapper", "CnnBWrapper"],
    'FLRnnReceiverGS': ["ProtoWrapper", "ProtoBWrapper", "MultiHeadProtoWrapper",
                        "CwWrapper", 
                        "CnnWrapper", "CnnBWrapper"],
    'ProtoReceiver2GS': ["ProtoWrapper2"],
    'Top1ReifiedRnnReceiverGS': ["ProtoWrapper"],

}

percept_compat_lists = defaultdict(list)
percepts = []
for agent in list(agent_compat_lists.keys()):
    for percept in agent_compat_lists[agent]:
        percept_compat_lists[percept].append(agent)
        percepts.append(percept)
        
percepts = list(set(percepts))
for p in percepts:
    percept_compat_lists[p] = list(set(percept_compat_lists[p]))


# arch_key - class name from models.py
# ckpt_key - string index into state for architecture checkpoint
# base_cnn_key - string index into state for torchvision.models class name for base CNN model
def build_perceptual_wrapper(state, arch_key, ckpt_key, base_cnn_key, mean_key, std_key):
    # Normalization parameters
    mean = state[mean_key]
    std = state[std_key]
    
    # ===== init perceptual module. 
    # Some wrappers use the same base model so initialize the base model first. 
    if "Proto" in state[arch_key]:
        from ProtoPNet.settings import prototype_activation_function, add_on_layers_type, num_data_workers

        img_size = state.get('img_size', 224)
        if 'send' in arch_key:
            ppc = state['sender_prototypes_per_class']
        else:
            ppc = state['recv_prototypes_per_class']
        
        # shouldn't matter in either case since we load ProtoPNet state dict
        pretrained = 'ProtoB' not in state[arch_key]

        if state[ckpt_key] != '':
            model_state = torch.load(state[ckpt_key])
            try:
                proto_channels = model_state['prototype_shape'][1]
            except KeyError as e:
                # checkpoint from old version of code
                proto_channels = 128
        else:
            logging.warning("Auto-selected prototype channels to 128!")
            # Let constructor decide
            proto_channels = 128
        
        prototype_shape = (state['num_classes'] * ppc, proto_channels, 1, 1)
    
        enc = construct_PPNet(base_architecture=state[base_cnn_key],
                              pretrained=pretrained, 
                              img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=state['num_classes'],
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
        
        if state[ckpt_key] != '':
            logging.info(f"Load {arch_key} checkpoint from {state[ckpt_key]}")
            # model_state = torch.load(state[ckpt_key], map_location=state['device'])
            model_state = torch.load(state[ckpt_key])
            enc.load_state_dict(model_state['state_dict'])
    elif state[arch_key] == "CwWrapper":
        logging.info(f"Load {arch_key} checkpoint from {state[ckpt_key]}")
        model_state = torch.load(state[ckpt_key])
        model_state["num_classes"] = state["num_classes"]
        
        enc = construct_CW_features(model_state)
        
    elif "CnnB" in state[arch_key]:
        logging.info(f"Load {arch_key} checkpoint from {state[ckpt_key]}")
        model_state = torch.load(state[ckpt_key])
        model_state['num_classes'] = state['num_classes']
        model_state['cnn_pretrained'] = False
        
        enc = construct_Cnn_features(model_state)
    else:
        # no longer support TODO: remove
        if state['cnn_pretrained']: 
            logging.info(f"Load {arch_key} checkpoint from torchvision.models")
        else:
            logging.info(f"Pretrained off.")
            
        model_state = {
            'architecture': state[base_cnn_key],
            'num_classes': 1000, # use ImageNet weights in this configuration
            'cnn_pretrained': state['cnn_pretrained'],
            'state_dict': None,
        }
        enc = construct_Cnn_features(model_state)
    
            
    enc = enc.to(state['device'])
    enc_multi = torch.nn.DataParallel(enc)
    
    # ===== wrapper
    wrapper_arch = models.__dict__[state[arch_key]]
    
    if state[arch_key] == "ProtoWrapper2":
        wrapper = wrapper_arch(enc, enc_multi, topk=state['topk'], mean=mean, std=std).to(state['device'])
    else:
        wrapper = wrapper_arch(enc, enc_multi, mean=mean, std=std).to(state['device'])
        
    return wrapper
        

def build_complete_sender(state):
    assert state['sender_percept_arch'] in agent_compat_lists[state['sender_arch']], \
        f"The perceptual architecture {state['sender_percept_arch']} is not compatible with {state['sender_arch']}"
    
    percept_wrapper = build_perceptual_wrapper(state, 
                                               'sender_percept_arch', 
                                               'sender_percept_ckpt', 
                                               'sender_base_cnn',
                                               'sender_mean',
                                               'sender_std')

    sender_arch = agents2.__dict__[state['sender_arch']]
    if "MultiHeadRnnSenderGS" in state['sender_arch']:
        sender = sender_arch(input_size=state['sender_input_dim'], 
                             structure_size=state['sender_structure_dim'],
                             heads=state['max_len'],
                             vocab_size=state['vocab_size'], 
                             hidden_size=state['hidden_dim'], 
                             max_len=state['max_len'], 
                             embed_dim=state['embed_dim'], 
                             straight_through=state['gs_st'], 
                             cell=state['sender_cell'], 
                             trainable_temperature=state['learnable_temperature']).to(state['device'])
    elif "Top1ReifiedRnnSenderGS" in state['sender_arch']:
        sender = sender_arch(input_size=state['sender_input_dim'], 
                             sender_symbols=percept_wrapper.model.prototype_vectors, 
                             hidden_size=state['hidden_dim'], 
                             vocab_size=state['vocab_size'], 
                             max_len=state['max_len'], 
                             embed_dim=state['embed_dim'], 
                             cell=state['sender_cell'], 
                             straight_through=state['gs_st'],
                             trainable_temperature=state['learnable_temperature']).to(state['device'])
    else:
        sender = sender_arch(input_size=state['sender_input_dim'], 
                             vocab_size=state['vocab_size'], 
                             hidden_size=state['hidden_dim'], 
                             max_len=state['max_len'], 
                             embed_dim=state['embed_dim'], 
                             straight_through=state['gs_st'], 
                             cell=state['sender_cell'], 
                             trainable_temperature=state['learnable_temperature']).to(state['device'])
    
    if state['sender_ckpt'] != '':
        logging.info(f"Loading sender agent checkpoint.")
        sender.load_state_dict(torch.load(state['sender_ckpt']))
        
    return percept_wrapper, sender
    
    
def build_complete_receiver(state):
    assert state['recv_percept_arch'] in agent_compat_lists[state['recv_arch']], \
        f"The perceptual architecture {state['recv_percept_arch']} is not compatible with {state['recv_arch']}"
    
    percept_wrapper = build_perceptual_wrapper(state, 
                                               'recv_percept_arch', 
                                               'recv_percept_ckpt', 
                                               'recv_base_cnn',
                                               'recv_mean',
                                               'recv_std')

    recv_arch = agents2.__dict__[state['recv_arch']]
    recv_agent_arch = agents2.__dict__[state.get('recv_agent_arch', 'DistractedReceiverAgent')]
    recv_agent = recv_agent_arch(state['recv_input_dim'], state['hidden_dim'])
    if "Top1ReifiedRnnReceiverGS" in state['recv_arch']:
        receiver = recv_arch(percept=percept_wrapper,
                             signal_agent=recv_agent,
                             vocab_size=state['vocab_size'], 
                             embed_dim=state['embed_dim'], 
                             hidden_size=state['hidden_dim'], 
                             cell=state['receiver_cell']).to(state['device']) 
    else:
        receiver = recv_arch(recv_agent,
                              vocab_size=state['vocab_size'], 
                              embed_dim=state['embed_dim'], 
                              hidden_size=state['hidden_dim'], 
                              cell=state['receiver_cell']).to(state['device']) 
    
    if state['recv_ckpt'] != '':
        logging.info(f"Loading receiver agent checkpoint.")
        receiver.load_state_dict(torch.load(state['recv_ckpt']))
        
    return percept_wrapper, receiver


def full_system_from_row(row, log=print, test=True, from_epoch=None):
    state = row
    
    if from_epoch is None:
        from_epoch = int(state['best_epoch'])

    save_dir = os.path.join(state['save_dir'], str(state['run_id']))
    latest_ckpt_file = get_last_semiotic_model_file(save_dir, by_epoch=from_epoch)    
    
    if latest_ckpt_file:
        # take push model instead if it is available
        state['sender_percept_ckpt'] = os.path.join(save_dir, latest_ckpt_file)
        print(f"Loading {sender_encoder_path} based on best epoch {curr_epoch}")
        
    basename = os.path.join(state['save_dir'], str(state['run_id']), f"sender_e{from_epoch}")
    if os.path.exists(basename + '.pt'):
        state['sender_ckpt'] = basename + '.pt'
    else:
        state['sender_ckpt'] = basename + '.pth'
    
    basename = os.path.join(state['save_dir'], str(state['run_id']), f"receiver_e{from_epoch}")
    if os.path.exists(basename + '.pt'):
        state['recv_ckpt'] = basename + '.pt'
    else:
        state['recv_ckpt'] = basename + '.pth'
    
    sender_wrapper, sender = build_complete_sender(state)
    recv_wrapper, receiver = build_complete_receiver(state)
    
    for model in [sender_wrapper.model, sender_wrapper.model_multi, 
                      recv_wrapper.model, recv_wrapper.model_multi, 
                      sender, receiver]:
        model = model.to(state['device'])
    
    if test:
        for model in [sender_wrapper.model, sender_wrapper.model_multi, 
                      recv_wrapper.model, recv_wrapper.model_multi, 
                      sender, receiver]:
            model.eval()
    
    return sender_wrapper, sender, recv_wrapper, receiver
