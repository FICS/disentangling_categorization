import pickle
import string
import random
import pickle
import numpy as np
import torch
import fnmatch
import os
import json
import re
import pandas as pd
import yaml

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torchvision import datasets, transforms
from copy import deepcopy


project_root = os.environ.get('DISENT_ROOT', None)  # added to sys.path
DATA_ROOT = os.environ.get('DATA_ROOT', None)  # added to paths with apply_env(). 
SAVE_ROOT = os.environ.get('SAVE_ROOT', None)  # added to paths with apply_env(). 
if DATA_ROOT is None:
    raise IOError("Please set DATA_ROOT environment variable in your shell using installation directions.")
if SAVE_ROOT is None:
    raise IOError("Please set SAVE_ROOT environment variable in your shell using installation directions.")
if project_root is None:
    raise IOError("Please set DISENT_ROOT environment variable in your shell using installation directions.")
    

def get_time_stamp():
    date_object = datetime.now()
    return date_object.strftime('%m%d%y-%H%M%S')


def get_latest_checkpoint(folder, prefix):
    items = os.listdir(folder)
    matches = fnmatch.filter(items, f"{prefix}*")
    matches = sorted(matches)
    return os.path.join(folder, matches[-1])


def any_matching_prefix(folder, prefix):
    items = os.listdir(folder)
    matches = fnmatch.filter(items, f"{prefix}*")
    return len(matches) != 0


def apply_env(state, log=print):
    state = deepcopy(state)
    
    log(f"env:DISENT_ROOT={project_root}")
    log(f"env:DATA_ROOT={DATA_ROOT}")
    log(f"env:SAVE_ROOT={SAVE_ROOT}")
    
    # data 
    for key in ['train_dir', 'test_dir', 'val_dir', 
            'sender_percept_ckpt', 
            'recv_percept_ckpt', 'train_push_dir', 
            'concept_train_dir', 'concept_test_dir',
            'sender_ckpt', 'recv_ckpt', 'base_cnn_ckpt']:

        if state.get(key, None) is not None:
            if len(state[key]) == 0:
                continue

            if state[key][0] == '/':
                state[key] = state[key][1:]

            state[key] = os.path.join(DATA_ROOT, state[key])
 
    # byproducts/analysis file
    for key in ['save_dir', 'analysis_dir']:
        if state.get(key, None) is not None:
            if len(state[key]) == 0:
                continue

            if state[key][0] == '/':
                state[key] = state[key][1:]

            state[key] = os.path.join(SAVE_ROOT, state[key])
    
    return state
        
    
# def check_model(sender_wrapper, game,
#                 sender_interpret, receiver_interpret, 
#                 loader, device, return_metrics=False):
#     from test import semiotic_social_test
#     return semiotic_social_test


def process_exchange(message, recvr_output):
    trunc_messages = []
    receiver_outputs = []
    message = message.argmax(dim=-1)
    
    for i in range(message.size(0)):
        eos_positions = (message[i, :] == 0).nonzero()
        message_end = eos_positions[0].item() if eos_positions.size(0) > 0 else -1
        
        assert message_end == -1 or message[i, message_end] == 0
        if message_end < 0:
            trunc_messages.append(message[i, :])
        else:
            trunc_messages.append(message[i, :message_end + 1])
        
        # take last step of the receiver
        receiver_outputs.append(recvr_output[i, message_end, ...])
    
    return trunc_messages, torch.stack(receiver_outputs)
    

def safe_load(o_path):
    import numpy as np
    
    if not os.path.exists(o_path):
        print(f"Warn: {o_path} was not found!")
        return None
    else:
        o = np.load(o_path, allow_pickle=True)
        return o


def report(fname, s, log=print):
    log(s)
    with open(fname, 'a') as f:
        f.write(f"{s}\n")

        
def list_directories(directory: str):
    res = np.random.permutation(sorted(os.listdir(directory)))
    l = [di for di in res if os.path.isdir(os.path.join(directory, di))]
    return l


def load_json(config_filepath):
    with open(config_filepath) as config_file:
        state = json.load(config_file)
    return state


def save_json(state, f_path, dry_run=False):
    with open(f_path, 'w') as config_file:
        json.dump(state, config_file)
        
        
def load_yaml(filepath):
    with open(filepath, 'r') as f:
        state = yaml.safe_load(f)
    return state

        
def save_yaml(state, f_path):
    with open(f_path, 'w') as f:
        yaml.dump(state, f)

        
def pickle_write(fpath, obj):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)

    return obj


def get_semiotic_epoch_pairs(save_dir, by_epoch=np.inf) -> dict:
    # by_epoch: sender/receiver epoch that is the latest we search
    push_pattern = re.compile('([0-9]+)(push)')
    nopush_pattern = re.compile('([0-9]+)(nopush)')
    ext_pattern = re.compile('.pth')
    ext = ".pth"
    
    files = os.listdir(save_dir)

    found_push_epochs = []
    found_nopush_epochs = []
    max_push = 0
    max_nopush = 0

    for file in files:
        nop = nopush_pattern.match(file)
        p = push_pattern.match(file)
        if nop:
            epoch, project = nop.groups()
            epoch = int(epoch)
            if epoch <= by_epoch:
                found_nopush_epochs.append(epoch)
                if epoch > max_nopush:
                    max_nopush = epoch
            
            # old version of code didn't add .pth 
            if ext_pattern.match(file):
                ext = ".pth"
            
        elif p:
            epoch, project = p.groups()
            epoch = int(epoch)
            if epoch <= by_epoch:
                found_push_epochs.append(epoch)
                if epoch > max_push:
                    max_push = epoch
                    
            if ext_pattern.match(file):
                ext = ".pth"
    
    return {'push': [f"{i}push{ext}" for i in np.sort(found_push_epochs)],
            'nopush': [f"{i}nopush{ext}" for i in np.sort(found_nopush_epochs)],
            'max_push': max_push,
            'max_nopush': max_nopush}


def get_last_semiotic_model_file_pair(save_dir: str, by_epoch=np.inf) -> dict:
    # by_epoch: sender/receiver epoch that is the latest we search
    epoch_pairs = get_semiotic_epoch_pairs(save_dir, by_epoch)
    res = {'push': None, 'nopush': None}
    for project in ['push', 'nopush']:
        if len(epoch_pairs[project]):
            res[project] = epoch_pairs[project][-1]
            
    return res


def get_last_semiotic_model_file(save_dir: str, by_epoch: int) -> str:
    # return the latest semiotic model file
    epoch_pairs = get_semiotic_epoch_pairs(save_dir, by_epoch)
    if epoch_pairs['max_push'] == 0 and epoch_pairs['max_nopush'] == 0:
        return None
    
    # either nopush or push are avaialble, choose latest
    if epoch_pairs['max_push'] >= epoch_pairs['max_nopush']:
        return epoch_pairs['push'][-1]
    else:
        return epoch_pairs['nopush'][-1]
    

class EpochHistory(object):
    def __init__(self, epoch):
        self.epoch = epoch
        self.msg_lengths = defaultdict(list)
        self.hist_main_loss = []
        self.hist_aux_info = []
        self.sender_data = []
        self.reconstruction_data = []
        
        # Dict of form {'original': tensor, 'target': tensor, 'output': tensor}
        self.sender_sample_pairs = {}
        self.accuracy = 0.0
        self.concept_accuracy_map = {}
        self.concept_costs_map = {}
        
    def update_main_loss(self, main_loss):
        self.hist_main_loss.append(main_loss)
        
    def update_aux_info(self, aux_info_i):
        self.hist_aux_info.append(aux_info_i)
        
    def update_reconstructions(self, sender_data, reconstruction):
        self.sender_data.extend(sender_data)
        self.reconstruction_data.extend(reconstruction)
        
    def log_accuracy(self, accuracy):
        self.accuracy = accuracy
    
    def log_concept_accuracies(self, concept_accuracy: dict):
        for key, val in concept_accuracy.items():
            self.concept_accuracy_map[key] = val
            
    def log_concept_costs(self, concept_costs: dict):
        # This is for test time costs
        # Train time costs are kept track of using update_aux_info inside train.py functions
        for key, pack in concept_costs.items():
            xent, cluster_cost, accu, l1, p_avg_pair_dist, separation_cost, avg_separation_cost = pack
            self.concept_costs_map[key] = {
                'xent': xent,
                'cluster_cost': cluster_cost,
                'l1': l1,
                'p_avg_pair_dist': p_avg_pair_dist,
                'separation_cost': separation_cost,
                'avg_separation_cost': avg_separation_cost,
            }
        
    def set_sample_dict(self, d):
        self.sender_sample_pairs = d
        

class ParamSet(object):
    def __init__(self, series_obj):
        self.state = series_obj
        try:
            self.id = int(self.state['run_id'])
        except:
            print(series_obj)
            
        self.epoch_histories = self.init_histories()
        self.printable_label = self.init_label()
        self.color_a = None
        self.color_b = None
        self.linestyle = None
        
    @staticmethod
    def proc_concept_accuracy_map(acc_map: dict):
        res = {}
        for k, val in acc_map:
            res[k] = val * 100
        return res
    
    def init_histories(self):
        state = self.state
        aux_losses = state['aux_losses']
        aux_weights = state['aux_weights']
        
        res = {}
        history_path = os.path.join(SAVE_ROOT, self.state['save_dir'], str(self.id), 'history.pkl')
        eh = pickle_load(history_path)
        n = len(eh)
        
        res['epochs'] = np.asarray([eh[j].epoch for j in range(n)])
        
        res['receiver_accuracies'] = np.asarray([eh[j].accuracy * 100 for j in range(n)])
        
        concept_accuracies_map = defaultdict(list)
        for i in range(n):
            for key, val in eh[i].concept_accuracy_map.items():
                concept_accuracies_map[key].append([eh[i].epoch, val])
        # convert to 2d numpy
        np_concept_accuracies_map = {}
        for key, val in concept_accuracies_map.items():                
            np_concept_accuracies_map[key] = np.asarray(val)

        res['concept_accuracies_map'] = np_concept_accuracies_map
        
        concept_costs_df = pd.DataFrame()
        for i in range(n):
            for key, val in eh[i].concept_costs_map.items():
                concept_costs_df = concept_costs_df.append(pd.Series(data={
                    'push_type': key,
                    'epoch': i,
                    **val
                }, name=f"{i}-{key}"))
        
        res['concept_costs_df'] = concept_costs_df
        
        epochs = res['epochs']
        push_idxes = state['semiotic_push_epochs']
        sgd_idxes = state['semiotic_sgd_epochs']
        
        if len(push_idxes):
            # grab from epochs that were static after a push or on a push
            valid_epochs = list(deepcopy(push_idxes))
            valid_with_sentinal = list(valid_epochs) + [int(epochs[-1])]
            # print(valid_with_sentinal)
            for i in range(len(valid_with_sentinal) - 1):
                start = int(valid_with_sentinal[i])
                end = int(valid_with_sentinal[i+1])

                between = list(range(start, end, 1))
                for k in between:
                    if k not in sgd_idxes:
                        valid_epochs.append(k)
                        
        elif len(sgd_idxes):
            # select after first sgd epoch
            valid_epochs = list(range(sgd_idxes[0], int(epochs[-1])))
            
        else:
            valid_epochs = deepcopy(epochs)
        
        res['human_interp_epochs'] = valid_epochs
        
        res['main_loss'] = np.concatenate([eh[j].hist_main_loss for j in range(n)])
        
        res['expected_length'] = []
        res['main_loss'] = []
        res['least_effort'] = []
        
        for epoch_obj in eh:
            aux_dicts = epoch_obj.hist_aux_info
            for aux_dict in aux_dicts:
                res['expected_length'].append(float(aux_dict['expected_length']))
                
            res['main_loss'].extend(epoch_obj.hist_main_loss)
            
        # one update per dict (minibatch)
        res['expected_length_frequency'] = len(eh[1].hist_aux_info)
        res['main_loss_frequency'] = len(eh[1].hist_main_loss)
        
        return res
    
    def init_label(self):
        state = self.state

#         hidden_dim = state['hidden_dim']
#         embed_dim = state['embed_dim']
#         vocab_size = state['vocab_size']
#         sender_arch = state['sender_arch']
        aux_losses = state['aux_losses']
        aux_weights = state['aux_weights']
        sse = state['semiotic_sgd_epochs']
        spe = state['semiotic_push_epochs']
#         max_len = state['max_len']
        pretty_loss = {
            'least_effort': 'LEP',
            'Lp_reconstruction_loss': 'ABS'
        }
    
        def prettify(attr):
            pretty_attr = {
                "social_coef": "$\\beta$=",
                "sign_coef": "$\\alpha$=",
                "prototype_vectors_lr": "$\\eta_{P}$=",
                "add_on_layers_lr": "$\\eta_{\\theta^+}$=",
                "last_layer_lr": "$\\eta_{C}$=",
                "features_lr": "$\\eta_{\\theta}$=",
                "semiotic_sgd_epochs": f"SSGD-{len(sse)}",
                "semiotic_push_epochs": f"SP-{len(spe)}",
                "sender_arch": "$S$ Arch.=",
                "learnable_temperature": "$\\tau$-Opt.=",
                "vocab_size": "|A|=",
                "approach": "",
                "sender_percept_arch": "$S_f=$ ",
                "recv_percept_arch": "$R_f=$ ",
                "sender_prototypes_per_class": "$S_k=$ ",
                "recv_prototypes_per_class": "$R_k=$ ",
                "seed": "",
            }
            pretty = pretty_attr.get(attr, None)
            
            if pretty is not None:
                return pretty
            else:
                return attr.replace('_', ' ').capitalize()
    
        # s = f"H{hidden_dim} E{embed_dim} |V|={vocab_size} S={sender_arch} L={max_len} |S-SGD|={len(sse)} |S-Push|={len(spe)}"
        # s = f"|S-SGD|={len(sse)} |S-Push|={len(spe)}"
        s = []
        for attrb in state['experiments_variables']:
            if attrb == 'aux_weights' or attrb == 'aux_losses' or attrb == 'seed':
                continue  # handle below
            try:
                val = state[attrb]
            except KeyError:
                continue
            
            if attrb == 'approach':
                val = {'proto': 'Semiotic', 'feats': 'End2End'}[val]
                
            if type(val) is list or type(val) is tuple:
                if len(val) > 10:
                    val = ""
                else:
                    val = f"={val}"
            
            # fix architecture string
            if type(val) is str and "Wrapper" in val:
                lookup = {
                    "ProtoWrapper": "ProtoPNet", 
                    # "ProtoBWrapper":, 
                    "CwWrapper": "CW", 
                    # "CnnWrapper": "ConvNet", 
                    "CnnBWrapper": "ConvNet",
                }
                val = lookup[val] # val.replace("Wrapper", "")
                
                        # fix architecture string
            if type(val) is str and ("Sender" in val or "Receiver" in val):
                lookup = {
                    "RnnSenderGS": "Vanilla RNN",
                    "FLRnnSenderGS": "Vanilla RNN",
                    "OLRnnSenderGS": "1-Length",
                    "MultiHeadRnnSenderGS": "Self-attention RNN",
                    "MultiHeadRnnSenderGS2": "Self-attention RNN",
                    "ProtoSenderGS": "ProtoRNN",
                    "ProtoSender2GS": "ProtoRNN",
                    "ProtoSender3GS": "ProtoRNN",
                    "RnnReceiverGS": "Vanilla RNN",
                    "FLRnnReceiverGS": "Vanilla RNN",
                    "ProtoReceiver2GS": "ProtoRNN",
                }
                val = lookup[val] # val.replace("Wrapper", "")
                
            s.append(f"{prettify(attrb)}{val}")
            
        for loss, weight in zip(aux_losses, aux_weights):
            if type(weight) is float:
                s.append(f"{pretty_loss[loss]} {weight:.2f}")
            else:
                s.append(f"{weight[0]}-{pretty_loss[loss]} {weight[1]:.2f}")
        
        return ", ".join(s)
        
    def set_color(self, color_tuple):
        self.color_a, self.color_b = color_tuple
    
    def set_linestyle(self, ls):
        self.linestyle = ls


def save_enc_model(model, model_dir, model_name, log=print):
    '''
    model: this is not the multigpu model
    '''
    torch.save(obj=model.state_dict(), f=os.path.join(model_dir, f"{model_name}.pth"))
    log(f"Wrote prototype model to {model_dir}")


def construct_prototype_model(model_details: dict):
    from ProtoPNet import model
    prototype_shape = (model_details['num_classes'] * model_details['prototypes_per_class'], 
                       model_details['proto_channels'], 1, 1)
    
    ppnet = model.construct_PPNet(base_architecture=model_details['base_architecture'],
                                  pretrained=model_details['pretrained'], 
                                  img_size=model_details['img_size'],
                                  prototype_shape=prototype_shape,
                                  num_classes=model_details['num_classes'],
                                  prototype_activation_function=model_details['prototype_activation_function'],
                                  add_on_layers_type=model_details['add_on_layers_type'])
                                #   pretrain_ckpt=model_details.get('pretrain_ckpt', None))
    
    return ppnet


def merge_accuracies(concept_accuracies_map):
    # replace blank epochs in push matrix with nopush epochs (if they both exist)
    pass


def build_class_to_prototype_files(vocab_size, proto_per_class, epoch_folder, file_prefix):
    class_to_prototype_files = defaultdict(list)
    k = 0
    for ix in range(vocab_size):
        file = f'{file_prefix}{ix}.png'
        class_to_prototype_files[k].append(os.path.join(epoch_folder, file))

        if (ix + 1) % proto_per_class == 0:
            k += 1
            
    return class_to_prototype_files


def calculate_bbrf_areas(bbrf):
    '''
    proto_rf_boxes and proto_bound_boxes column (ProtoPNet/push.py):
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    vocab_size = bbrf.shape[0]
    heights = bbrf[:, 2] - bbrf[:, 1]
    widths = bbrf[:, 4] - bbrf[:, 3]
    areas = heights * widths
    
    return areas


def test_message_identity(messages, k, vocab_size, prototypes_per_class):
    ands = []
    for i in range(messages.size(0)):
        message = messages[i].argmax(axis=-1)
        identities = torch.zeros(vocab_size - 1, dtype=int)
        start = k[i] * prototypes_per_class
        end = start + prototypes_per_class
        identities[start:end] = 1
        proto_ids = message[message > 0].unique() # - 1  # remove eos
        proto_chosen = identities[proto_ids]
        
        if 0 in proto_chosen:
            ands.append(0)
        else:
            ands.append(1)
    
    return ands
        

def loader_to_message_data(loader, wrappers, agents, max_msgs=1000):
    sender, receiver = agents
    sender_percept, receiver_percept = wrappers
    in_images = []
    in_vectors = []
    in_structs = []
    messages = []
    actuals = []
    preds = []
    ands = []
    sender_matches = []
    recv_matches = []
    sender_recon = []
    recv_recon = []
    
    # play signal game with preprocessed feats
    # n = len(curr_loader.cache)
    class_to_symbols = defaultdict(list)

    with torch.no_grad():
        with tqdm(total=max_msgs) as pb:
            # loader.start_epoch('semiotic')
            start = 0
            for i, (sender_repr, recv_targets, recv_repr, sender_labels) in enumerate(loader):
                sender_images, _ = sender_repr
                recv_images, _ = recv_repr
                sender_images = sender_images.cuda()
                recv_images = recv_images.cuda()
                sender_repr, sender_structure = sender_percept.prelinguistic(sender_images)
                _, recv_repr, _ = receiver_percept(recv_images)
                
                message = sender((sender_repr, sender_structure))
                outputs, hiddens = receiver(message, recv_repr)
                
                end = min(start + sender_repr.size(0), loader.dataset_size)
                    
                for actual, message_am in zip(sender_labels, message.argmax(axis=-1)):
                    actual = actual.detach().cpu().item()
                    message_am = message_am.detach().cpu().numpy()
                    class_to_symbols[actual].extend(list(message_am))

                in_images.extend(sender_images[:end-start].detach().cpu())
                messages.extend(message[:end-start].detach().cpu().numpy())
                in_vectors.extend(sender_repr[:end-start].detach().cpu().numpy())
                in_structs.extend(sender_structure[:end-start].detach().cpu().numpy())
                actuals.extend(sender_labels[:end-start].detach().cpu().numpy())
                
                try:
                    sender_matches.extend(sender.matches[:end-start].detach().cpu().numpy())
                    sender_recon.extend(sender.recons[:end-start].detach().cpu().numpy())
                except AttributeError:
                    smatches = None
                    
                try:
                    recv_matches.extend(receiver.matches[:end-start].detach().cpu().numpy())
                    recv_recon.extend(receiver.recons[:end-start].detach().cpu().numpy())
                except AttributeError:
                    rmatches = None
                    
                
#                 if "Proto" in sender_wrapper.__class__.__name__:
#                     ppc = sender_wrapper.model.num_prototypes // sender_wrapper.model.num_classes
#                     ands_i = test_message_identity(message[:end-start].detach().cpu(), 
#                                                    sender_labels[:end-start].detach().cpu(), 
#                                                    _sender.vocab_size, ppc)
#                     ands.extend(ands_i)
                
                pb.update(
                    min(
                        max_msgs - len(message[:end-start]), 
                        len(message[:end-start])
                    )
                )
                start = end
                
                if len(messages) >= max_msgs:
                    break
    
    print(f"Reached max messages count of {max_msgs}")
    
    loader.reset()
    in_images = in_images[:max_msgs]
    messages = messages[:max_msgs]
    in_vectors = in_vectors[:max_msgs]
    in_structs = in_structs[:max_msgs]
    actuals = actuals[:max_msgs]
    
    if "Proto" in sender_percept.__class__.__name__:
        ands = ands[:max_msgs]
    
    messages = torch.as_tensor(messages).argmax(axis=-1)
    in_structs = torch.as_tensor(in_structs)
    in_vectors = torch.as_tensor(in_vectors)
    lengths = []
    uniques = []
    # print('messages', messages.shape)
    # print('in_vectors', in_vectors.shape)
    
    for i in range(messages.size(0)):
        eos_positions = (messages[i, :] == 0).nonzero()
        message_end = eos_positions[0].item() if eos_positions.size(0) > 0 else -1
        messages[i, message_end:] = 0
        lengths.append(eos_positions[0].item())
        nz = messages[i][messages[i] > 0]
        uniques.append(len(nz.unique()))
        
    if len(ands):
        ident = np.mean(ands)
    else:
        ident = None
        
    return {
        'in_images': in_images,
        'in_vectors': in_vectors,
        'in_structs': in_structs,
        'messages': messages,
        'actuals': actuals,
        'preds': preds,
        'class_to_symbols': class_to_symbols,
        'length_average': np.mean(lengths),
        'length_std': np.std(lengths),
        'uniques_average': np.mean(uniques),
        'uniques_std': np.std(uniques),
        'percent_identified': ident,
        'matches': [sender_matches, recv_matches],
        'recons': [sender_recon, recv_recon],
    }

