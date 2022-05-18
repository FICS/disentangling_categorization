import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import copy
import numpy as np
import pandas as pd
import re
import argparse

from tqdm import tqdm

from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from fnmatch import filter
from copy import deepcopy

import util
from util import *
from util import project_root, DATA_ROOT, SAVE_ROOT
sys.path.append(project_root)
from util import process_exchange, EpochHistory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device={device}")

import agents2
from models import CnnWrapper, ProtoWrapper
from agents2 import RnnSenderGS, RnnReceiverGS, ProtoSenderGS
from games import SignalGameGS
import losses
from losses import loss_nll, loss_xent, least_effort
from test_tasks import semiotic_signal_test as check_model
from research_pool import config as config
import community.train_and_test as tnt
from research_pool.plotting_util import *


experiments = {}
experiments['q1'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-065556', # 5x resnet50
]

# experiments['q1c'] = [
#     f'{project_root}/research_pool/config_archive/COMMS-TEST/111121-065556',
# ]

experiments['q2'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111221-014304', # 5x resnet50
]

# experiments['q2c'] = [
# ]

experiments['q3'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-154815', # 5x proto resnet50
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-154941', # 5x cw resnet50 
]

# experiments['q3c'] = [
# ]

experiments['q4'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111221-013220', # 5x proto resnet50
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111221-013404', # 5x proto resnet50 (FL)
]

# experiments['q4c'] = [
# ]

experiments['q5'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/011522-001252', # 5x proto
    f'{project_root}/research_pool/config_autogen/COMMS-TEST/011522-001252', # 5x proto
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-154941', # 5x cw resnet50 
    f'{project_root}/research_pool/config_autogen/COMMS-TEST/111121-154941', # 5x cw resnet50
]

# experiments['q5c'] = [
# ]


args = argparse.ArgumentParser()
args.add_argument('-e', '--experiments', 
                  default=['q1'], nargs='+',
                  help='List of experiments to analyze. Use like cascade.py -e q1 q2 q3 [...]. Current options: q1 q2 q3 q4 q5')
args.add_argument('-g', '--node_gpus', 
                  default=[1], nargs='+', type=int,
                  help='List of #gpus each compute node has. If node0 has 2 gpu and node1 has 3 gpu, use like plotting_main.py -g 2 3')
# args.add_argument('-t', '--experiment_type',
#                   choices=['train_modular', 'analyze_comms'], required=False, default='train_modular',
#                   help='Which type of experiment to run, either train agents (train_modular) or analyze communications (analyze_comms).')
args.add_argument('--retry', 
                  action='store_true', required=False, default=False,
                  help='Retry the experiment plotting from scratch.')
args.add_argument('--recreate', 
                  action='store_true', required=False, default=False,
                  help='Recreate Table 1 on a single GPU (NAACL Reproducibility track).')

args = vars(args.parse_args())


selected_config = []
for experiment in args['experiments']:
    q = experiment.lower()
        
    selected_config.extend(experiments[q])

    
# ====================================
# ====================================
# ====================================


opt = {
    'from_dir': selected_config,
    'byproduct_dir': os.path.join(SAVE_ROOT, 'byproduct/disent/'),
}


opt['byproduct_base_dir'] = copy.deepcopy(opt['byproduct_dir'])

np.random.seed(9)
torch.manual_seed(9)
torch.backends.cudnn.deterministic = True

ts = []
for from_dir in opt['from_dir']:
    ts.append(from_dir.split('/')[-1])

timestamp = "+".join(ts)
print(timestamp)

byproduct_dir = os.path.join(opt['byproduct_dir'], timestamp)

if not os.path.exists(byproduct_dir):
    os.makedirs(byproduct_dir)

opt['byproduct_dir'] = byproduct_dir

log = print

expid_to_configs, expid_to_vars, experiment_obj, expid_to_eo = extract_experiment_structure(opt['from_dir'])


print(f"Found {sum([len(configs) for configs in expid_to_configs.values()])} configs.")


# Try to load, otherwise retry from scratch
try:
    seeds, db, db_paramsets = load_dfs_and_ps([os.path.join(opt['byproduct_base_dir'], eid) for eid in list(expid_to_eo.keys())])
except:
    print('Did not find existing analysis database.')
    args['retry'] = True
    

if args['retry']:
    print('Creating the analysis database from scratch.')
    
    from research_pool.plotting_util import *

    db = pd.DataFrame()
    # build configs database
    seeds = []

    for exp_id, configs in expid_to_configs.items():
        for config in configs:
            row_data = util.load_json(config)
            seeds.append(row_data['seed'])
            # fill in missing data
            # row_data['num_classes'] = 10
            save_dir = os.path.join(SAVE_ROOT, row_data['save_dir'], str(row_data['run_id']))
            canary_path = os.path.join(SAVE_ROOT, save_dir, "canary.txt")

            if not os.path.exists(canary_path):
                row_data['completed'] = False
            else:
                row_data['completed'] = True

            row_data['needs_update'] = True

            if row_data.get('prototypes_per_class', None) is None:
                row_data['prototypes_per_class'] = 0

            # make hashable
            row_data['semiotic_sgd_epochs'] = tuple(row_data['semiotic_sgd_epochs'])
            row_data['semiotic_push_epochs'] = tuple(row_data['semiotic_push_epochs'])

            row_name = config.split('/')[-1]
            row_name = row_name.replace('.json', '')
            row_name = f"{exp_id}_{row_name}_seed-{row_data['seed']}"

            if 'AGENTS' in row_data['save_dir']:
                row_data['experiment'] = 'agents'
            else:
                row_data['experiment'] = 'semiosis'

            if "Proto" in row_data['sender_percept_arch']:
                row_data['approach'] = "proto"
            elif "Cw" in row_data['sender_percept_arch']:
                row_data['approach'] = "cw"
            else:
                row_data['approach'] = "feats"

            row_data['experiments_id'] = exp_id
            row_data['experiments_variables'] = expid_to_vars[exp_id]
            row = pd.Series(data=row_data, name=row_name)
            db = db.append(row)


    seeds = list(set(seeds))
    db = format_db_types(db)

        
    if len(db[db['completed'] ==False]['save_dir']):
        print(db[db['completed'] ==False]['save_dir'])
        raise RuntimeError('Some experiments did not complete! See above for a list of those')


    # ====================================
    # ====================================
    # ====================================


    from util import ParamSet

    db_paramsets = {}


    linestyle_tuple = [
         ('solid', '-'),
         # ('loosely dotted',        (0, (1, 10))),
         ('dotted',                (0, (1, 1))),
         ('densely dotted',        (0, (1, 1))),

         ('loosely dashed',        (0, (5, 10))),
         ('dashed',                (0, (5, 5))),
         ('densely dashed',        (0, (5, 1))),

         ('loosely dashdotted',    (0, (3, 10, 1, 10))),
         ('dashdotted',            (0, (3, 5, 1, 5))),
         ('densely dashdotted',    (0, (3, 1, 1, 1))),

         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

    l = 0
    nl = len(linestyle_tuple)

    # add accuracies to database

    # model
    approach_key = 'approach'
    # ppc
    # approach_key = 'sender_prototypes_per_class'

    if approach_key == 'approach':
        # q1
        approaches = ['proto', "feats", 'cw']
        color_lookup = {
            ('CnnBWrapper', 'CnnBWrapper'): ('lightcoral', 'maroon'),
            ('CnnBWrapper', 'CwWrapper'): ('orange', 'darkorange'),
            ('CnnBWrapper', 'ProtoWrapper'): ('yellowgreen', 'darkolivegreen'),
            ('ProtoWrapper', 'ProtoWrapper'): ('lightcoral', 'maroon'),
            ('ProtoWrapper', 'CwWrapper'): ('yellowgreen', 'darkolivegreen'),
            ('ProtoWrapper', 'CnnBWrapper'): ('orange', 'darkorange'),
            ('CwWrapper', 'CwWrapper'): ('lightcoral', 'maroon'),
            ('CwWrapper', 'CnnBWrapper'): ('orange', 'darkorange'),
            ('CwWrapper', 'ProtoWrapper'): ('yellowgreen', 'darkolivegreen'),
        }
    else:
        # q4
        approaches = [1, 3, 5, 10]
        # approaches = [1, 10, 100]
        color_lookup = {
            10: ('orange', 'darkorange'),
            5: ('green', 'darkgreen'),
            3: ('deeppink', 'mediumvioletred'),
            1: ('darkturquoise', 'darkcyan'),
        }

    for i, approach in enumerate(approaches):
        if len(db[db[approach_key]==approach]) == 0:
            continue

        q = db[db[approach_key]==approach]

        if approach_key == 'sender_prototypes_per_class':
            a_colors = ["orchid", "skyblue", "deeppink",  "darkturquoise"]
        elif approach_key == 'approach':
            a_colors = ["orange", "yellowgreen", "darkturquoise", "skyblue", "slateblue", "orchid", "deeppink"]

        b_colors = ["darkorange", "darkolivegreen", "darkcyan", "steelblue", "darkslateblue", "darkviolet", "mediumvioletred"]
        c = 0

        for j, run_id in enumerate(q['run_id'].unique()):
            rdb = q[q['run_id'] == run_id]
            for ji, (name, row) in enumerate(rdb.iterrows()):
                ps = ParamSet(row)
                if approach_key == 'approach':
                    ps.set_color(color_lookup[(row['sender_percept_arch'], row['recv_percept_arch'])])
                elif approach_key == 'sender_prototypes_per_class':
                    ps.set_color(color_lookup[row['recv_prototypes_per_class']])
                else:
                    ps.set_color(zip(a_colors, b_colors)[c % len(a_colors)])
                    c += 1
                # ps.set_color(list(zip(a_colors, b_colors))[c % len(a_colors)])
                # c += 1

                ps.set_linestyle(linestyle_tuple[l % nl][1])
                db_paramsets[name] = ps

                epochs = ps.epoch_histories['epochs']

                sgd_idxes = row['semiotic_sgd_epochs']
                push_idxes = row['semiotic_push_epochs']

                valid_epochs = ps.epoch_histories['human_interp_epochs']
                accs = [ps.epoch_histories['receiver_accuracies'][i] for i in valid_epochs]
                idx = np.asarray(accs).argmax()
                idx = max(idx, 1)
                db.at[name, 'recv_acc'] = ps.epoch_histories['receiver_accuracies'][valid_epochs[idx]]
                db.at[name, 'best_epoch'] = valid_epochs[idx]
                db.at[name, 'num_sgd_epochs'] = len(row['semiotic_sgd_epochs'])
                db.at[name, 'num_push_epochs'] = len(row['semiotic_push_epochs'])

                if len(sgd_idxes) == 0:
                    continue

                concept_acc_map = ps.epoch_histories['concept_accuracies_map']


    db = db.astype({"best_epoch": int})

    # Checkpoint
    save_dfs_and_ps(db, db_paramsets, opt['byproduct_base_dir'])

    # ====================================
    # ====================================
    # ====================================

    from data.data_helpers import quiet_command, PipelineError
    from research_pool.config_utils import write_state

    # Inititate channel analysis 


    split = "TEST"
    configs_colle = []
    comms_config_dir = f"research_pool/config_archive/COMMS-{split}"
    comms_out_dir = opt['byproduct_base_dir']
    
    prerun_cmd = f"export DATA_ROOT={DATA_ROOT};SAVE_ROOT={SAVE_ROOT};DISENT_ROOT={project_root};"
    cascade = []
    
    for eid in list(expid_to_eo.keys()):
        configs_dir = os.path.join(project_root, comms_config_dir, eid)
        outs_dir = os.path.join(comms_out_dir, eid, f"COMMS-{split}")
        for d in [configs_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        exdb = db[db['experiments_id'] == eid]

        for name, row in exdb.iterrows():
            state = copy.deepcopy(dict(row[row.notnull()]))
            state['name'] = name
            state['split'] = split.lower()
            state['max_msgs'] = 1000
            state[f"{split.lower()}_batch_size"] = 20 # account for less samples
            # same as generate_configs
            state['analysis_dir'] = os.path.join(outs_dir, f"seed-{row['seed']}")
            state['analysis_dir'] = state['analysis_dir'].replace(SAVE_ROOT + '/', '')
            # print(f"Queueing {state['analysis_dir']}")
            json_path = os.path.join(configs_dir, f"{row['approach']}_{row['run_id']}_seed-{row['seed']}.json")
            write_state(state, json_path)
            ###
            cascade.append((json_path, f"{prerun_cmd}conda run --no-capture-output -n disent python $DISENT_ROOT/analyze_comms.py --config {json_path} --gpuid 0"))

        print(f"{configs_dir} is ready for cascade.")
        configs_colle.append(configs_dir.replace(project_root, ''))

    
    # Run the analysis scripts on single GPU 
    with tqdm(cascade, total=len(cascade)) as pb:
        for (json_path, cmd) in cascade:
            pb.set_postfix(status=f"Analyzing {os.path.split(json_path)[-1].replace('.json', '')}")
            ret = quiet_command(cmd)
            if ret != 0:
                raise PipelineError("Got nono-zero exit code from analyze_comms.py! Check traceback or try running manually to debug the issue."
                                    f"cmd={cmd}")
            pb.update(1)
            
    

    for j, (name, row) in enumerate(db.iterrows()):
        eid = row['experiments_id']
        run_id = row['run_id']
        pd_file = os.path.join(comms_out_dir, eid, f'COMMS-{split}', f"seed-{row['seed']}", str(run_id), f"{name}.pkl")

        if not os.path.exists(pd_file):
            print(f"Run did not finish: {pd_file}" )
            continue

        run_pd = pd.read_pickle(pd_file)
        assert run_pd['name'].iloc[0] == name

        for k in list(run_pd.keys()):
            if k != 'name':
                db.at[name, k] = run_pd[k].iloc[0]
                
    
    # Update data
    save_dfs_and_ps(db, db_paramsets, opt['byproduct_base_dir']) 


if args['recreate']:
    # Table 1
    approaches = ["feats", 'proto', 'cw']
    split = 'test'

    lookup = {
        "ProtoWrapper": "ProtoPNet", 
        "CwWrapper": "CW", 
        "CnnBWrapper": "ConvNet",
    }

    s_lines = [f"Sender Arch., Recv. Arch., Top. Sim., Dis. Sim."]
    ac_lines = [f"Sender, Recv., Recv. Acc."]
    
    
    def write_csv(lines, save_path):
        with open(save_path, 'w') as f:
            for l in lines:
                f.write(l + '\n')
            
    def fmt(std):
        return "${\\scriptstyle\\pm" + f"{std:.3f}" + "}$" 
    
    def mean_std(key, rows):
        # data = rdb[key]
        data = rdb[key][pd.notna(rdb[key])]
        _mean = np.mean(data, axis=0)
        _std = np.std(data, axis=0)
        return _mean, _std

    for i, approach in enumerate(approaches):
        if len(db[db['approach']==approach]) == 0:
            continue

        q1 = db[db['approach']==approach]
        q = q1

        for j, run_id in enumerate(q['run_id'].unique()):
            rdb = q[q['run_id'] == run_id]

            ts_mean, ts_std = mean_std(f'{split}_top_sim', rdb)

            S = rdb.iloc[0]['sender_percept_arch']
            R = rdb.iloc[0]['recv_percept_arch']

            # ts_lines.append(f"{lookup[S]},{lookup[R]},${ts_mean:.3f}$ {fmt(ts_std)}")

            ac_mean, ac_std = mean_std('recv_acc', rdb)

            ac_lines.append(f"{lookup[S]},{lookup[R]},${ac_mean:.3f}$ {fmt(ac_std)}")

            if 'CnnBWrapper' in rdb['sender_percept_arch'].unique():
                s_lines.append(f"{lookup[S]},{lookup[R]},${ts_mean:.3f}$ {fmt(ts_std)},-")
            else:
                ds_mean, ds_std = mean_std(f'{split}_dis_sim', rdb)
                s_lines.append(f"{lookup[S]},{lookup[R]},${ts_mean:.3f}$ {fmt(ts_std)},${ds_mean:.3f}$ {fmt(ds_std)}")
    
    
    if 'ProtoWrapper' in db['sender_percept_arch'].unique():
        approach = 'proto'
    else:
        approach = 'cw'

    save_path = os.path.join(SAVE_ROOT, opt['byproduct_base_dir'], list(expid_to_eo.keys())[0], f"{split}_ts.csv")
    write_csv(s_lines, save_path)
    print(save_path)
    print('\n'.join(s_lines))
    print()
    # cross reference for above
    save_path = os.path.join(SAVE_ROOT, opt['byproduct_base_dir'], list(expid_to_eo.keys())[0], f"recv_acc.csv")
    write_csv(ac_lines, save_path)
    print(save_path)
    print('\n'.join(ac_lines))
    
    # Done with Table 1