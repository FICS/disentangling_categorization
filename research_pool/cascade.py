import sys
from config import project_root, DATA_ROOT, SAVE_ROOT
sys.path.append(project_root)

"""
Cascade experiments using tmuxp session file(s). 

Example session file:

session_name: 4-pane-split
windows:
- window_name: dev window
  layout: tiled
  shell_command_before:
    - cd ~/                    # run as a first command in all panes
  panes:
    - shell_command:           # pane no. 1
        - cd /var/log          # run multiple commands in this pane
        - ls -al | grep \.log
    - echo second pane         # pane no. 2
    - echo third pane          # pane no. 3
    - echo forth pane          # pane no. 4
"""
import argparse
import os
import torch
import binpacking
import numpy as np
from fnmatch import filter
from util import get_time_stamp, load_json, save_json

from config import global_shell_commands_before

# Maximum # of panes given the job weights we have is based on the device it is managing
class Pane(object):
    def __init__(self):
        self.jobs = []

    def add_job(self, job):
        self.jobs.append(job)

    def is_empty(self):
        return len(self.jobs) == 0


class Window(object):
    def __init__(self, max_pane):
        # Multiple panes run concurrently
        self.panes = []
        self.pane_counter = 0
        self.max_pane_count = max_pane
        
    def add_job(self, job):
        if len(self.panes) < self.max_pane_count:
            p = Pane()
            p.add_job(job)
            self.panes.append(p)
        else:
            self.panes[self.pane_counter % self.max_pane_count].add_job(job)
            self.pane_counter += 1

    def is_empty(self):
        return len(self.panes) == 0
    

experiments = {}
experiments['q1'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-065556', # 5x resnet50
]

experiments['q1c'] = [
    f'{project_root}/research_pool/config_archive/COMMS-TEST/111121-065556',
]

experiments['q2'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111221-014304', # 5x resnet50
]

experiments['q2c'] = [
]

experiments['q3'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-154815', # 5x proto resnet50
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-154941', # 5x cw resnet50 
]

experiments['q3c'] = [
]

experiments['q4'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111221-013220', # 5x proto resnet50
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111221-013404', # 5x proto resnet50 (FL)
]

experiments['q4c'] = [
]

experiments['q5'] = [
    f'{project_root}/research_pool/config_archive/SEMIOSIS/011522-001252', # 5x proto
    f'{project_root}/research_pool/config_autogen/COMMS-TEST/011522-001252', # 5x proto
    f'{project_root}/research_pool/config_archive/SEMIOSIS/111121-154941', # 5x cw resnet50 
    f'{project_root}/research_pool/config_autogen/COMMS-TEST/111121-154941', # 5x cw resnet50
]

experiments['q5c'] = [
]


args = argparse.ArgumentParser()
args.add_argument('-e', '--experiments', 
                  default=['q1'], nargs='+',
                  help='List of experiments to run. Use like cascade.py -e q1 q2 q3 [...]. Current options: q1 q2 q3 q4 q5')
args.add_argument('-g', '--node_gpus', 
                  default=[1], nargs='+', type=int,
                  help='List of #gpus each compute node has. If node0 has 2 gpu and node1 has 3 gpu, use like cascade.py -g 2 3')
args.add_argument('-t', '--experiment_type',
                  choices=['train_modular', 'analyze_comms'], required=False, default='train_modular',
                  help='Which type of experiment to run, either train agents (train_modular) or analyze communications (analyze_comms).')
args.add_argument('--retry', 
                  action='store_true', required=False, default=False,
                  help='Retry the experiment runs from scratch.')
args.add_argument('--restart', 
                  action='store_true', required=False, default=False,
                  help='Restart the experiment runs from their last saved epoch.')
args = vars(args.parse_args())


selected_config = []
for experiment in args['experiments']:
    q = experiment.lower()
    if args['experiment_type'] == 'analyze_comms':
        q += 'c'
        
    selected_config.extend(experiments[q])

retry_mode = None
if args.get('retry', False):
    retry_mode = 'retry'
if args.get('restart', False):
    # restart overrides retry
    retry_mode = 'restart'

    
opt = {
    # config root directory: /path/to/config_root
    'config_dir': selected_config,
    'node_to_gpuid': {
        node_id: list(range(node_gpus)) for node_id, node_gpus in enumerate(args['node_gpus'])
    },
    'gpu_per_proc': 1,  # Number of GPUs per experiment process
    # retry: restart without reloading 
    # restart: restart with reloading
    # extend: reload and continue
    'script_name': args['experiment_type'] + '.py',
    'retry_mode': retry_mode
}

# ================================================
# ================================================

nodes = list(opt['node_to_gpuid'].keys())

if opt['script_name'] == 'analyze_comms.py':
    exp_name = 'COMMS'
    opt['save_key'] = 'analysis_dir'
elif opt['script_name'] == 'train_modular.py':
    exp_name = 'SEMIOSIS'
    opt['save_key'] = 'save_dir'
else:
    raise ValueError(f"The script is not recognized: {opt['script_name']}")

save_key = opt['save_key']

    
# Every device gets a window
# Every proc gets a pane
# If we exceed number of panes allowed for each device, queue it in first available pane

configs = []
for cd in opt['config_dir']:
    configs.extend([os.path.join(cd, c) for c in filter(os.listdir(cd), "*.json")])

print(f"Found {len(configs)} {opt['script_name']} configs.")


if opt['retry_mode']:
    keep_configs = []
    for config_path in configs:
        state = load_json(config_path)
        save_dir = os.path.join(state[save_key], str(state['run_id']))
        canary_path = os.path.join(SAVE_ROOT, save_dir, "canary.txt")
            
        if not os.path.exists(canary_path):
            if opt['retry_mode'] == 'restart':
                print(f"Set configuration {config} to restart.")
                state['restart'] = 1
                  
            keep_configs.append(config_path)
            
    configs = keep_configs
    print(f"Keeping {len(configs)} configs for start.")
    

def id_to_max_pane(j):
    return 1

node_to_windows = {}
for node, gpuid in opt['node_to_gpuid'].items():
    node_to_windows[node] = {f'device{i}': Window(id_to_max_pane(i)) for i in gpuid}

jobs = {}
for config_path in configs:
    state = load_json(config_path)
    jobs[config_path] = 1
    
    
window_items = []
for node in nodes:
    for item in node_to_windows[node].items():
        window_items.append(item)
        
        
# Treat each device as a bin and perform bin packing
capacity_per_device = 1
pool = np.concatenate([opt['node_to_gpuid'][node] for node in nodes])

total_capacity = capacity_per_device * len(pool)
print(f"Total capcity: {total_capacity}")

    
# print(f"\nprocess queue {weight_so_far}\n", *[f"{j}\n" for j in jobs])
bins = binpacking.to_constant_bin_number(jobs, len(pool))

for dev_id, dev_bin, (_, window) in zip(pool, bins, window_items):
    for config_path, weight in dev_bin.items():
        window.add_job(f"python {opt['script_name']} --config {config_path} --gpuid {dev_id}")
        # print(f"python train_full.py --config {config_path} --gpuid {dev_id}")


# Prune
node_to_pruned_windows = {}
for node in nodes:
    node_to_pruned_windows[node] = {}
    for k in node_to_windows[node].keys():
        if not node_to_windows[node][k].is_empty():
            node_to_pruned_windows[node][k] = node_to_windows[node][k]
        
# for node in nodes:
    # print(node_to_pruned_windows[node].keys())
    
    
ts = []

for cd in opt['config_dir']:
    ts.append(cd.split('/')[-1])
    
ts = '+'.join(ts)

sessions_dir = f'{project_root}/research_pool/sessions/'
if not os.path.isdir(sessions_dir):
    os.makedirs(sessions_dir)

restart = f"_{opt['retry_mode']}" if opt['retry_mode'] else ""

job_counter = 0

for node in nodes:
    pruned_windows = node_to_pruned_windows[node]
    if len(pruned_windows.keys()) == 0:
        continue
    
    n_devices = len(opt['node_to_gpuid'][node])
    session_path = os.path.join(sessions_dir, f"{exp_name}_{ts}_node={node}_devices={n_devices}{restart}.yaml")
    with open(session_path, 'w') as yf:
        yf.write(f"session_name: {ts}x{node}\n")
        yf.write(f"windows:\n")
        for k in pruned_windows.keys():
            yf.write(f"- window_name: {k}\n")
            yf.write(f"  layout: tiled\n")
            yf.write(f"  shell_command_before:\n")
            for sc in global_shell_commands_before:
                yf.write(f"   - {sc}\n")

            yf.write(f"  panes:\n")
            for pane in pruned_windows[k].panes:
                yf.write(f"    - shell_command:\n")
                for job in pane.jobs:
                    yf.write(f"        - {job}\n")
                    job_counter += 1

    print(f"Session file is ready for node {node}:\n{session_path}")    
    
        
total_jobs = len(list(jobs.keys())) 
assert job_counter == total_jobs, f"Expected to have {total_jobs} jobs, but only wrote {job_counter}!"

