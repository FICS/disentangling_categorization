import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from community.ProtoPNet.preprocess import mean, std
from community.ProtoPNet.settings import img_size, base_architecture, \
                               prototype_activation_function, add_on_layers_type, num_data_workers, coefs
import numpy as np
from util import pickle_load, pickle_write, get_last_semiotic_model_file, build_class_to_prototype_files
from collections import defaultdict
from fnmatch import filter

GOLDEN_RATIO = 1.618

twocolumn_width = 6.25  # NeurIPS style file
columnsep = 0.25
onecolumn_width = twocolumn_width / 2  - (columnsep / 2)
onecolumn_height = onecolumn_width / GOLDEN_RATIO
onecolumn_thumb = onecolumn_height / 2

twocolumn_height = twocolumn_width / GOLDEN_RATIO
twocolumn_height_half = twocolumn_height / 2

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathptmx}',
                                       r'\usepackage[T1]{fontenc}',
                                       r'\usepackage[utf8]{inputenc}',
                                       r'\usepackage{pslatex}']

# axes.labelsize	Fontsize of the x and y labels
# axes.titlesize	Fontsize of the axes title
# figure.titlesize	Size of the figure title (Figure.suptitle())
# xtick.labelsize	Fontsize of the tick labels
# ytick.labelsize	Fontsize of the tick labels
# legend.fontsize	Fontsize for legends (plt.legend(), fig.legend())
# legend.title_fontsize

plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('legend', title_fontsize=6)
plt.rc('legend', fontsize=5)
plt.rc('figure', titlesize=10)



def plot_single(x, normalized=True):
    f, ax = plt.subplots(1, 1)
    # f.set_size_inches(8, 7)
    
    if type(x) is torch.Tensor:
        x = x.detach().cpu().squeeze()
    else:
        x = torch.as_tensor(x).squeeze()
        
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    x = x.numpy()        
    if normalized:
        x = (x * std) + mean
    
    ax.imshow(x)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def change_spline_color(ax_obj, color, thickness=2):
    for spine in ax_obj.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(thickness)


    
def extract_experiment_structure(from_dirs):
    expid_to_configs = defaultdict(list)
    expid_to_vars = defaultdict(list)
    experiment_obj = []
    expid_to_eo = {}

    # from config import SEMIOSIS

    for from_dir in from_dirs:
        exp_pkl_path = os.path.join(from_dir, filter(os.listdir(from_dir), "*.pkl")[0])
        eo = pickle_load(exp_pkl_path)
        experiment_obj.append(eo)
        expid_to_eo[eo.experiments_id] = eo

        print(f"Experimental variables of experiment object {eo.experiments_id}")    
        for attr in dir(eo):
            if '__' not in attr:
                if type(eo.__dict__[attr]) is list and len(eo.__dict__[attr]) > 1:
                    if attr == 'baseline_vocab_sizes':
                        attr = 'vocab_size'
                    elif attr == 'aux_losses_proto':
                        attr = 'aux_losses'
                    elif attr == 'aux_weights_proto':
                        attr = 'aux_weights'
                    elif attr == 'aux_losses_baseline':
                        attr = 'aux_losses'
                    elif attr == 'aux_weights_baseline':
                        attr = 'aux_weights'
                    elif attr == 'hidden_dims':
                        attr = 'hidden_dim'
                    elif attr == 'embed_dims':
                        attr = 'embed_dim'

                    print(f"\t{attr}")
                    expid_to_vars[eo.experiments_id].append(attr)

        expid_to_vars[eo.experiments_id] = list(set(expid_to_vars[eo.experiments_id]))
        found_configs = filter(os.listdir(from_dir), "*.json")
        found_configs = [os.path.join(from_dir, cf) for cf in found_configs]
        expid_to_configs[eo.experiments_id].extend(found_configs)
        
    return expid_to_configs, expid_to_vars, experiment_obj, expid_to_eo
        
        
        
def is_spe(row):
    return len(row['semiotic_push_epochs']) > 0


def is_lep(row):
    return len(row['aux_losses']) > 0

def is_sgd(row):
    return len(row['semiotic_sgd_epochs']) > 0

def num_sgd(row):
    return len(row['semiotic_sgd_epochs']) >= 12 and is_sgd(row)


def is_arch(row):
    return row['sender_arch'] == 'ProtoSenderGS'


def df_cond(df, cond, flip='', bools=False):
    b = True
    if flip == 'not':
        b = False
        
    names = []
    bix = []
    for name, row in df.iterrows():
        if (b == cond(row)):
            names.append(name)
            bix.append(True)
        else:
            bix.append(False)
        
    if bools:
        return bix
    else:
        return names


def load_dfs_and_ps(from_dirs):
    dfs = []
    seeds = []
    ps = None
    for from_dir in from_dirs:
        df_path = os.path.join(from_dir, 'db.pkl')
        print(f"Load from {df_path}")
        db = pd.read_pickle(df_path)
        dfs.append(db)
        seeds.extend(db['seed'].unique())
        
        ps_path = os.path.join(from_dir, 'db_paramsets.pkl')
        print(f"Load from {ps_path}")
        db_paramsets = pickle_load(ps_path)
        if ps is None:
            ps = db_paramsets
        else:
            ps.update(db_paramsets)
        
    df = pd.concat(dfs)
    seeds = set(seeds)
    
    return seeds, df, ps


def save_dfs_and_ps(db, paramset, base_dir):
    lk = len(list(paramset.keys()))
    accounted = 0
    for eid in db['experiments_id'].unique():
        if not os.path.isdir(os.path.join(base_dir, str(eid))):
            os.makedirs(os.path.join(base_dir, str(eid)))
            
        df_path = os.path.join(base_dir, str(eid), 'db.pkl')
        db[(db['experiments_id'] == eid)].to_pickle(df_path)
        print(f"Saved to {df_path}")
        
        save_dict = {}
        for k, v in paramset.items():
            if eid in k:
                save_dict[k] = v
                accounted += 1
                
        ps_path = os.path.join(base_dir, str(eid), 'db_paramsets.pkl')
        pickle_write(ps_path, save_dict)
        print(f"Saved to {ps_path}")
        
    assert accounted == lk, 'There were missing keys!'

    
def get_color(row):
    color = 'blue'
    # if row['sign_coef'] == 0:
    #    color = 'darkblue'
    
    if row['sender_percept_arch'] == 'CnnBWrapper':
        color = 'black'
    elif row['sender_percept_arch'] == 'ProtoWrapper':
        color = 'blue'
    elif row['sender_percept_arch'] == 'CwWrapper':
        color = 'red'
        
    if len(row['semiotic_push_epochs']) > 0:
        color = 'red'
        # if row['sign_coef'] == 0:
        #    color = 'darkred'
            
    return color


def get_sign_coef_color(row):
    oranges = {0.0:  'limegreen', 
               0.25: 'royalblue', 
               0.50: 'gold', 
               0.75: 'brown'}
    return oranges[row['sign_coef']]



import matplotlib.lines as mlines


class FigureHelper(object):
    def __init__(self):
        self.labels = []
        
    def get_legend_handles(self):
        m1 = mlines.Line2D([], [], color='limegreen', marker='.', linestyle='None',
                                   markersize=4, label="$\\mu_1=$" + str(0.00))
        m2 = mlines.Line2D([], [], color='royalblue', marker='.', linestyle='None',
                                   markersize=4, label="$\\mu_1=$" + str(0.25))
        m3 = mlines.Line2D([], [], color='gold', marker='.', linestyle='None',
                                   markersize=4, label="$\\mu_1=$" + str(0.50))
        m4 = mlines.Line2D([], [], color='brown', marker='.', linestyle='None',
                                   markersize=4, label="$\\mu_1=$" + str(0.75))
        m5 = mlines.Line2D([], [], color='purple', marker='.', linestyle='None',
                                   markersize=4, label="Non-semiotic")
        p1 = mlines.Line2D([], [], color='white', marker='s', linestyle='None', 
                                   markeredgecolor='black',
                                   markersize=4, label="No-push")
        p2 = mlines.Line2D([], [], color='white', marker='^', linestyle='None',
                                   markeredgecolor='black',
                                   markersize=4, label="Push")
        return [m1, m2, m3, m4, m5, p1, p2]
    
    def get_clm(self, row):
        if pd.notnull(row['sign_coef']):
            label = "$\\mu_1=$" + str(row['sign_coef'])
            color = get_sign_coef_color(row)
            
        else:
            label = "Non-Semiotic"
            color = 'purple'
            
        if label in self.labels:
            label = None
        else:
            self.labels.append(label)
            
        marker = "s"
        if is_spe(row):
            marker = '^'
            
        return color, label, marker
    
    

def plot_prototype_row(frow, x, y, axes, row, baseline_folder=None, k_select=0):
    state = dict(row)
    img_dir = os.path.join(state['save_dir'], str(state['run_id']), 'sign-img')
    spe = np.asarray(row['semiotic_push_epochs'])
    epoch = spe[np.where(spe <= row['best_epoch'])].max()
    
    if baseline_folder is None:
        epoch_folder = os.path.join(img_dir, f"epoch-{epoch}")
    else:
        epoch_folder = baseline_folder

    bb_file = os.path.join(epoch_folder, f"bb{epoch}.npy")
    bbrf_file = os.path.join(epoch_folder, f"bb-receptive_field{epoch}.npy")

    class_to_prototype_files = build_class_to_prototype_files(row['vocab_size'], 
                                                              row['prototypes_per_class'],
                                                              epoch_folder,
                                                             "prototype-img-original_with_self_act")
    for fcolumn in range(0, x):
        if y == 1:
            axes_obj = axes[fcolumn]
        else:
            axes_obj = axes[frow, fcolumn]

        axes_obj.set_xticks([])
        axes_obj.set_yticks([])
        # if fcolumn == 0:
        #     axes_obj.set_ylabel(str(epoch))

        try:
            prototype_file = class_to_prototype_files[k_select][fcolumn]
            prototype = plt.imread(prototype_file)
        except FileNotFoundError as e:
            print(f"prototype file not found! {prototype_file}")
            continue

        # if epoch == last_pepoch:
        #     change_spline_color(axes_obj, 'darkgreen', thickness=2)

        axes_obj.imshow(prototype)
        
        
def scores_to_tableA_csv(db, scores, scores_real, distractors=5, sign_coef=0.50):
    ss = r'{\scriptstyle'
    se = r'}'
    
    header = ['Variant', *scores]

    # sign_coef = 0.50
    lines = [','.join(header) + '\n']

    for lep_cond, lead_str in zip(['not', ''], ["Non-semiotic", "+LEP"]):
        subfields = [lead_str]

        # baseline non-semiotic
        select = db[(db['approach'] == 'proto') & 
                (db['learnable_temperature'] == 1.) & 
                (db['distractors'] == distractors) & 
                (df_cond(db, is_sgd, 'not', bools=True)) &
                (df_cond(db, is_lep, lep_cond, bools=True))]

        for score in scores_real:    
            mean = select[score].mean()
            std = select[score].std()

            if pd.isnull(std):
                std = 0.00
            subfields.append(f"${mean:.3f}{ss}\\pm{std:.3f}{se}$")

        lines.append(','.join(subfields) + '\n')


    # for sign_coef in [0.00, 0.5]:
    for spe_cond, push_str in zip(['', 'not'], ['(push)', '(no-push)']):
        for lep_cond, lead_str in zip(['not', ''], [f"Semiotic {push_str}", "+LEP"]):
            select = db[(db['approach'] == 'proto') & 
                        (db['learnable_temperature'] == 1.) & 
                        (db['distractors'] == distractors) & 
                        (df_cond(db, is_spe, spe_cond, bools=True)) &
                        (df_cond(db, is_lep, lep_cond, bools=True)) &
                        (db['sign_coef'] == sign_coef)]

            subfields = [lead_str]

            for score in scores_real:
                mean = select[score].mean()
                std = select[score].std()

                if pd.isnull(std):
                    std = 0.00
                subfields.append(f"${mean:.3f}{ss}\\pm{std:.3f}{se}$")

            lines.append(','.join(subfields) + '\n')
            
    return lines


def scores_to_figureA(db, scores, scores_real, csv=False):
    score_x, score_y = scores
    scr_x, scr_y = scores_real

    header = ['Variant', 'K', 'Sign Coef.', 'Op', *scores]
    lines = [','.join(header) + '\n']
    
    f, axes = plt.subplots(1, 3, sharey=True, sharex=True)
    f.set_size_inches(onecolumn_width, onecolumn_height * 0.8)
    f.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.85, wspace=0.2, hspace=0.1)

    for column, distractors in enumerate(sorted(db['distractors'].unique())):
        print('K=', distractors)
        axes_obj = axes[column]

        baseline = db['baseline_pm_acc'][db['baseline_pm_acc'].notnull()].iloc[0]
        axes_obj.axhline(baseline, xmin=0, xmax=100, color='green', linestyle='--', lw=0.5)

        fh = FigureHelper()

        # average with seed and LEP
        select = db[(db['approach'] == 'proto') & 
                    (db['learnable_temperature'] == 1.) & 
                    (db['distractors'] == distractors) & 
                    (db['sign_coef'].isnull())]

            # (df_cond(db, is_lep, lep_cond, bools=True))]

        color, label, marker = fh.get_clm(select.iloc[0])
        # print('sign_coef=', sign_coef, 'sgd_cond=', is_sgd(select.iloc[0]), 'lep_cond=', is_lep(select.iloc[0]))
        axes_obj.errorbar(select[scr_x].mean(), 
                          select[scr_y].mean(), 
                          xerr=select[scr_x].std(),
                          yerr=select[scr_y].std(),
                          linewidth=0.05,
                          marker=marker,
                          markersize=2,
                          color=color,
                          ecolor=color,
                          label=label,
                          linestyle='None') #, c=color, s=1)
        lines.append(f"Non-Sem.,{distractors},-, -, {select[scr_x].mean():.3f} \pm {select[scr_x].std():.3f}, {select[scr_y].mean():.3f} \pm {select[scr_x].std():.3f}\n")

        for sign_coef in sorted(db['sign_coef'].unique()[pd.notna(db['sign_coef'].unique())]):
            for spe_cond, spe_str in zip(['', 'not'], ['push', 'no-push']):
                # for lep_cond in ['', 'not']:
                select = db[(db['approach'] == 'proto') & 
                            (db['learnable_temperature'] == 1.) & 
                            (db['distractors'] == distractors) & 
                            (db['sign_coef'] == sign_coef) &
                            (df_cond(db, is_spe, spe_cond, bools=True))]
                    # (df_cond(db, is_lep, lep_cond, bools=True))]

                color, label, marker = fh.get_clm(select.iloc[0])
                # print('sign_coef=', sign_coef, 'sgd_cond=', is_sgd(select.iloc[0]), 'lep_cond=', is_lep(select.iloc[0]))
                axes_obj.errorbar(select[scr_x].mean(), 
                                  select[scr_y].mean(), 
                                  xerr=select[scr_x].std(),
                                  yerr=select[scr_y].std(),
                                  linewidth=0.05,
                                  marker=marker,
                                  markersize=4,
                                  color=color,
                                  ecolor=color,
                                  label=label,
                                  linestyle='None') #, c=color, s=1)
                
                lines.append(f"Semiotic,{distractors}, {sign_coef}, {spe_str}, {select[scr_x].mean():.3f} \pm {select[scr_x].std():.3f}, {select[scr_y].mean():.3f} \pm {select[scr_x].std():.3f}\n")


        axes_obj.set_title(f"$K=${distractors}")
        if column == 0:
            axes_obj.set_ylabel(score_y)

        if column == 1:
            axes_obj.set_xlabel(score_x)
        
        if scr_y == 'top_sim':
            axes_obj.set_ylim([0, db[scr_y].max() + 0.1])
        else:
            axes_obj.set_ylim([db[scr_y].min() - 5, 100])
            
        axes_obj.set_xlim([db[scr_x].min() - 5, 100])
    
    if csv:
        return f, lines
    else:
        return f

    
def format_db_types(db):
    data_types_dict = {
        'checkpoint_interval': int,
        'run_id': int,
        'distractors': int,
        'hidden_dim': int,
        'sender_input_dim': int,
        'recv_input_dim': int,
        'embed_dim': int,
        'epochs': int,
        'max_len': int,
        'gs_st': int,
        'test_batch_size': int,
        'train_batch_size': int,
        'vocab_size': int,
        'trainable_temperature': bool,
        'sender_prototypes_per_class': int,
        'recv_prototypes_per_class': int,
        'num_classes': int,
        'semiosis_start': int,
        'num_data_workers': int,
        'class_specific': bool,
        'sign_coef': float,
        'social_coef': float,
        'train_push_batch_size': int,
        'completed': bool,
        'needs_update': bool,
        'concept_classifier_epochs': int,
        'separated': bool,
        'seed': int,  # only allow integer
        'sender_structure_dim': int,
    }

    for key in list(db.keys()):
        if key in list(data_types_dict.keys()):
            db[key] = db[key].fillna(0).astype(int)
            db[key] = db[key].astype(data_types_dict[key])
            
    return db