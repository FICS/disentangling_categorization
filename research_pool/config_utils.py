import sys
import os, argparse
import numpy as np
import json
import util
import shutil
import copy
import yaml

from typing import Iterable

# NumpyEncoder (c) Hunter Allen
# https://github.com/hmallen/numpyencoder/tree/f8199a61ccde25f829444a9df4b21bcb2d1de8f2
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    
        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)


def write_state(state, f_path, dry_run=False):
    with open(f_path, 'w') as config_file:
        json.dump(state, config_file, indent=2, cls=NumpyEncoder)
        
        
def write_yaml(state, f_path, dry_run=False):
    with open(f_path, 'w') as config_file:
        yaml.dump(state, config_file)
        
    
class Node(object):
    def __init__(self, parent, state: dict, name="", children=None):
        self.parent = parent
        self.children = children if children else []
        self.state = state
        self.name = name
        
    def is_root(self):
        return True if self.parent is None else False
    
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False
        
    def branch(self, key: str, params: list):
        for p in params:
            child_state = copy.deepcopy(self.state)
            child_state[key] = p
            self.children.append(Node(parent=self, state=child_state))
                
    def branch_zipped(self, keys: Iterable[str], param_matrix: Iterable[Iterable[object]]):
        """
        Handle the special scenario where we branch from a list of lists.
        keys: 2-tuple of keys
        param_matrix: list of 2-tuple, each a value for the key
        """
        assert len(keys) == 2
        k1, k2 = keys
        for v1, v2 in zip(*param_matrix):
            child_state = copy.deepcopy(self.state)
            child_state[k1] = v1
            child_state[k2] = v2
            self.children.append(Node(parent=self, state=child_state))
    
    def apply_state(self, new_dict):
        for key in list(new_dict.keys()):
            self.state[key] = new_dict[key]
            
    def __str__(self):
        s = ""
        for key, value in self.state.items():
            s += f"{key}: {value}\n"
        return s
    
    
def recursive_branch(subtree, params_list, params_lookup, debug=False):
    # Pre-condition: Root call passes a copied object of params_list if inside a loop
    if len(params_list) == 0 and subtree.is_leaf():
        return [subtree]
    if len(params_list) == 0:
        return []
    
    next_param = params_list.pop()
    if debug:
        print(next_param)
    subtree.branch(next_param, params_lookup[next_param])
    to_add = []
    
    for i, s2tree in enumerate(subtree.children):
        res = recursive_branch(s2tree, copy.deepcopy(params_list), params_lookup)
        to_add.extend(res)
        
    return to_add
