#!/bin/bash
SAVE_ROOT=/data/workspace
DATA_ROOT=/data/workspace
DISENT_ROOT=/data/workspace/src/disentangling_categorization

cd $DISENT_ROOT

conda run -n disent --no-capture-output python $DISENT_ROOT/plotting_main.py -e q1 -g 1 --recreate

