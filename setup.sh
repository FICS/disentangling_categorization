#!/bin/bash
# Make sure to run this as `bash -i setup.sh`

conda env create -f $DISENT_ROOT/environment.yml
conda run -n disent pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110 --no-cache-dir
