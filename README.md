# Disentangling categorization

This repo contains code and experiment files for the paper "Disentangling Categorization in Multi-agent Emergent Communication" to appear in proceedings of NAACL 2022. 

## Docker installation

As part of the NAACL 2022 reproducibility track, we provide a dockerfile which can reproduce a result from the paper. Our file (`Dockerfile`) pulls the necessary data and performs the analysis for Table 1 in the paper, then prints the LaTeX table to standard output. 

First, make sure docker-cli is installed based on the Docker instructions for your platform. To build the image, from the repo folder do:

`docker build -t disent .`


## Bare-metal installation

Clone the repo to a folder, and set up environment variables in your shell:
```
$ export DISENT_ROOT=<path to cloned repo>
$ export DATA_ROOT=<path to folder to store input data>
$ export SAVE_ROOT=<path to folder to store output data>
```
`DATA_ROOT` and `SAVE_ROOT` can be any folders you choose, just make sure they exist before continuing. `DATA_ROOT` will house the processed input data, while `SAVE_ROOT` will store the byproducts of experiments and analysis. 

Install the conda environment and activate it (assumed to be active for rest of the instructions):

```
$ cd $DISENT_ROOT
$ bash -i setup.sh
$ conda activate disent
```

Note: Scripts in this project were so far only tested on Linux systems (e.g., Debian and Ubuntu installs). If you have ideas to add the functionality for Windows or MacOS, open an issue or MR. 


### JupyterLab

Additionally, if you are inside a JupyterLab instance, install the kernel for JupyterLab to use in notebooks:
```
$ ipython kernel install --user --name=disent
```
Figure out where it installed:
```
$ jupyter kernelspec list
```

In the folder where it installed, open `kernel.json` and define the same environment variables for the kernel by adding the following entry to the spec:
```
"env": {"DISENT_ROOT": "<path to cloned repo>",
        "DATA_ROOT": "<path to data root folder>",
        "SAVE_ROOT": "<path to output root folder>"}
```


## Data

Here are instructions to re-create the datasets in our paper. 

### CUB200 & CUB10

CUB200 recently moved to a new data provider, so report any dead links if you come across them. 

#### Automatic script

With `DISENT_ROOT` and `DATA_ROOT` set, simply activate the conda environment and run `python $DISENT_ROOT/data/cub200.py` to automatically download and prepare the dataset, or as a one-liner: `conda run --no-capture-output -n disent python $DISENT_ROOT/data/cub200.py`. It is recommended to run in a tmux session, since the data augmentation step can take between ten minutes to a couple hours depending on your storage speed. This option should work for most people. 


#### Manual

<details>
<summary>If you want to manually run through the steps, perform the following.</summary>

1. Download the dataset archive CUB_200_2011.tgz from http://www.vision.caltech.edu/datasets/cub_200_2011/: 
```
mkdir -p $DATA_ROOT/dataset && wget -O $DATA_ROOT/dataset/CUB_200_2011.tgz https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/
```
2. Unpack CUB_200_2011.tgz into `$DATA_ROOT/dataset`
```
tar -xzf $DATA_ROOT/dataset/CUB_200_2011.tgz -C $DATA_ROOT/dataset
```
3. Crop the images using `python $DISENT_ROOT/data/crop_cub200.py`
4. `mkdir -p $DATA_ROOT/dataset/cub200_cropped_seed=100`
5. Put the cropped training images in the directory `$DATA_ROOT/dataset/cub200_cropped/train_cropped/` (containing 200 folders): 
```
rsync -avh --progress $DATA_ROOT/dataset/CUB_200_2011/cropped_split/train/ $DATA_ROOT/dataset/cub200_cropped_seed=100/train_cropped
```
6. Put the cropped test images in the directory `$DATA_ROOT/dataset/cub200_cropped/test_cropped/` (containing 200 folders).
```
rsync -avh --progress $DATA_ROOT/dataset/CUB_200_2011/cropped_split/test/ $DATA_ROOT/dataset/cub200_cropped_seed=100/test_cropped
```
7. Augment the training set using `python $DISENT_ROOT/data/augment_cub200.py --seed <some integer>`
8. (optional) Run `data/dataset_statistics.py --dataset cub200` to get the updated mean/std values, then add them to config.py. This is only if you want to train a percept model from scratch using CUB200, instead of using ImageNet weights. 
9. Run `data/subsets_cub200.py` to get CUB10 datasets for each multi-agent seed. 

</details>


### miniImageNet

Before continuing, you must agree to terms of the data use here: https://mtl.yyliu.net/download/Lmzjm9tX.html

#### Automatic script

As before with `DISENT_ROOT` and `DATA_ROOT` set, simply run `conda run --no-capture-output -n disent python $DISENT_ROOT/data/miniImagenet.py` to automatically download and prepare the dataset. 


#### Manual

<details>
<summary>The manual steps are as follows.</summary>

1. Download the dataset miniImageNet into `$DATA_ROOT/dataset/miniImageNet` using `gdown`:
```
$ mkdir -p $DATA_ROOT/dataset/miniImageNet
$ pip install gdown 
$ gdown https://drive.google.com/uc?id=1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl -O $DATA_ROOT/dataset/miniImageNet/val.tar 
$ gdown https://drive.google.com/uc?id=107FTosYIeBn5QbynR46YG91nHcJ70whs -O $DATA_ROOT/dataset/miniImageNet/train.tar 
$ gdown https://drive.google.com/uc?id=1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v -O $DATA_ROOT/dataset/miniImageNet/test.tar
$ for f in $(ls $DATA_ROOT/dataset/miniImageNet ); do tar -xf $DATA_ROOT/dataset/miniImageNet/$f -C $DATA_ROOT/dataset/miniImageNet/; done
```
2. Run `data/resize_miniImagenet.py` to resize the dataset images to 224px. Note that the relative paths are coded into the script to avoid confusion. 
3. Run `data/dataset_statistics.py --dataset miniImagenet` to obtain mean/std and add them to config.py. 
4. Run `data/subsets_miniImagenet.py` to obtain CW sets for rotation matrix. 


You can do the same steps in a similar way for Fewshot-CIFAR100 (`fc100`), but in the paper we only report mini-ImageNet and CUB10 to save space. 

</details>

### Data pipeline

We provide custom data loaders that should be useful for big experiments involving Lewis signaling games. The signaling games can be very I/O heavy, because the game necessitates loading `B*D` images, where B is batch size and D is distractors count.  To speed this up, we created a custom data pipeline based on NVIDIA DALI (`dataloader_dali.py`). For this paper, the dataloaders cache features seperately for sender and receiver, and then use those for the entire signaling game to avoid hitting images on disk. However, our loaders also support "semiotic" mode, which is when the representations of the percept model are also optimized and updated during the game. In that case, the dataloaders re-cache features on-the-fly for subsequent "static" epochs. Check `dataloader_dali.py` for details. 


## Percept models (vision modules)

Similar to other papers in this area, each agent comprises of two parts: a perception module and an encoder/decoder module. Our code is adapted from the GumbelSoftmax agents in Facebook's EGG framework, which follows a similar agent composition. In our code, the object architecture of an agent translate to the following terms: Percept model (CNN, CW, ProtoPNet, etc.) -> Percept wrapper (standardized interface to agents) -> Features+Structures interface (e.g., `community/ConceptWhitening/features.py`) -> EGG Sender or Receiver wrapper (`agents2.py`, which are wrappers around the encoder/decoder components). 

### Checkpoints (download)

You can download the checkpoints for all the percept models used in the paper. The current filename is `disent_ckpt_041922-211456.zip`. Download it to `SAVE_ROOT` and unzip:

```
cd $SAVE_ROOT
wget https://www.dropbox.com/s/ep3p9cvcnulrign/percept_ckpt_041922-211456.zip?dl=0 -O percept_ckpt.zip
unzip percept_ckpt.zip
```

It will create a file structure rooted at `ckpt/autotrain/` where many checkpoints for each agent experiment are stored. 


### Checkpoints (re-create)

We provide the configuration files to run the train scripts for each percept type in `research_pool/config_archive/[PROTOPNET|CW|ConvNet]`. The automation code is still being polished and not released yet, but we include the main training scripts for now. The main train file for each type is as follows:

* ProtoPNet: `community/ProtoPNet/main_modular.py`
* CW: `community/ConceptWhitening/train.py`
* ConvNet: `community/ConvNet/train_baseline_percept.py`


## Agents

### Checkpoints (download)

You can download the checkpoints for the agents (which use percept model checkpoints above):

```
cd $SAVE_ROOT
wget https://www.dropbox.com/s/e6151tbhvmnie5m/agent_savedata_042722-150315.zip?dl=0 -O agent_savedata.zip
unzip agent_savedata.zip
```


## Figure generation

You can recreate a subset of the experiments (Q1) on a single GPU by running 

```
conda run -n disent --no-capture-output python $DISENT_ROOT/plotting_main.py -e q1 -g 1 --recreate
```

It will try to grab agent results database if it exists, otherwise it will schedule the analysis runs using agent checkpoints from above, then print results for Table 1 in the paper. 




