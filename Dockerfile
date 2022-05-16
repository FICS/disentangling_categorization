FROM danieldeutsch/python:3.7-cuda11.0.3-base

# Update NVIDIA gpg keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install Ubuntu packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    python3-dev \
    python3-pip \
    python3-venv \
    unzip \
    tmux \
    rsync \
    && rm -rf /var/lib/apt/lists/*


# RUN bash -c "yes || true" | unminimize

# Install Miniconda
ENV MINICONDA_VERSION 4.10.3
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_${MINICONDA_VERSION}-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# ============ result recreate =============
RUN mkdir -p /data/workspace/

ENV DATA_ROOT /data/workspace
ENV SAVE_ROOT /data/workspace
ENV DISENT_ROOT /data/workspace/src/disentangling_categorization

WORKDIR /data/workspace/src
RUN git clone https://github.com/FICS/disentangling_categorization
RUN bash -i $DISENT_ROOT/setup.sh

ENV CONDA_DEFAULT_ENV=disent
ENV CONDA_PREFIX=/opt/conda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Download agent data
WORKDIR /data/workspace
RUN wget https://www.dropbox.com/s/fozfvhfs73e7z82/agent-database-111121-065556.zip?dl=0 -q -O agent-database-111121-065556.zip && unzip agent-database-111121-065556.zip && rm -rf agent-database-111121-065556.zip

WORKDIR /data/workspace/src/disentangling_categorization
CMD ["sh", "/data/workspace/src/disentangling_categorization/reproduce.sh"]

# ============ /result recreate =============
