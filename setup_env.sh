#!/usr/bin/bash

#################### create miniconda env
ENV_NAME="rail-image"

set -e
source ~/miniconda3/etc/profile.d/conda.sh

##### rail #####
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    conda deactivate
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.12 -y

conda activate $ENV_NAME

#################### install packages
pip cache purge
conda clean --all -y

pip install pandas json5 scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install openai google-genai # for transcribing report form in json format
pip install transformers accelerate bitsandbytes torchvision # for labeling news & populating the report form
pip install sentencepiece # for running VLMs from Hugging Face
pip install py360convert # for extracting perspective views from equirectangular images
pip install pycolmap gsplat plyfile viser