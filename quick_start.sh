#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0 
# export PYTHONPATH=/mnt/sda/quick_start_demonstration/Awesome-Music-Generation:$PYTHONPATH
# export PYTHONPATH=/mnt/sda/quick_start_demonstration/Awesome-Music-Generation/data:$PYTHONPATH

CONFIG_YAML="MMGen_train/config/quick_start/quick_start.yaml"
LIST_INFERENCE="tests/captionlist/ChatGPT-4.0_prompt.lst"
RELOAD_FROM_CKPT="/root/mg2emotion/logs/latent_diffusion/config/train/checkpoints/checkpoint-fad-133.00-global_step=79999.ckpt"

python3 MMGen_train/infer.py \
    --config_yaml $CONFIG_YAML \
    --list_inference $LIST_INFERENCE \
    --reload_from_ckpt $RELOAD_FROM_CKPT