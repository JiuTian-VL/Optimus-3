#!/bin/bash

export MINESTUDIO_SAVE_DIR="checkpoints/"
export MINESTUDIO_DATABASE_DIR="datas/minestudio"


CUDA_VISIBLE_DEVICES="1,2" python MineStudio/minestudio/tutorials/offline/1_finetune_vpts/main.py \
    devices=2 \
