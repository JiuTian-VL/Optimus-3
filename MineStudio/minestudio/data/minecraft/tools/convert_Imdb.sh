#!/bin/bash

vpt_root=datas/MGOA
vpt_version=action_video
output=datas/minestudio/MGOA


python -m minestudio.data.minecraft.tools.convert_lmdb \
       --num-workers 16 \
       --input-dir ${vpt_root}/${vpt_version}/action \
       --action-dir  ${vpt_root}/${vpt_version}/action \
       --output-dir ${output} \
       --source-type 'action'

python -m minestudio.data.minecraft.tools.convert_lmdb \
       --num-workers 16 \
       --input-dir ${vpt_root}/${vpt_version}/video \
       --action-dir ${vpt_root}/${vpt_version}/action \
       --output-dir ${output} \
       --source-type 'video'