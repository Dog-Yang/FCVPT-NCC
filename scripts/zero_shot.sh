#!/bin/bash
# custom config

DATA=./data
TRAINER=FCVPT
CFG=vit_b32   # vit_b32 vit_b16 vit_l14
dset="$1"
txt_cls=zero_shot
CUDA_VISIBLE_DEVICES=0 python main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--config-file configs/backbone/${CFG}.yaml \
--dataset-config-file configs/datasets/"${dset}".yaml \
--output-dir output/${TRAINER}/${CFG}/"${dset}" \
--lr 0.0005 \
--zero_shot \
--txt_cls ${txt_cls}
