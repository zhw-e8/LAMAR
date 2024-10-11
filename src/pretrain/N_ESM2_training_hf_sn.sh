#!/bin/bash

work_dir=/home/zhouhanwen/

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
--nnodes 1 \
--nproc_per_node 8 \
N_ESM2_training_hf.py \
--tokenizer_path=${work_dir}/nucTran/tokenizer/single_nucleotide \
--model_max_length=2050 \
--model_name=${work_dir}/nucTran/config/config_150M.json \
--positional_embedding_type=rotary \
--hidden_size=768 \
--intermediate_size=3072 \
--num_attention_heads=12 \
--num_hidden_layers=12 \
--data_for_pretrain_path=${work_dir}/nucTran/mammalian/data/pretrain_data_train.2048.sedN.shuf.txt \
--batch_size=64 \
--peak_lr=1e-4 \
--warmup_ratio=0.02 \
--max_steps=500000 \
--grad_clipping_norm=1 \
--accum_steps=1 \
--output_dir=${work_dir}/nucTran/pretrain/saving_model/mammalian80D_2048len1mer1sw_80M_12head \
--save_steps=1000 \
--logging_steps=100 \
--fp16 \
--data_collator_patch \
--flash_attention \
--resume_training
