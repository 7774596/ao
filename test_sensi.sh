#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python sensitivity_scan.py \
    --model_id /data/sza/model/Meta-Llama-3.1-8B \
    --dataset wikitext2 \
    --dataset_path /data/sza/local_dataset \
    --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
    --skip_first_n_layers 0 \
    --skip_last_n_layers 0 \
    --sparse_layer_range all \
    --max_calib_tokens 32768 \
    --max_calib_windows 48 \
    --output_json sensitivity_ranking.json