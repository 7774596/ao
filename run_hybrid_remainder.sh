#!/bin/bash
# run_hybrid.sh 中 E1–E3（D1–D5 已跑完时可单独执行本脚本）
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PY=eval_llama3_fp8_ppl_ablation.py
MODEL_ID=/data/sza/model/Meta-Llama-3.1-8B
DATASET_PATH=/data/sza/local_dataset
# WikiText-103 必须用子目录（含 wikitext103.test.txt 等）；勿用 DATASET_PATH 根目录，
# 否则 load_from_disk 会先命中 WikiText-2 缓存，E2 实际仍在评 WT2。
DATASET_PATH_WT103="${DATASET_PATH}/wikitext103"
OUTDIR=results/hybrid
mkdir -p "${OUTDIR}"

COMMON_ARGS=(
  --model_id "${MODEL_ID}"
  --dataset_path "${DATASET_PATH}"
  --seq_lengths 2048
  --num_runs 5
  --tokenize_by sample
  --throughput_warmup_windows 2
  --cuda_warmup_passes 2
)

# E2 专用：仅含 WT103 路径，避免与 COMMON_ARGS 里父目录 --dataset_path 重复/歧义
COMMON_ARGS_E2=(
  --model_id "${MODEL_ID}"
  --dataset_path "${DATASET_PATH_WT103}"
  --seq_lengths 2048
  --num_runs 5
  --tokenize_by sample
  --throughput_warmup_windows 2
  --cuda_warmup_passes 2
)

echo "===== E2 泛化：最佳 act24 / best hybrid on WT103 ====="
python "${PY}" \
  "${COMMON_ARGS_E2[@]}" \
  --dataset wikitext103 \
  --modes baseline,fp8_only \
  --output_json "${OUTDIR}/e2_wt103_baselines.json"

python "${PY}" \
  "${COMMON_ARGS_E2[@]}" \
  --dataset wikitext103 \
  --modes act24_only \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --skip_last_n_layers 4 \
  --output_json "${OUTDIR}/e2_wt103_best_act24.json"

python "${PY}" \
  "${COMMON_ARGS_E2[@]}" \
  --dataset wikitext103 \
  --modes fp8_act24 \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --skip_last_n_layers 4 \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/e2_wt103_best_hybrid.json"


echo "run_hybrid_remainder.sh 全部任务结束。"
