#!/bin/bash
# 核心消融矩阵（eval_llama3_fp8_ppl_ablation.py）
# B6：细粒度稀疏层（gate 仅前半，down/q 全层），与敏感度「gate 更敏感」叙事一致。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PY=eval_llama3_fp8_ppl_ablation.py
MODEL_ID=/data/sza/model/Meta-Llama-3.1-8B
DATASET=wikitext2
DATASET_PATH=/data/sza/local_dataset
OUTDIR=results/core
mkdir -p "${OUTDIR}"

COMMON_ARGS=(
  --model_id "${MODEL_ID}"
  --dataset "${DATASET}"
  --dataset_path "${DATASET_PATH}"
  --seq_lengths 2048
  --num_runs 5
  --tokenize_by sample
  --throughput_warmup_windows 2
  --cuda_warmup_passes 2
)

echo "===== A1 baseline / fp8 baseline ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes baseline,fp8_only \
  --output_json "${OUTDIR}/a1_baseline_fp8only.json"

echo "===== A2 down_proj: fp8_down_only / act24_only / fp8_act24 ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes fp8_down_only,act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj \
  --fp8_targets mlp.down_proj \
  --output_json "${OUTDIR}/a2_down_only.json"

echo "===== B1 q_proj only ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only \
  --sparse_targets self_attn.q_proj \
  --output_json "${OUTDIR}/b1_q_only.json"

echo "===== B2 gate_proj only ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only \
  --sparse_targets mlp.gate_proj \
  --output_json "${OUTDIR}/b2_gate_only.json"

echo "===== B3 down + q ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --output_json "${OUTDIR}/b3_down_q.json"

echo "===== B4 down + gate ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj \
  --output_json "${OUTDIR}/b4_down_gate.json"

echo "===== B5 down + gate + q（各后缀同一套默认层范围 all）====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --output_json "${OUTDIR}/b5_down_gate_q.json"

echo "===== B6 细粒度：down+q 全层 + gate 仅前半(层0..15) ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --sparse_layer_range all \
  --sparse_rules "mlp.gate_proj:0:16;mlp.down_proj:all;self_attn.q_proj:all" \
  --output_json "${OUTDIR}/b6_down_q_gate_half.json"

echo "===== Negative controls ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only \
  --preset neg_o_proj \
  --output_json "${OUTDIR}/neg_o_proj.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only \
  --preset neg_up_proj \
  --output_json "${OUTDIR}/neg_up_proj.json"

echo "run_core.sh done."
