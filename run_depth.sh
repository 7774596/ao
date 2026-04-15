#!/bin/bash
# 层深 / 保护带扫描（eval_llama3_fp8_ppl_ablation.py）
# C3 使用 --sparse_rules：仅压缩 gate 的层窗，down / q 仍为全层（再与 skip 取交）。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PY=eval_llama3_fp8_ppl_ablation.py
MODEL_ID=/data/sza/model/Meta-Llama-3.1-8B
DATASET=wikitext2
DATASET_PATH=/data/sza/local_dataset
OUTDIR=results/depth
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

echo "===== C1 down + q : skip_last sweep ====="
for SKIP_LAST in 0 4 8; do
  python "${PY}" \
    "${COMMON_ARGS[@]}" \
    --modes act24_only,fp8_act24 \
    --sparse_targets mlp.down_proj,self_attn.q_proj \
    --skip_last_n_layers "${SKIP_LAST}" \
    --output_json "${OUTDIR}/c1_down_q_skiplast_${SKIP_LAST}.json"
done

echo "===== C2 gate only : layer-range sweep ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only \
  --sparse_targets mlp.gate_proj \
  --sparse_layer_range first_third \
  --output_json "${OUTDIR}/c2_gate_first_third.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only \
  --sparse_targets mlp.gate_proj \
  --sparse_layer_range 0:19 \
  --output_json "${OUTDIR}/c2_gate_0_19.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only \
  --sparse_targets mlp.gate_proj \
  --skip_last_n_layers 8 \
  --output_json "${OUTDIR}/c2_gate_skiplast8.json"

echo "===== C3 down + gate + q：细粒度（gate 限层，down/q 全层）====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --sparse_layer_range all \
  --sparse_rules "mlp.gate_proj:first_third;mlp.down_proj:all;self_attn.q_proj:all" \
  --output_json "${OUTDIR}/c3_rules_first_third.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --sparse_layer_range all \
  --sparse_rules "mlp.gate_proj:0:19;mlp.down_proj:all;self_attn.q_proj:all" \
  --output_json "${OUTDIR}/c3_rules_0_19.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --sparse_layer_range all \
  --sparse_rules "mlp.gate_proj:all;mlp.down_proj:all;self_attn.q_proj:all" \
  --skip_last_n_layers 8 \
  --output_json "${OUTDIR}/c3_rules_skiplast8.json"

echo "===== C4 early dense / late protected ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --modes act24_only,fp8_act24 \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --skip_first_n_layers 4 \
  --skip_last_n_layers 4 \
  --output_json "${OUTDIR}/c4_down_q_skipfirst4_skiplast4.json"

echo "run_depth.sh done."
