#!/bin/bash
# 混合实验 + 泛化（与 eval_llama3_fp8_ppl_ablation.py 配套）
#
# 细粒度稀疏层：--sparse_rules "suffix:range;..."
#   range 与 --sparse_layer_range 相同（all|first_third|0:19 等），
#   并与 --skip_first_n_layers / --skip_last_n_layers 取交。
#   未在 sparse_rules 中出现的后缀仍用 --sparse_layer_range。
#
# 注意：Llama3.1-8B 为 32 层，「0:19」表示层下标 0..18（Python range 右开）。
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

echo "===== D1 主候选 H1: down + q 全层稀疏（末 4 层保护）+ FP8 ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset wikitext2 \
  --modes fp8_act24 \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --skip_last_n_layers 4 \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/d1_h1_wt2.json"

echo "===== D2 主候选 H2: gate 仅前 1/3 层；down / q 全层（与全局层规则一致时用 sparse_rules）====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset wikitext2 \
  --modes fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --sparse_layer_range all \
  --sparse_rules "mlp.gate_proj:first_third;mlp.down_proj:all;self_attn.q_proj:all" \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/d2_h2_first_third_wt2.json"

echo "===== D3 主候选 H2b: gate 仅层 0..18；down / q 全层 ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset wikitext2 \
  --modes fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --sparse_layer_range all \
  --sparse_rules "mlp.gate_proj:0:19;mlp.down_proj:all;self_attn.q_proj:all" \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/d3_h2_0_19_wt2.json"

echo "===== D3c H3: 同 D1 末层保护 + gate 仅前半层(0..15) + down/q 全层 ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset wikitext2 \
  --modes fp8_act24 \
  --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \
  --sparse_layer_range all \
  --sparse_rules "mlp.gate_proj:0:16;mlp.down_proj:all;self_attn.q_proj:all" \
  --skip_last_n_layers 4 \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/d3c_h3_gate_half_skiplast4_wt2.json"

echo "===== D4 异构量化：只对部分模块做 FP8 ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset wikitext2 \
  --modes fp8_act24 \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --skip_last_n_layers 4 \
  --fp8_skip_first_n_layers 5 \
  --fp8_quant_targets self_attn.q_proj,mlp.gate_proj,self_attn.o_proj,mlp.up_proj \
  --output_json "${OUTDIR}/d4_partial_fp8_targets_wt2.json"

echo "===== D5 负对照：o_proj / up_proj 稀疏 + 其余按 FP8 规则 ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset wikitext2 \
  --modes fp8_act24 \
  --preset neg_o_proj \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/d5_neg_o_proj_wt2.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset wikitext2 \
  --modes fp8_act24 \
  --preset neg_up_proj \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/d5_neg_up_proj_wt2.json"

echo "===== E1 泛化：最佳 act24 / best hybrid on PTB ====="
python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset ptb \
  --modes baseline,fp8_only \
  --output_json "${OUTDIR}/e1_ptb_baselines.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset ptb \
  --modes act24_only \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --skip_last_n_layers 4 \
  --output_json "${OUTDIR}/e1_ptb_best_act24.json"

python "${PY}" \
  "${COMMON_ARGS[@]}" \
  --dataset ptb \
  --modes fp8_act24 \
  --sparse_targets mlp.down_proj,self_attn.q_proj \
  --skip_last_n_layers 4 \
  --fp8_skip_first_n_layers 5 \
  --output_json "${OUTDIR}/e1_ptb_best_hybrid.json"

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

echo "===== E3 长度泛化：最佳 hybrid on WT2 ====="
for SEQ in 1024 2048 4096; do
  python "${PY}" \
    --model_id "${MODEL_ID}" \
    --dataset wikitext2 \
    --dataset_path "${DATASET_PATH}" \
    --modes fp8_act24 \
    --sparse_targets mlp.down_proj,self_attn.q_proj \
    --skip_last_n_layers 4 \
    --fp8_skip_first_n_layers 5 \
    --seq_lengths "${SEQ}" \
    --num_runs 5 \
    --tokenize_by sample \
    --throughput_warmup_windows 2 \
    --cuda_warmup_passes 2 \
    --output_json "${OUTDIR}/e3_wt2_best_hybrid_seq${SEQ}.json"
done

echo "run_hybird.sh 全部任务结束。"
