#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

bash run_core.sh
bash run_depth.sh
bash run_hybird.sh
# =============================================================================
# eval_llama3_fp8_ppl_ablation.py — 全量 PPL / 吞吐 / 显存消融
# =============================================================================
# 测量协议（默认）：--throughput_warmup_windows 2 + CUDA 微预热；与旧脚本对齐可加：
#   --throughput_warmup_windows 0 --no_cuda_warmup
# 降低 tok/s 方差：增大 --num_runs（如 5）
#
# Tokenize：默认 --tokenize_by sample（按数据集样本 encode 再拼接，推荐）
# 与历史 char 切块结果对比：--tokenize_by char
#
# 稀疏侧：--sparse_targets / --sparse_layer_range / --sparse_rules（按后缀细粒度层）/
#         --skip_first_n_layers / --skip_last_n_layers（与层集合取交）
# FP8 侧：--fp8_layer_range / --fp8_skip_first_n_layers / --fp8_skip_last_n_layers
#         --fp8_quant_targets 留空 = 在层规则下量化「全部」剩余 Linear（fp8_only / fp8_act24）
#
# 负对照快捷：--preset neg_o_proj 或 neg_up_proj（会覆盖 --sparse_targets）
#
# 更多示例（按需取消注释后单独跑）：
#   Down+Gate 稀疏：     --sparse_targets mlp.down_proj,mlp.gate_proj --modes act24_only,fp8_act24
#   末层保护：           --skip_last_n_layers 4
#   仅前 1/3 层稀疏：   --sparse_layer_range first_third
#   fp8_act24 只量化部分层： --fp8_quant_targets mlp.gate_proj,self_attn.q_proj
# =============================================================================

# python eval_llama3_fp8_ppl_ablation.py \
#   --model_id /data/sza/model/Meta-Llama-3.1-8B \
#   --dataset wikitext2 \
#   --dataset_path /data/sza/local_dataset \
#   --modes baseline,fp8_only,fp8_down_only,act24_only,fp8_act24 \
#   --seq_lengths 2048 \
#   --num_runs 5 \
#   --tokenize_by sample \
#   --output_json results_wt2.json

# =============================================================================
# sensitivity_scan.py — 逐模块敏感度（需与上同一 conda / GPU；在项目根目录执行）
# =============================================================================
# 含义：每次只改一个 Linear，换为激活 2:4 FP8 稀疏核，在校准短语料上算 Δmean_nll。
# 输出：--output_json，按 delta_mean_nll 降序；越小越适合作为优先替换候选。
#
# 示例（仅扫 down_proj，更快）：
#   python sensitivity_scan.py \
#     --model_id /data/sza/model/Meta-Llama-3.1-8B \
#     --dataset wikitext2 \
#     --dataset_path /data/sza/local_dataset \
#     --sparse_targets mlp.down_proj \
#     --skip_first_n_layers 0 --skip_last_n_layers 0 --sparse_layer_range all \
#     --max_calib_tokens 32768 --max_calib_windows 48 \
#     --output_json sensitivity_ranking.json
#
# 与消融脚本层筛选一致：同上 --skip_first_n_layers / --skip_last_n_layers / --sparse_layer_range
# =============================================================================

