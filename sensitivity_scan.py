#!/usr/bin/env python3
"""
逐模块敏感度扫描：每次只替换**一个** ``nn.Linear`` 为激活 2:4 **FP8 稀疏核**
（与 ``eval_llama3_fp8_ppl_ablation.py`` 中 ``act24_only`` 一致），在短校准集上比较
平均 NLL（与 PPL 同源的滑窗 CE），得到 ``delta_mean_nll``。

分数约定：
  - ``delta_mean_nll = mean_nll_perturbed - mean_nll_baseline``
  - **越大**表示该模块对扰动越敏感（越不宜优先替换）；**越小**越适合作为 Bottom-k% 替换候选。

运行方式（在项目根目录 ``ao/`` 下，与消融脚本共用同一 conda 环境）::

    cd /path/to/ao
    conda activate <your-env>
    export CUDA_VISIBLE_DEVICES=0

    python sensitivity_scan.py \\
        --model_id /path/to/Meta-Llama-3.1-8B \\
        --dataset wikitext2 \\
        --dataset_path /path/to/local_dataset \\
        --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \\
        --skip_first_n_layers 0 \\
        --skip_last_n_layers 0 \\
        --sparse_layer_range all \\
        --max_calib_tokens 32768 \\
        --max_calib_windows 48 \\
        --output_json sensitivity_ranking.json

主要参数说明：
  - ``--sparse_targets``：逗号分隔 FQN 后缀，候选模块集合（与消融脚本相同语法）。
  - ``--skip_first_n_layers`` / ``--skip_last_n_layers`` / ``--sparse_layer_range``：
    与消融中 **稀疏替换** 的层筛选一致，用于缩小扫描范围。
  - ``--sparse_rules``：与 ``eval_llama3_fp8_ppl_ablation.py`` 相同，按后缀单独设层范围。
  - ``--max_calib_tokens``：只使用前 T 个 token；``--max_calib_windows``：最多滑窗数（先达到上限即停）。
  - ``--sequence_length`` / ``--stride``：与评测脚本一致；默认 stride 为 ``max(256, seq_len//4)``。
  - 语料 tokenize 与消融默认一致：内部使用 ``tokenize_by=sample``（按样本 encode 再拼接）。

输出 JSON：
  - ``ranked``：每项含 ``fqn``、``layer_index``、``delta_mean_nll`` 等；按 ``delta_mean_nll`` **降序**。
  - 选「最不敏感」的模块：看列表**末尾**或按 ``delta_mean_nll`` **升序**重排。

依赖：transformers / torchao / CUDA SM90 稀疏核（与 ``eval_llama3_fp8_ppl_ablation.py`` 相同）。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 复用消融脚本中的替换与工具函数
from eval_llama3_fp8_ppl_ablation import (
    FP8IdentitySemiSparseActivationLinear,
    enumerate_sparse_target_fqns,
    _layer_index_from_fqn,
    _parse_suffix_list,
    _set_module_by_fqn,
    load_eval_texts,
    parse_sparse_rules_arg,
    tokenize_corpus_no_truncation,
)


def calib_mean_nll(
    model: nn.Module,
    input_ids_1bt2: torch.Tensor,
    *,
    sequence_length: int,
    stride: int,
    max_windows: int,
    device: str,
) -> tuple[float, int, int]:
    """
    滑窗语言建模平均 NLL（每个预测 token 的平均 loss），与消融脚本 PPL 定义一致。

    Returns:
        (mean_nll, num_windows_used, total_trg_tokens)
    """
    enc = input_ids_1bt2
    total_len = enc.size(1)
    if total_len < sequence_length:
        raise ValueError(f"Corpus length {total_len} < sequence_length {sequence_length}")

    sum_ll = 0.0
    trg_tokens = 0
    win_idx = 0
    for i in range(0, total_len, stride):
        if win_idx >= max_windows:
            break
        begin_loc = max(i + stride - sequence_length, 0)
        end_loc = min(i + stride, total_len)
        trg_len = end_loc - i

        input_ids = enc[:, begin_loc:end_loc]
        target_ids = torch.full_like(input_ids, -100)
        target_ids[:, -trg_len:] = input_ids[:, -trg_len:]

        with torch.inference_mode():
            outputs = model(input_ids, labels=target_ids)
            sum_ll += float((outputs.loss * trg_len).detach().item())

        trg_tokens += trg_len
        win_idx += 1
        if end_loc == total_len:
            break

    mean_nll = sum_ll / trg_tokens if trg_tokens else float("nan")
    return mean_nll, win_idx, trg_tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama 线性层激活稀疏敏感度扫描（单模块替换）")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sequence_length", type=int, default=2048)
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="默认 max(256, sequence_length//4)",
    )
    p.add_argument(
        "--sparse_targets",
        type=str,
        default="mlp.down_proj,mlp.gate_proj,self_attn.q_proj",
        help="逗号分隔 FQN 后缀",
    )
    p.add_argument("--skip_first_n_layers", type=int, default=0)
    p.add_argument("--skip_last_n_layers", type=int, default=0)
    p.add_argument("--sparse_layer_range", type=str, default="all")
    p.add_argument(
        "--sparse_rules",
        type=str,
        default=None,
        help="与 eval_llama3_fp8_ppl_ablation --sparse_rules 相同（suffix:range;...）",
    )
    p.add_argument(
        "--max_calib_tokens",
        type=int,
        default=32768,
        help="校准语料最多使用前这么多 token（截断）",
    )
    p.add_argument(
        "--max_calib_windows",
        type=int,
        default=48,
        help="最多滑窗数（与 max_calib_tokens 同时生效，先达到限制者为准）",
    )
    p.add_argument(
        "--no-fused-row-absmax",
        action="store_true",
        help="稀疏模块内关闭 fused row abs-max",
    )
    p.add_argument("--output_json", type=str, default="sensitivity_ranking.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    stride = args.stride if args.stride is not None else max(256, args.sequence_length // 4)
    suffixes = _parse_suffix_list(args.sparse_targets)
    if not suffixes:
        raise SystemExit("sparse_targets 不能为空")

    sparse_rules = parse_sparse_rules_arg(args.sparse_rules)
    if sparse_rules:
        unknown = sorted(set(sparse_rules) - set(suffixes))
        if unknown:
            raise SystemExit(
                f"--sparse_rules 中的后缀不在 sparse_targets 内: {unknown}"
            )

    print(f"Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading dataset={args.dataset!r} ...")
    texts = load_eval_texts(args.dataset, args.dataset_path)
    input_ids_cpu = tokenize_corpus_no_truncation(
        tokenizer, texts, tokenize_by="sample"
    )
    T = input_ids_cpu.size(1)
    max_t = min(T, args.max_calib_tokens)
    input_ids_cpu = input_ids_cpu[:, :max_t].contiguous()
    print(f"  calib tokens={max_t} (cap={args.max_calib_tokens})")

    device = args.device
    input_ids = input_ids_cpu.to(device)

    if input_ids.size(1) < args.sequence_length:
        raise SystemExit(
            f"校准语料太短: {input_ids.size(1)} < sequence_length {args.sequence_length}"
        )

    print(f"Loading model {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    candidates = enumerate_sparse_target_fqns(
        model,
        suffixes,
        args.skip_first_n_layers,
        args.skip_last_n_layers,
        args.sparse_layer_range,
        sparse_rules=sparse_rules,
    )
    print(f"候选模块数: {len(candidates)} (suffixes={suffixes})")
    if not candidates:
        raise SystemExit("无候选模块，请检查 sparse_targets / layer_range / skip_last_n")

    use_fused = not args.no_fused_row_absmax

    print("Baseline NLL ...")
    t0 = time.perf_counter()
    base_nll, n_win, n_tok = calib_mean_nll(
        model,
        input_ids,
        sequence_length=args.sequence_length,
        stride=stride,
        max_windows=args.max_calib_windows,
        device=device,
    )
    t1 = time.perf_counter()
    print(
        f"  baseline mean_nll={base_nll:.6f} | windows={n_win} | trg_tokens={n_tok} | "
        f"time={t1 - t0:.1f}s"
    )

    results: list[dict[str, Any]] = []
    modules_cache = dict(model.named_modules())

    for fqn in tqdm(candidates, desc="per-module"):
        if fqn not in modules_cache:
            modules_cache = dict(model.named_modules())
        linear = modules_cache[fqn]
        if not isinstance(linear, nn.Linear):
            tqdm.write(f"skip (not Linear): {fqn}")
            continue

        orig = linear
        sparse_mod = FP8IdentitySemiSparseActivationLinear.from_dense(
            orig,
            use_fused_row_absmax=use_fused,
        )
        _set_module_by_fqn(model, fqn, sparse_mod)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        pert_nll, _, _ = calib_mean_nll(
            model,
            input_ids,
            sequence_length=args.sequence_length,
            stride=stride,
            max_windows=args.max_calib_windows,
            device=device,
        )
        delta = pert_nll - base_nll

        _set_module_by_fqn(model, fqn, orig)
        modules_cache = dict(model.named_modules())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()

        li = _layer_index_from_fqn(fqn)
        results.append(
            {
                "fqn": fqn,
                "layer_index": li,
                "mean_nll_baseline": base_nll,
                "mean_nll_perturbed": pert_nll,
                "delta_mean_nll": delta,
            }
        )

    results.sort(key=lambda r: r["delta_mean_nll"], reverse=True)

    out: dict[str, Any] = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "sequence_length": args.sequence_length,
        "stride": stride,
        "max_calib_tokens": max_t,
        "max_calib_windows": args.max_calib_windows,
        "calib_windows_used": n_win,
        "calib_trg_tokens": n_tok,
        "baseline_mean_nll": base_nll,
        "sparse_targets": suffixes,
        "skip_last_n_layers": args.skip_last_n_layers,
        "sparse_layer_range": args.sparse_layer_range,
        "sparse_rules": sparse_rules,
        "metric": "delta_mean_nll = mean_nll_perturbed - mean_nll_baseline; larger => more sensitive",
        "ranked": results,
    }

    path = os.path.abspath(args.output_json)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}")
    print("Top 10 most sensitive (largest delta_mean_nll):")
    for r in results[:10]:
        print(
            f"  d_nll={r['delta_mean_nll']:+.6f}  layer={r['layer_index']}  {r['fqn']}"
        )
    print("Bottom 10 least sensitive (smallest delta):")
    for r in results[-10:]:
        print(
            f"  d_nll={r['delta_mean_nll']:+.6f}  layer={r['layer_index']}  {r['fqn']}"
        )


if __name__ == "__main__":
    main()
