"""
Llama3 推理评测（消融 + 多数据集 + 稳态测速）

在保留 ``eval_llama3_fp8_ppl_test.py`` 的前提下，本脚本补充：

- 实验模式：baseline | fp8_only | fp8_down_only | act24_only | fp8_act24
- 激活 2:4 稀疏：``--sparse_targets`` 指定 FQN 后缀（如 ``mlp.down_proj``、``mlp.gate_proj``、``self_attn.q_proj``）
- 层范围：``--sparse_layer_range``（all / first_third / middle_third / last_third / ``start:end``）与 ``--skip_first_n_layers`` / ``--skip_last_n_layers``（与层集合**取交**）
- **细粒度**：``--sparse_rules suffix:range;...`` 可为每个 ``--sparse_targets`` 后缀单独指定层范围；未写的后缀仍用 ``--sparse_layer_range``
- 多数据集：wikitext2（含本地路径逻辑）、ptb、wikitext103
- 完整测试语料分块 tokenize，避免单次 ``max_length`` 截断导致 PPL 不严谨
- 吞吐计时默认跳过前若干滑窗（与 compile warmup 取较大值），减少 cuDNN/首轮分配噪声；PPL 仍用全量窗口
- 可选每次评测前对首个滑窗做 CUDA 微预热，降低多轮 tok/s 方差
- 同一配置多次运行，报告 tok/s 与 peak mem 的 mean/std、median，以及（>=3 次时）去掉极值后的 trimmed mean
- 可选多组 ``sequence_length`` 敏感性扫描
- 结果写入 JSON（可选）

用法示例::

    export CUDA_VISIBLE_DEVICES=0
    python eval_llama3_fp8_ppl_ablation.py \\
        --model_id meta-llama/Meta-Llama-3.1-8B \\
        --dataset wikitext2 \\
        --modes baseline,fp8_only,act24_only,fp8_act24 \\
        --num_runs 3 \\
        --output_json results.json

细粒度层示例（gate 仅前半、down/q 全层，且与 ``skip_last`` 取交）::

    python eval_llama3_fp8_ppl_ablation.py \\
        --sparse_targets mlp.down_proj,mlp.gate_proj,self_attn.q_proj \\
        --sparse_rules "mlp.gate_proj:0:15;mlp.down_proj:all;self_attn.q_proj:all" \\
        --skip_last_n_layers 4 \\
        --modes fp8_act24 ...

环境变量：未在脚本内硬编码 ``CUDA_VISIBLE_DEVICES``，请在 shell 中自行指定。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import time
import warnings
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset, load_from_disk
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao.ops import (
    rowwise_abs_max_fp32_scale,
    rowwise_scaled_linear_sparse_cutlass_f8f8,
)
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    quantize_,
)
from torchao.quantization.quant_api import _float8_cutlass_quant


def _is_cuda_device(device: str) -> bool:
    """兼容 ``cuda``、``cuda:0`` 等；勿用 ``device == 'cuda'`` 硬比较。"""
    return str(device).startswith("cuda")


# ---------------------------------------------------------------------------
# 与 eval_llama3_fp8_ppl_test.py 保持一致的模块（Llama MLP down_proj + FP8 稀疏核）
# ---------------------------------------------------------------------------


class FP8IdentitySemiSparseActivationLinear(nn.Module):
    """Runtime activation 2:4 + FP8 Linear（Llama ``mlp.down_proj``）。"""

    def __init__(
        self,
        weight: torch.Tensor,
        activation_dtype: torch.dtype = torch.float8_e4m3fn,
        weight_dtype: torch.dtype = torch.float8_e4m3fn,
        *,
        use_fused_row_absmax: bool = True,
    ) -> None:
        super().__init__()
        self.activation_dtype = activation_dtype
        self.use_fused_row_absmax = use_fused_row_absmax

        device = weight.device
        # 在 GPU 上对尚未释放的 BF16 权重再做 FP8 量化会短暂叠加显存，易 OOM。
        # 先在 CPU 上完成 _float8_cutlass_quant，再把 float8 权重与 scale 搬回 GPU。
        w_src = weight.detach()
        if w_src.is_cuda:
            w_src = w_src.cpu()
        w_aqt = _float8_cutlass_quant(w_src, weight_dtype)
        self.wq = w_aqt.tensor_impl.float8_data.to(device, non_blocking=True)
        self.w_scale = w_aqt.tensor_impl.scale.squeeze(-1).to(
            device, non_blocking=True
        )
        inv_max = 1.0 / torch.finfo(activation_dtype).max
        self.register_buffer(
            "_fp8_inv_max",
            torch.tensor(inv_max, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        # sparse24_sm90_sparsify (SM90) requires n_rows % 64 == 0 (stricter than 32).
        block_rows = 64
        pad_rows = (-x_2d.size(0)) % block_rows

        if pad_rows:
            x_pad = torch.zeros(
                (pad_rows, x_2d.size(1)),
                dtype=x_2d.dtype,
                device=x_2d.device,
            )
            x_2d_for_sparse = torch.cat((x_2d, x_pad), dim=0)
        else:
            x_2d_for_sparse = x_2d

        if (
            self.use_fused_row_absmax
            and x_2d_for_sparse.is_cuda
            and x_2d_for_sparse.dtype == torch.bfloat16
        ):
            try:
                x_scale_2d = rowwise_abs_max_fp32_scale(
                    x_2d_for_sparse, self._fp8_inv_max
                )
            except (RuntimeError, NotImplementedError):
                x_scale_2d = (
                    x_2d_for_sparse.abs()
                    .amax(dim=1, keepdim=True)
                    .float()
                    .mul_(self._fp8_inv_max)
                )
        else:
            x_scale_2d = (
                x_2d_for_sparse.abs()
                .amax(dim=1, keepdim=True)
                .float()
                .mul_(self._fp8_inv_max)
            )

        if pad_rows:
            # Padded rows are zeros; keep scale as 1 to avoid 0-scale edge cases.
            x_scale_2d[-pad_rows:] = 1.0

        xq_sparse, x_meta = torch.ops.torchao.sparse24_sm90_sparsify(
            x_2d_for_sparse,
            "cutlass",
            "identity",
            "largest_abs",
            dtype=self.activation_dtype,
            scale=x_scale_2d,
        )

        out_2d = rowwise_scaled_linear_sparse_cutlass_f8f8(
            self.wq,
            self.w_scale,
            xq_sparse,
            x_meta,
            x_scale_2d.view(-1),
            bias=None,
            out_dtype=torch.bfloat16,
        ).t().contiguous()

        if pad_rows:
            out_2d = out_2d[:-pad_rows]

        return out_2d.view(*orig_shape[:-1], out_2d.shape[-1])

    @classmethod
    def from_dense(
        cls,
        linear: nn.Module,
        *,
        use_fused_row_absmax: bool = True,
    ):
        if linear.bias is not None:
            raise NotImplementedError("bias is not supported")
        if linear.weight.dtype != torch.bfloat16:
            raise NotImplementedError("weight dtype must be bf16")
        return cls(linear.weight.data, use_fused_row_absmax=use_fused_row_absmax)


def _set_module_by_fqn(root_module: nn.Module, fqn: str, new_module: nn.Module) -> None:
    parts = fqn.split(".")
    parent = root_module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _num_decoder_layers(model: nn.Module) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise ValueError(
        "无法推断 decoder 层数：需要 ``model.model.layers``（Llama 等）"
    )


def _layer_index_from_fqn(fqn: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", fqn)
    if m:
        return int(m.group(1))
    return None


def _parse_sparse_layer_range(range_str: str, num_layers: int) -> set[int] | None:
    """返回允许替换的层下标集合；``None`` 表示不限制（在 skip_last 内全层）。"""
    rs = range_str.strip().lower()
    if not rs or rs == "all":
        return None
    if rs == "first_third":
        cut = max(1, num_layers // 3)
        return set(range(0, cut))
    if rs == "middle_third":
        a, b = num_layers // 3, 2 * num_layers // 3
        return set(range(a, b)) if a < b else set()
    if rs == "last_third":
        a = 2 * num_layers // 3
        return set(range(a, num_layers))
    if ":" in rs:
        a, b = rs.split(":", 1)
        start, end = int(a.strip()), int(b.strip())
        return set(range(max(0, start), min(num_layers, end)))
    raise ValueError(
        f"Unknown sparse_layer_range={range_str!r}; "
        "use all|first_third|middle_third|last_third|start:end"
    )


def _layer_eligible(
    layer_idx: int,
    num_layers: int,
    skip_first_n_layers: int,
    skip_last_n_layers: int,
    layer_set: set[int] | None,
) -> bool:
    if layer_idx < 0 or layer_idx >= num_layers:
        return False
    if layer_idx < skip_first_n_layers:
        return False
    if layer_idx >= num_layers - skip_last_n_layers:
        return False
    if layer_set is not None and layer_idx not in layer_set:
        return False
    return True


def _parse_suffix_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_sparse_rules_arg(s: str | None) -> dict[str, str] | None:
    """
    解析 ``--sparse_rules``：``suffix:range`` 对，以 ``;`` 或换行分隔。

    ``range`` 语法与 ``--sparse_layer_range`` 相同（``all``、``first_third``、``0:16`` 等）。
    返回 ``None`` 表示未启用细粒度规则。
    """
    if s is None:
        return None
    text = str(s).strip()
    if not text:
        return None
    rules: dict[str, str] = {}
    for chunk in re.split(r"[;\n]+", text):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid --sparse_rules entry {chunk!r}; expected suffix:range "
                "(same vocabulary as --sparse_layer_range)"
            )
        suf, rspec = chunk.split(":", 1)
        suf = suf.strip()
        rspec = rspec.strip()
        if not suf or not rspec:
            raise ValueError(f"Invalid --sparse_rules entry {chunk!r}")
        rules[suf] = rspec
    return rules or None


def _longest_matching_sparse_suffix(
    fqn: str, sparse_suffixes: list[str]
) -> str | None:
    best: str | None = None
    best_len = -1
    for suf in sparse_suffixes:
        if fqn.endswith(suf) and len(suf) > best_len:
            best = suf
            best_len = len(suf)
    return best


def _total_model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _total_linear_param_and_matmul_flops(model: nn.Module) -> tuple[int, int]:
    """所有 ``nn.Linear`` 参数量与 matmul FLOPs 近似（每层前向 2*in*out）。"""
    p, fl = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            p += sum(x.numel() for x in m.parameters())
            fl += 2 * m.in_features * m.out_features
    return p, fl


def _linear_param_and_flops_for_fqns(model: nn.Module, fqns: list[str]) -> tuple[int, int]:
    modules = dict(model.named_modules())
    p, fl = 0, 0
    for fqn in fqns:
        m = modules.get(fqn)
        if isinstance(m, nn.Linear):
            p += sum(x.numel() for x in m.parameters())
            fl += 2 * m.in_features * m.out_features
    return p, fl


def enumerate_sparse_target_fqns(
    model: nn.Module,
    sparse_suffixes: list[str],
    skip_first_n_layers: int,
    skip_last_n_layers: int,
    sparse_layer_range: str,
    sparse_rules: dict[str, str] | None = None,
) -> list[str]:
    """
    枚举待替换为激活 2:4 稀疏核的 ``nn.Linear`` FQN。

    ``sparse_rules``：后缀 -> 层范围字符串；与 ``skip_first_n_layers`` /
    ``skip_last_n_layers`` **取交**。某后缀未出现在 ``sparse_rules`` 中时，
    对该后缀使用 ``sparse_layer_range`` 解析层集合。
    """
    if not sparse_suffixes:
        return []
    n_layers = _num_decoder_layers(model)
    default_layer_set = _parse_sparse_layer_range(sparse_layer_range, n_layers)
    per_suffix_cache: dict[str, set[int] | None] = {}
    if sparse_rules:
        for suf, rspec in sparse_rules.items():
            per_suffix_cache[suf] = _parse_sparse_layer_range(rspec, n_layers)

    def _layer_set_for_suffix(matched_suffix: str) -> set[int] | None:
        if sparse_rules and matched_suffix in sparse_rules:
            return per_suffix_cache[matched_suffix]
        return default_layer_set

    target_fqns: list[str] = []
    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        matched = _longest_matching_sparse_suffix(fqn, sparse_suffixes)
        if matched is None:
            continue
        layer_set = _layer_set_for_suffix(matched)
        li = _layer_index_from_fqn(fqn)
        if li is None:
            if (
                skip_last_n_layers > 0
                or skip_first_n_layers > 0
                or layer_set is not None
            ):
                continue
            target_fqns.append(fqn)
        elif _layer_eligible(
            li, n_layers, skip_first_n_layers, skip_last_n_layers, layer_set
        ):
            target_fqns.append(fqn)
    return sorted(target_fqns)


def replace_linears_with_activation_sparse(
    model: nn.Module,
    *,
    use_fused_row_absmax: bool = True,
    sparse_suffixes: list[str],
    skip_first_n_layers: int = 0,
    skip_last_n_layers: int = 0,
    sparse_layer_range: str = "all",
    sparse_rules: dict[str, str] | None = None,
) -> list[str]:
    """
    将匹配到的 ``nn.Linear`` 换为 ``FP8IdentitySemiSparseActivationLinear``
    （激活 2:4 + FP8 权重/稀疏核；非“无精度损失的纯稀疏”）。

    Returns:
        实际替换的 FQN 列表（与配置一致，便于写 JSON）。
    """
    target_fqns = enumerate_sparse_target_fqns(
        model,
        sparse_suffixes,
        skip_first_n_layers,
        skip_last_n_layers,
        sparse_layer_range,
        sparse_rules=sparse_rules,
    )
    modules = dict(model.named_modules())
    for fqn in target_fqns:
        linear = modules[fqn]
        _set_module_by_fqn(
            model,
            fqn,
            FP8IdentitySemiSparseActivationLinear.from_dense(
                linear,
                use_fused_row_absmax=use_fused_row_absmax,
            ),
        )
        modules = dict(model.named_modules())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return target_fqns


def build_fp8_linear_filter(
    model: nn.Module,
    *,
    fp8_layer_range: str,
    skip_first_n_layers: int,
    skip_last_n_layers: int,
    fp8_suffixes: list[str] | None,
) -> Any:
    """
    ``fp8_suffixes`` 为 ``None`` 或空：在满足层规则时对**所有** ``nn.Linear`` 量化；
    否则仅对后缀匹配的 Linear 量化。
    """
    n_layers = _num_decoder_layers(model)
    layer_set = _parse_sparse_layer_range(fp8_layer_range, n_layers)

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        li = _layer_index_from_fqn(fqn)
        if li is None:
            if (
                skip_first_n_layers > 0
                or skip_last_n_layers > 0
                or layer_set is not None
            ):
                return False
        elif not _layer_eligible(
            li, n_layers, skip_first_n_layers, skip_last_n_layers, layer_set
        ):
            return False
        if fp8_suffixes:
            return any(fqn.endswith(s) for s in fp8_suffixes)
        return True

    return filter_fn


def list_linear_fqns_if(model: nn.Module, filter_fn) -> list[str]:
    return sorted(fqn for fqn, m in model.named_modules() if filter_fn(m, fqn))


def apply_quantize_fp8_with_filter(model: nn.Module, filter_fn) -> None:
    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        filter_fn=filter_fn,
    )


def _coverage_dict(
    sparse_fqns: list[str],
    total_params: int,
    total_linear_p: int,
    total_linear_flops: int,
    sparse_param_flops: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """若已在替换前算好 ``(params, flops)``，传入 ``sparse_param_flops``。"""
    if sparse_param_flops is not None:
        sp, sf = sparse_param_flops
    else:
        sp, sf = 0, 0
    return {
        "num_sparse_replaced": len(sparse_fqns),
        "matched_sparse_fqns": sparse_fqns,
        "sparse_replaced_linear_params": sp,
        "sparse_replaced_linear_params_ratio_total_params": (
            sp / total_params if total_params else 0.0
        ),
        "sparse_replaced_linear_matmul_flops": sf,
        "sparse_replaced_linear_matmul_flops_ratio_all_linear": (
            sf / total_linear_flops if total_linear_flops else 0.0
        ),
        "total_model_params": total_params,
        "total_linear_params": total_linear_p,
        "total_linear_matmul_flops": total_linear_flops,
    }


def apply_mode(
    model: nn.Module,
    mode: str,
    *,
    use_fused_row_absmax: bool,
    sparse_targets: list[str],
    skip_first_n_layers: int,
    skip_last_n_layers: int,
    sparse_layer_range: str,
    sparse_rules: dict[str, str] | None,
    fp8_targets: list[str],
    fp8_layer_range: str,
    fp8_skip_first_n_layers: int,
    fp8_skip_last_n_layers: int,
    fp8_quant_targets: list[str] | None,
) -> dict[str, Any]:
    """
    Returns:
        含 ``description``、``matched_sparse_fqns``、``matched_fp8_fqns``、``coverage`` 等。
    """
    st = ",".join(sparse_targets)
    sk = skip_last_n_layers
    sf = skip_first_n_layers
    lr = sparse_layer_range.strip() or "all"
    sr_note = f" sparse_rules={sparse_rules!r}" if sparse_rules else ""
    total_params = _total_model_param_count(model)
    total_linear_p, total_linear_flops = _total_linear_param_and_matmul_flops(model)

    def _base_out(
        desc: str,
        matched_sparse_fqns: list[str],
        fp8_fqns: list[str] | None,
        *,
        sparse_param_flops: tuple[int, int] | None = None,
        fp8_param_flops: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        if sparse_param_flops is None:
            sparse_param_flops = (
                _linear_param_and_flops_for_fqns(model, matched_sparse_fqns)
                if matched_sparse_fqns
                else (0, 0)
            )
        cov = _coverage_dict(
            matched_sparse_fqns,
            total_params,
            total_linear_p,
            total_linear_flops,
            sparse_param_flops=sparse_param_flops,
        )
        if fp8_param_flops is not None:
            fp, ffl = fp8_param_flops
            cov["fp8_quantized_linear_params"] = fp
            cov["fp8_quantized_linear_params_ratio_total_params"] = (
                fp / total_params if total_params else 0.0
            )
            cov["fp8_quantized_linear_matmul_flops"] = ffl
            cov["fp8_quantized_linear_matmul_flops_ratio_all_linear"] = (
                ffl / total_linear_flops if total_linear_flops else 0.0
            )
        return {
            "description": desc,
            "num_sparse_replaced": len(matched_sparse_fqns),
            "matched_sparse_fqns": matched_sparse_fqns,
            "matched_fp8_fqns": fp8_fqns if fp8_fqns is not None else [],
            "coverage": cov,
        }

    if mode == "baseline":
        return _base_out(
            "BF16 baseline (no FP8 / no activation 2:4 sparse kernel)",
            [],
            None,
        )

    if mode == "fp8_only":
        fq_suffix = fp8_quant_targets
        flt = build_fp8_linear_filter(
            model,
            fp8_layer_range=fp8_layer_range,
            skip_first_n_layers=fp8_skip_first_n_layers,
            skip_last_n_layers=fp8_skip_last_n_layers,
            fp8_suffixes=fq_suffix,
        )
        fp8_list = list_linear_fqns_if(model, flt)
        fp8_pf = _linear_param_and_flops_for_fqns(model, fp8_list)
        apply_quantize_fp8_with_filter(model, flt)
        return _base_out(
            "fp8_only: Per-row FP8 on nn.Linear per fp8_* layer/suffix rules",
            [],
            fp8_list,
            fp8_param_flops=fp8_pf,
        )

    if mode == "fp8_down_only":
        flt = build_fp8_linear_filter(
            model,
            fp8_layer_range=fp8_layer_range,
            skip_first_n_layers=fp8_skip_first_n_layers,
            skip_last_n_layers=fp8_skip_last_n_layers,
            fp8_suffixes=fp8_targets,
        )
        fp8_list = list_linear_fqns_if(model, flt)
        fp8_pf = _linear_param_and_flops_for_fqns(model, fp8_list)
        apply_quantize_fp8_with_filter(model, flt)
        return _base_out(
            f"fp8_down_only: FP8 only on suffixes {fp8_targets} (+fp8 layer rules)",
            [],
            fp8_list,
            fp8_param_flops=fp8_pf,
        )

    if mode == "act24_only":
        sparse_fqns = enumerate_sparse_target_fqns(
            model,
            sparse_targets,
            skip_first_n_layers,
            skip_last_n_layers,
            sparse_layer_range,
            sparse_rules=sparse_rules,
        )
        sparse_pf = _linear_param_and_flops_for_fqns(model, sparse_fqns)
        replaced = replace_linears_with_activation_sparse(
            model,
            use_fused_row_absmax=use_fused_row_absmax,
            sparse_suffixes=sparse_targets,
            skip_first_n_layers=skip_first_n_layers,
            skip_last_n_layers=skip_last_n_layers,
            sparse_layer_range=sparse_layer_range,
            sparse_rules=sparse_rules,
        )
        desc = (
            f"act24_only: activation 2:4 **FP8 sparse kernel** (not pure sparsity) on "
            f"[{st}], default_layer_range={lr}, skip_first={sf}, skip_last_n={sk};"
            f"{sr_note} no global FP8"
        )
        return _base_out(
            desc,
            replaced,
            None,
            sparse_param_flops=sparse_pf,
        )

    if mode == "fp8_act24":
        sparse_fqns = enumerate_sparse_target_fqns(
            model,
            sparse_targets,
            skip_first_n_layers,
            skip_last_n_layers,
            sparse_layer_range,
            sparse_rules=sparse_rules,
        )
        sparse_pf = _linear_param_and_flops_for_fqns(model, sparse_fqns)
        replaced = replace_linears_with_activation_sparse(
            model,
            use_fused_row_absmax=use_fused_row_absmax,
            sparse_suffixes=sparse_targets,
            skip_first_n_layers=skip_first_n_layers,
            skip_last_n_layers=skip_last_n_layers,
            sparse_layer_range=sparse_layer_range,
            sparse_rules=sparse_rules,
        )
        fq_suffix = fp8_quant_targets
        flt = build_fp8_linear_filter(
            model,
            fp8_layer_range=fp8_layer_range,
            skip_first_n_layers=fp8_skip_first_n_layers,
            skip_last_n_layers=fp8_skip_last_n_layers,
            fp8_suffixes=fq_suffix,
        )
        fp8_list = list_linear_fqns_if(model, flt)
        fp8_pf = _linear_param_and_flops_for_fqns(model, fp8_list)
        apply_quantize_fp8_with_filter(model, flt)
        desc = (
            f"fp8_act24: sparse [{st}] (default_range={lr}, skip_first={sf}, skip_last={sk})"
            f"{sr_note} + FP8 on Linear per fp8_quant_targets & fp8 layer rules"
        )
        return _base_out(
            desc,
            replaced,
            fp8_list,
            sparse_param_flops=sparse_pf,
            fp8_param_flops=fp8_pf,
        )

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# 数据：多数据集 + 分块 tokenize（无整体截断）
# ---------------------------------------------------------------------------


def _load_wikitext2_local_or_hub(dataset_path: str | None):
    if dataset_path:
        try:
            dataset = load_from_disk(dataset_path)
            if hasattr(dataset, "keys") and "test" in dataset.keys():
                dataset = dataset["test"]
            return dataset
        except Exception:
            test_file = os.path.join(dataset_path, "wiki.test.raw")
            if os.path.exists(test_file):
                return load_dataset(
                    "text", data_files={"test": test_file}, split="test"
                )
    # 本地路径不可用或 dataset_path 未设置时，从 Hub 拉取 wikitext-2
    return load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        trust_remote_code=True,
    )


def _load_ptb_local_or_hub(dataset_path: str | None):
    """
    PTB（``ptb_text_only`` / penn_treebank test）：优先本地，避免离线环境无法访问 Hub。

    支持：

    - ``load_from_disk(path)`` 保存的 Dataset（列名 ``sentence``，或 dict 含 ``test``）
    - 目录下 ``ptb.test.txt`` / ``test.txt``：一行一句（与常见 PTB 文本格式一致）
    - 单文件 ``*.txt``：一行一句
    """
    if dataset_path:
        p = os.path.abspath(dataset_path)
        if os.path.isfile(p) and p.lower().endswith(".txt"):
            return load_dataset("text", data_files={"test": p}, split="test")
        # 目录下优先读 ptb.test.txt：避免与同目录的 WikiText load_from_disk 缓存混淆
        if os.path.isdir(p):
            for fname in ("ptb.test.txt", "test.txt", "ptb.test"):
                test_file = os.path.join(p, fname)
                if os.path.isfile(test_file):
                    return load_dataset(
                        "text", data_files={"test": test_file}, split="test"
                    )
        try:
            dataset = load_from_disk(p)
            if hasattr(dataset, "keys") and "test" in dataset.keys():
                dataset = dataset["test"]
            return dataset
        except Exception:
            pass
    return load_dataset("ptb_text_only", "penn_treebank", split="test")


def _load_wikitext103_local_or_hub(dataset_path: str | None):
    """WikiText-103 test：``load_from_disk`` 或目录下文本（``wiki.test.raw`` 等）。"""
    if dataset_path:
        p = os.path.abspath(dataset_path)
        if os.path.isfile(p) and p.lower().endswith(".txt"):
            print(f"[wikitext103] loading single-file test: {p}", flush=True)
            return load_dataset("text", data_files={"test": p}, split="test")
        if os.path.isdir(p):
            for fname in (
                "wiki.test.raw",
                "wikitext103.test.txt",
                "test.txt",
            ):
                test_file = os.path.join(p, fname)
                if os.path.isfile(test_file):
                    print(
                        f"[wikitext103] loading test from: {test_file}",
                        flush=True,
                    )
                    return load_dataset(
                        "text", data_files={"test": test_file}, split="test"
                    )
        try:
            dataset = load_from_disk(p)
            if hasattr(dataset, "keys") and "test" in dataset.keys():
                dataset = dataset["test"]
            print(f"[wikitext103] load_from_disk OK: {p}", flush=True)
            return dataset
        except Exception:
            pass
    print(
        "[wikitext103] local path unset or unusable; loading from Hugging Face Hub",
        flush=True,
    )
    return load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="test",
        trust_remote_code=True,
    )


def load_eval_texts(dataset_name: str, dataset_path: str | None) -> list[str]:
    name = dataset_name.lower().strip()
    if name in ("wikitext2", "wikitext-2", "wt2"):
        ds = _load_wikitext2_local_or_hub(dataset_path)
        col = "text"
    elif name in ("ptb", "penn_treebank"):
        ds = _load_ptb_local_or_hub(dataset_path)
        col = "sentence" if "sentence" in ds.column_names else "text"
    elif name in ("wikitext103", "wikitext-103", "wt103"):
        ds = _load_wikitext103_local_or_hub(dataset_path)
        col = "text"
    else:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. "
            "Use: wikitext2 | ptb | wikitext103"
        )

    texts = [str(t).strip() for t in ds[col] if str(t).strip()]
    if not texts:
        raise ValueError(f"Empty corpus after filtering ({dataset_name})")
    return texts


def tokenize_corpus_no_truncation(
    tokenizer,
    texts: list[str],
    *,
    max_chunk_chars: int = 200_000,
    tokenize_by: str = "sample",
) -> torch.Tensor:
    """
    将语料拼成 ``[1, T]``，避免单次 ``max_length`` 截断。

    - ``tokenize_by=sample``（默认）：**按样本**分别 ``encode`` 再拼接，避免在 UTF-8
      字符块边界切开导致子词与真实文档不一致（PPL 更严谨）。
    - ``tokenize_by=char``：按 ``max_chunk_chars`` 字符块编码（旧行为，便于对照）。
    """
    if tokenize_by not in ("sample", "char"):
        raise ValueError("tokenize_by must be 'sample' or 'char'")

    if tokenize_by == "sample":
        pieces: list[torch.Tensor] = []
        buf: list[str] = []
        buf_len = 0
        for t in texts:
            s = str(t).strip()
            if not s:
                continue
            add_len = len(s) + 2
            if buf and buf_len + add_len > max_chunk_chars:
                chunk_txt = "\n\n".join(buf)
                enc = tokenizer(chunk_txt, return_tensors="pt", add_special_tokens=False)
                pieces.append(enc["input_ids"][0])
                buf = []
                buf_len = 0
            buf.append(s)
            buf_len += add_len
        if buf:
            chunk_txt = "\n\n".join(buf)
            enc = tokenizer(chunk_txt, return_tensors="pt", add_special_tokens=False)
            pieces.append(enc["input_ids"][0])
        if not pieces:
            raise ValueError("Empty text after tokenize_by=sample")
        ids = torch.cat(pieces, dim=0)
        return ids.unsqueeze(0)

    full = "\n\n".join(texts)
    if not full:
        raise ValueError("Empty text after join")

    pieces: list[torch.Tensor] = []
    for start in range(0, len(full), max_chunk_chars):
        chunk = full[start : start + max_chunk_chars]
        enc = tokenizer(chunk, return_tensors="pt", add_special_tokens=False)
        pieces.append(enc["input_ids"][0])

    ids = torch.cat(pieces, dim=0)
    return ids.unsqueeze(0)


# ---------------------------------------------------------------------------
# PPL + 测速（warmup 窗口 + 多轮）
# ---------------------------------------------------------------------------


def _infer_timing_skip_windows(
    throughput_warmup_windows: int,
    compile_warmup_windows: int,
) -> int:
    """吞吐计时时跳过的前若干滑窗数：取 general 与 compile 相关配置的较大值。"""
    return max(int(throughput_warmup_windows), int(compile_warmup_windows))


def _cuda_micro_warmup_first_window(
    model: nn.Module,
    enc: torch.Tensor,
    *,
    sequence_length: int,
    stride: int,
    device: str,
    num_passes: int,
) -> None:
    """在首个滑窗上重复 forward，缓解 cuDNN/缓存冷启动带来的 tok/s 波动。"""
    if num_passes <= 0 or not _is_cuda_device(device) or not torch.cuda.is_available():
        return
    total_len = enc.size(1)
    if total_len < sequence_length:
        return
    i = 0
    begin_loc = max(i + stride - sequence_length, 0)
    end_loc = min(i + stride, total_len)
    input_ids = enc[:, begin_loc:end_loc]
    trg_len = end_loc - i
    target_ids = torch.full_like(input_ids, -100)
    target_ids[:, -trg_len:] = input_ids[:, -trg_len:]
    for _ in range(num_passes):
        with torch.inference_mode():
            model(input_ids, labels=target_ids)
    torch.cuda.synchronize()


def _trimmed_mean(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    s = sorted(values)
    return statistics.mean(s[1:-1])


def lm_sliding_window_eval(
    model,
    input_ids_1bt2: torch.Tensor,
    *,
    sequence_length: int,
    stride: int,
    verbose: bool,
    device: str,
    timing_skip_windows: int = 0,
) -> dict[str, Any]:
    """
    滑窗负对数似然 PPL；前 ``timing_skip_windows`` 个滑窗不计入吞吐计时（PPL 仍统计全量）。

    PPL 定义与 ``eval_llama3_fp8_ppl_test.py::wiki2_eval`` 一致：
    ``ppl = exp(sum(loss_i * trg_len_i) / end_loc)``，其中 ``end_loc`` 为触发 ``break``
    时的窗口右端点（评测到语料末尾为止）。

    ``input_ids_1bt2``: shape ``[1, T]``，已在 ``device``。
    """
    enc = input_ids_1bt2
    total_len = enc.size(1)
    if total_len < sequence_length:
        raise ValueError(
            f"Corpus length {total_len} < sequence_length {sequence_length}"
        )

    lls: list[torch.Tensor] = []
    timed_tokens = 0
    start_time: float | None = None
    end_time: float | None = None
    win_idx = 0

    if torch.cuda.is_available() and _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats()

    if verbose:
        print(
            f"Computing PPL (seq_len={sequence_length}, stride={stride}, "
            f"timing_skip_windows={timing_skip_windows})..."
        )

    end_loc_for_ppl = total_len

    num_windows_est = 0
    for i in range(0, total_len, stride):
        end_loc = min(i + stride, total_len)
        num_windows_est += 1
        if end_loc == total_len:
            break
    if timing_skip_windows >= num_windows_est and num_windows_est > 0:
        warnings.warn(
            f"timing_skip_windows={timing_skip_windows} >= num_sliding_windows="
            f"{num_windows_est}，tok/s 将为 nan；请减小 --throughput_warmup_windows "
            f"或 --compile_warmup_windows",
            UserWarning,
            stacklevel=2,
        )

    for i in tqdm(range(0, total_len, stride)):
        begin_loc = max(i + stride - sequence_length, 0)
        end_loc = min(i + stride, total_len)
        trg_len = end_loc - i

        input_ids = enc[:, begin_loc:end_loc]
        ntok = input_ids.numel()

        target_ids = torch.full_like(input_ids, -100)
        target_ids[:, -trg_len:] = input_ids[:, -trg_len:]

        # 稳态测速：从第 timing_skip_windows 个窗口开始计时（含该窗口）
        if win_idx == timing_skip_windows:
            if _is_cuda_device(device):
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            if torch.cuda.is_available() and _is_cuda_device(device):
                torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

        if win_idx >= timing_skip_windows:
            timed_tokens += ntok

        win_idx += 1

        if end_loc == total_len:
            end_loc_for_ppl = end_loc
            if start_time is not None:
                if _is_cuda_device(device):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
            break

    # 若未进入计时（例如窗口总数 <= warmup），补全 end_time 以便返回 nan tok/s
    if start_time is not None and end_time is None:
        if _is_cuda_device(device):
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    if start_time is None or end_time is None:
        timed_duration_sec = float("nan")
        tokens_per_sec = float("nan")
    else:
        timed_duration_sec = float(max(end_time - start_time, 1e-9))
        tokens_per_sec = timed_tokens / timed_duration_sec

    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available() and _is_cuda_device(device)
        else 0.0
    )

    ppl = float(torch.exp(torch.stack(lls).sum() / end_loc_for_ppl))

    if verbose:
        print(
            f"\n[Result] PPL: {ppl:.4f} | Timed tok/s: {tokens_per_sec:.2f} "
            f"| Peak Mem: {peak_mem_gb:.2f} GB"
        )

    return {
        "ppl": ppl,
        "tokens_per_sec": float(tokens_per_sec),
        "peak_mem_gb": float(peak_mem_gb),
        "num_windows": win_idx,
        "total_tokens": int(total_len),
        "timed_tokens": int(timed_tokens),
        "timed_duration_sec": float(timed_duration_sec),
        "ppl_denominator_end_loc": int(end_loc_for_ppl),
        "timing_skip_windows": int(timing_skip_windows),
    }


def run_eval_repeated(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    sequence_length: int,
    stride: int,
    device: str,
    num_runs: int,
    timing_skip_windows: int,
    cuda_warmup_passes: int,
    verbose: bool,
) -> dict[str, Any]:
    """多次运行：PPL 取第一次（应相同）；tok/s 与 mem 报告 mean/std/median/trimmed。"""
    ppls: list[float] = []
    speeds: list[float] = []
    mems: list[float] = []
    durations: list[float] = []
    first_detail: dict[str, Any] | None = None

    for r in range(num_runs):
        if verbose and num_runs > 1:
            print(f"--- Run {r + 1}/{num_runs} ---")

        _cuda_micro_warmup_first_window(
            model,
            input_ids,
            sequence_length=sequence_length,
            stride=stride,
            device=device,
            num_passes=cuda_warmup_passes,
        )

        detail = lm_sliding_window_eval(
            model,
            input_ids,
            sequence_length=sequence_length,
            stride=stride,
            verbose=verbose and num_runs == 1,
            device=device,
            timing_skip_windows=timing_skip_windows,
        )
        ppls.append(detail["ppl"])
        speeds.append(detail["tokens_per_sec"])
        mems.append(detail["peak_mem_gb"])
        durations.append(float(detail["timed_duration_sec"]))
        if r == 0:
            first_detail = detail

    assert first_detail is not None
    tok_trim = _trimmed_mean(speeds)
    mem_trim = _trimmed_mean(mems)
    out = {
        "ppl": ppls[0],
        "ppl_runs": ppls,
        "tokens_per_sec_runs": speeds,
        "peak_mem_gb_runs": mems,
        "timed_duration_sec_runs": durations,
        "tokens_per_sec_mean": statistics.mean(speeds),
        "tokens_per_sec_std": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "tokens_per_sec_median": statistics.median(speeds),
        "tokens_per_sec_trimmed_mean": tok_trim,
        "peak_mem_gb_mean": statistics.mean(mems),
        "peak_mem_gb_std": statistics.stdev(mems) if len(mems) > 1 else 0.0,
        "peak_mem_gb_median": statistics.median(mems),
        "peak_mem_gb_trimmed_mean": mem_trim,
        "num_runs": num_runs,
        **{
            k: v
            for k, v in first_detail.items()
            if k
            not in (
                "ppl",
                "tokens_per_sec",
                "peak_mem_gb",
                "timed_duration_sec",
            )
        },
    }
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Llama3: PPL 消融评测（FP8 / 激活2:4 / 组合）+ 多数据集 + 稳态测速"
    )
    p.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="本地数据路径（可选；不传则各数据集从 Hub 下载）。"
        "wikitext2：目录含 wiki.test.raw 或 load_from_disk；"
        "ptb：目录含 ptb.test.txt/test.txt、或单文件 .txt（一行一句）、或 HF save_to_disk；"
        "wikitext103：目录含 wiki.test.raw / wikitext103.test.txt / test.txt，"
        "或单文件 .txt、或 HF save_to_disk。",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        help="wikitext2 | ptb | wikitext103",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="baseline,fp8_only,act24_only,fp8_act24",
        help="逗号分隔：baseline,fp8_only,fp8_down_only,act24_only,fp8_act24",
    )
    p.add_argument(
        "--sparse_targets",
        type=str,
        default="mlp.down_proj",
        help="激活2:4替换对象：逗号分隔 FQN 后缀，例如 "
        "mlp.down_proj 或 mlp.down_proj,mlp.gate_proj,self_attn.q_proj",
    )
    p.add_argument(
        "--skip_first_n_layers",
        type=int,
        default=0,
        help="对 decoder 前 N 层不替换稀疏模块（early dense）",
    )
    p.add_argument(
        "--skip_last_n_layers",
        type=int,
        default=0,
        help="对 decoder 最后 N 层不替换稀疏模块（末层保护）",
    )
    p.add_argument(
        "--sparse_layer_range",
        type=str,
        default="all",
        help="仅在这些层下标上替换稀疏模块：all | first_third | middle_third | "
        "last_third | start:end（如 0:10）",
    )
    p.add_argument(
        "--sparse_rules",
        type=str,
        default=None,
        help="按 FQN 后缀细粒度层范围；与 --skip_first_n_layers / --skip_last_n_layers 取交。"
        "格式：suffix1:range1;suffix2:range2（range 语法同 --sparse_layer_range）。"
        "未列出的后缀仍使用 --sparse_layer_range。键必须是 --sparse_targets 的子集。",
    )
    p.add_argument(
        "--preset",
        type=str,
        choices=("none", "neg_o_proj", "neg_up_proj"),
        default="none",
        help="快捷负对照：覆盖 --sparse_targets（neg_o_proj / neg_up_proj）",
    )
    p.add_argument(
        "--fp8_targets",
        type=str,
        default="mlp.down_proj",
        help="fp8_down_only 模式：对匹配此后缀的 nn.Linear 做 FP8，逗号分隔",
    )
    p.add_argument(
        "--fp8_layer_range",
        type=str,
        default="all",
        help="FP8 量化层范围（与 sparse_layer_range 语法相同）；用于 fp8_only / fp8_down_only / fp8_act24",
    )
    p.add_argument(
        "--fp8_skip_first_n_layers",
        type=int,
        default=0,
        help="前 N 层 decoder 不做 FP8 量化",
    )
    p.add_argument(
        "--fp8_skip_last_n_layers",
        type=int,
        default=0,
        help="最后 N 层 decoder 不做 FP8 量化",
    )
    p.add_argument(
        "--fp8_quant_targets",
        type=str,
        default="",
        help="fp8_only / fp8_act24：仅对这些 FQN 后缀的 Linear 做 FP8；"
        "留空表示在满足 fp8 层规则下量化**全部**剩余 Linear",
    )
    p.add_argument(
        "--seq_lengths",
        type=str,
        default="2048",
        help="逗号分隔，例如 512,1024,2048",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="滑窗步长；默认 max(256, sequence_length//4)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_runs", type=int, default=1, help="吞吐/显存重复次数（>=1）")
    p.add_argument(
        "--compile",
        action="store_true",
        help="对整模型 torch.compile(max-autotune)",
    )
    p.add_argument(
        "--compile_warmup_windows",
        type=int,
        default=None,
        help="与 --throughput_warmup_windows 取 max：前若干个滑窗不计入 tok/s；"
        "未指定且启用 --compile 时默认为 2，否则为 0",
    )
    p.add_argument(
        "--throughput_warmup_windows",
        type=int,
        default=2,
        help="通用吞吐预热：前若干个滑窗不计入 tok/s（PPL 仍用全语料）；"
        "与 compile_warmup 取较大值，建议 1～4",
    )
    p.add_argument(
        "--cuda_warmup_passes",
        type=int,
        default=2,
        help="每次重复评测前，在首个滑窗上额外 forward 的次数（仅 CUDA；0 关闭）",
    )
    p.add_argument(
        "--no_cuda_warmup",
        action="store_true",
        help="等价于 --cuda_warmup_passes 0",
    )
    p.add_argument(
        "--no-fused-row-absmax",
        action="store_true",
        help="关闭 fused row abs-max（与旧脚本一致）",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="若设置且 mode 含 fp8，则尝试 save_pretrained（可能因自定义张量失败）",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="将完整结果写入 JSON 文件",
    )
    p.add_argument(
        "--max_chunk_chars",
        type=int,
        default=200_000,
        help="sample 模式下单次拼接的字符预算；char 模式为字符块大小",
    )
    p.add_argument(
        "--tokenize_by",
        type=str,
        choices=("sample", "char"),
        default="sample",
        help="sample=按数据集样本分别 encode 再拼接（推荐）；char=按字符块（旧行为）",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_seq_lengths(s: str) -> list[int]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return [int(x) for x in parts]


_PRESET_SPARSE: dict[str, list[str]] = {
    "neg_o_proj": ["self_attn.o_proj"],
    "neg_up_proj": ["mlp.up_proj"],
}


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seq_lengths = parse_seq_lengths(args.seq_lengths)
    if args.preset != "none":
        sparse_targets = list(_PRESET_SPARSE[args.preset])
    else:
        sparse_targets = _parse_suffix_list(args.sparse_targets)
    fp8_targets = _parse_suffix_list(args.fp8_targets)
    fq8_qt = _parse_suffix_list(args.fp8_quant_targets)
    fp8_quant_targets: list[str] | None = fq8_qt if fq8_qt else None

    sparse_rules = parse_sparse_rules_arg(args.sparse_rules)
    if sparse_rules:
        unknown = sorted(set(sparse_rules) - set(sparse_targets))
        if unknown:
            raise ValueError(
                f"--sparse_rules 中的后缀不在 sparse_targets 内: {unknown}. "
                f"sparse_targets={sparse_targets}"
            )

    needs_sparse = any(m in ("act24_only", "fp8_act24") for m in modes)
    if needs_sparse and not sparse_targets:
        raise ValueError(
            "act24_only / fp8_act24 需要非空 sparse_targets，或使用 --preset neg_o_proj / neg_up_proj"
        )
    if any(m == "fp8_down_only" for m in modes) and not fp8_targets:
        raise ValueError("fp8_down_only 需要 --fp8_targets")

    compile_warmup = args.compile_warmup_windows
    if compile_warmup is None:
        compile_warmup = 2 if args.compile else 0
    cuda_warmup_passes = 0 if args.no_cuda_warmup else max(0, args.cuda_warmup_passes)
    timing_skip = _infer_timing_skip_windows(
        args.throughput_warmup_windows, compile_warmup
    )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        gpu_name = "CPU"
        print("Warning: CUDA 不可用，速度/显存指标仅供参考")

    print(
        f"Throughput timing: skip_first_windows={timing_skip} "
        f"(throughput={args.throughput_warmup_windows}, compile_cfg={compile_warmup}) | "
        f"cuda_micro_warmup_passes={cuda_warmup_passes}"
    )
    print(
        f"Sparse strategy: targets={sparse_targets} | "
        f"default_layer_range={args.sparse_layer_range!r} | "
        f"skip_first_n={args.skip_first_n_layers} | "
        f"skip_last_n={args.skip_last_n_layers}"
        + (f" | per_suffix_rules={sparse_rules!r}" if sparse_rules else "")
    )
    print(
        f"FP8 strategy: fp8_layer_range={args.fp8_layer_range!r} | "
        f"fp8_skip_first={args.fp8_skip_first_n_layers} | "
        f"fp8_skip_last={args.fp8_skip_last_n_layers} | "
        f"fp8_quant_targets={'(all Linear under rules)' if fp8_quant_targets is None else fp8_quant_targets}"
    )
    print(f"fp8_down_only suffixes (fp8_targets): {fp8_targets}")
    print(f"tokenize_by={args.tokenize_by!r}")

    print(f"Loading tokenizer/model config from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading dataset={args.dataset!r} ...")
    texts = load_eval_texts(args.dataset, args.dataset_path)
    print(f"  num_segments={len(texts)}, building full token sequence ...")
    input_ids_cpu = tokenize_corpus_no_truncation(
        tokenizer,
        texts,
        max_chunk_chars=args.max_chunk_chars,
        tokenize_by=args.tokenize_by,
    )
    device = args.device
    input_ids = input_ids_cpu.to(device)
    print(f"  total_tokens={input_ids.size(1)}")

    all_results: list[dict[str, Any]] = []

    for sequence_length in seq_lengths:
        stride = args.stride if args.stride is not None else max(256, sequence_length // 4)
        if input_ids.size(1) < sequence_length:
            print(
                f"Skip seq_len={sequence_length}: corpus shorter than sequence_length."
            )
            continue

        for mode in modes:
            print("\n" + "=" * 60)
            print(f"Mode: {mode} | seq_len={sequence_length} | stride={stride}")
            print("=" * 60)

            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                dtype=torch.bfloat16,
                device_map=device,
            )

            setup = apply_mode(
                model,
                mode,
                use_fused_row_absmax=not args.no_fused_row_absmax,
                sparse_targets=sparse_targets,
                skip_first_n_layers=args.skip_first_n_layers,
                skip_last_n_layers=args.skip_last_n_layers,
                sparse_layer_range=args.sparse_layer_range,
                sparse_rules=sparse_rules,
                fp8_targets=fp8_targets,
                fp8_layer_range=args.fp8_layer_range,
                fp8_skip_first_n_layers=args.fp8_skip_first_n_layers,
                fp8_skip_last_n_layers=args.fp8_skip_last_n_layers,
                fp8_quant_targets=fp8_quant_targets,
            )
            desc = setup["description"]
            n_rep = setup["num_sparse_replaced"]
            cov = setup["coverage"]
            print(f"Setup: {desc}")
            print(
                f"  matched_sparse_fqns={len(setup['matched_sparse_fqns'])} | "
                f"matched_fp8_fqns={len(setup['matched_fp8_fqns'])}"
            )
            rp = cov.get("sparse_replaced_linear_params_ratio_total_params")
            if rp is not None:
                print(
                    f"  coverage: sparse_param_ratio_total={rp:.4f} | "
                    f"sparse_flops_ratio_linear={cov.get('sparse_replaced_linear_matmul_flops_ratio_all_linear', 0):.4f}"
                )
            if n_rep > 0:
                print(f"  Replaced {n_rep} Linear(s) with activation 2:4 FP8 sparse kernel.")
            elif mode in ("act24_only", "fp8_act24"):
                print(
                    "  Warning: 0 Linear matched; check --sparse_targets, "
                    "--sparse_layer_range, --sparse_rules, --skip_first/--skip_last"
                )

            if args.save_path and mode in ("fp8_only", "fp8_down_only", "fp8_act24"):
                try:
                    model.save_pretrained(args.save_path, safe_serialization=False)
                    tokenizer.save_pretrained(args.save_path)
                    print(f"Saved to {args.save_path}")
                except Exception as e:
                    print(f"[Warning] save_pretrained failed: {e}")

            if args.compile:
                print("torch.compile(mode=max-autotune) ...")
                model = torch.compile(model, mode="max-autotune")

            metrics = run_eval_repeated(
                model,
                tokenizer,
                input_ids,
                sequence_length=sequence_length,
                stride=stride,
                device=device,
                num_runs=max(1, args.num_runs),
                timing_skip_windows=timing_skip,
                cuda_warmup_passes=cuda_warmup_passes,
                verbose=True,
            )

            cov_ratio = cov.get(
                "sparse_replaced_linear_params_ratio_total_params", 0.0
            )
            row = {
                "mode": mode,
                "preset": args.preset,
                "dataset": args.dataset,
                "sequence_length": sequence_length,
                "stride": stride,
                "gpu_name": gpu_name,
                "model_id": args.model_id,
                "compile": bool(args.compile),
                "throughput_warmup_windows": args.throughput_warmup_windows,
                "compile_warmup_windows": compile_warmup,
                "timing_skip_windows": timing_skip,
                "cuda_warmup_passes": cuda_warmup_passes,
                "fused_row_absmax": not args.no_fused_row_absmax,
                "sparse_targets": sparse_targets,
                "skip_first_n_layers": args.skip_first_n_layers,
                "skip_last_n_layers": args.skip_last_n_layers,
                "sparse_layer_range": args.sparse_layer_range,
                "sparse_rules": sparse_rules,
                "fp8_targets": fp8_targets,
                "fp8_layer_range": args.fp8_layer_range,
                "fp8_skip_first_n_layers": args.fp8_skip_first_n_layers,
                "fp8_skip_last_n_layers": args.fp8_skip_last_n_layers,
                "fp8_quant_targets": fp8_quant_targets or [],
                "tokenize_by": args.tokenize_by,
                "description": desc,
                "num_sparse_replaced": setup["num_sparse_replaced"],
                "matched_sparse_fqns": setup["matched_sparse_fqns"],
                "matched_fp8_fqns": setup["matched_fp8_fqns"],
                "coverage": cov,
                "coverage_sparse_param_ratio_total": cov_ratio,
                **metrics,
            }
            all_results.append(row)

            tok_tm = metrics.get("tokens_per_sec_trimmed_mean")
            tok_extra = (
                f" trimmed={tok_tm:.2f}"
                if tok_tm is not None
                else ""
            )
            print(
                f"--> PPL={metrics['ppl']:.4f} | "
                f"tok/s median={metrics['tokens_per_sec_median']:.2f} "
                f"mean={metrics['tokens_per_sec_mean']:.2f}"
                f"±{metrics['tokens_per_sec_std']:.2f}{tok_extra} | "
                f"peak_mem_gb median={metrics['peak_mem_gb_median']:.2f} "
                f"mean={metrics['peak_mem_gb_mean']:.2f}"
                f"±{metrics['peak_mem_gb_std']:.2f}"
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "torch_version": torch.__version__,
        "gpu_name": gpu_name,
        "model_id": args.model_id,
        "dataset": args.dataset,
        "sparse_rules": sparse_rules,
        "results": all_results,
    }

    if all_results:
        print("\n" + "=" * 80)
        print("Summary (all runs)")
        print("=" * 80)
        hdr = (
            f"{'mode':<14} {'ds':<12} {'seq':>5} {'PPL':>10} "
            f"{'cov_spr':>8} {'tok/s_med':>11} {'tok/s_mean±std':>20} {'mem_med':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for row in all_results:
            ts_line = (
                f"{row['tokens_per_sec_mean']:.2f}"
                f"±{row['tokens_per_sec_std']:.2f}"
            )
            cr = row.get("coverage_sparse_param_ratio_total", 0.0) or 0.0
            print(
                f"{row['mode']:<14} {str(row['dataset']):<12} "
                f"{row['sequence_length']:>5} {row['ppl']:>10.4f} "
                f"{cr:>8.4f} {row['tokens_per_sec_median']:>11.2f} {ts_line:>20} "
                f"{row['peak_mem_gb_median']:>8.2f}"
            )
        print("=" * 80)

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {out_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
