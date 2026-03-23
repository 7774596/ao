"""
change_to_relu.py

将 Llama 3 的 SwiGLU 激活函数渐进式替换为 ReLU，
以在 down_proj 输入处得到自然的激活稀疏性，使其与 2:4 稀疏 GEMM 内核兼容。

背景
----
原始 SwiGLU（Llama 3 默认）：
    H = silu(gate_proj(x)) ⊙ up_proj(x)      # 平滑，激活值无自然零点

ReLU（目标状态）：
    H = relu(gate_proj(x)) ⊙ up_proj(x)     # gate ≤ 0 精确输出 0

替换后 H 具有大量自然零值，可直接被 sparse24_sm90_sparsify 利用，
不需要 eval_llama3_fp8_ppl.py 中的强制 top-2/4 裁剪。

训练策略
--------
1. 平滑过渡（alpha 调度）
   H = α · relu(gate)⊙up + (1-α) · silu(gate)⊙up
   alpha 在 [0, ramp_steps] 内从 0 线性增长到 1，之后保持 1。

2. 2:4 激活稀疏正则化（可选）
   对每 4 个相邻激活值中绝对值最小的 2 个施加 L1 惩罚，
   主动推动 H 向 2:4 结构靠拢。

3. 可训练参数
   默认只更新 MLP 层（gate/up/down），冻结 Attention 和 LN，
   以减少显存占用。可用 --train_all_params 开启全参数微调。

用法示例
--------
python change_to_relu.py \\
    --model_id  /data/sza/model/Meta-Llama-3.1-8B \\
    --dataset_path /data/sza/local_dataset \\
    --output_dir /data/sza/model/Meta-Llama-3.1-8B-ReluSparse \\
    --max_steps 2000 \\
    --ramp_steps 1000 \\
    --lr 2e-5 \\
    --sparse_reg 0.01 \\
    --eval_interval 200
"""

from __future__ import annotations
import math
import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# HuggingFace 镜像站（优先于代码中任何网络请求生效）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "warning")
# 将 HuggingFace 缓存移动到 /data 分区（home 分区空间不足）
os.environ["HF_HOME"] = "/data/sza/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/data/sza/.cache/huggingface/datasets"

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 核心 MLP 模块：SwiGLU → ReLU 平滑过渡
# ─────────────────────────────────────────────────────────────────────────────

class LlamaMLPReluTransition(nn.Module):
    """
    Llama MLP，支持从 SwiGLU 平滑过渡到 ReLU。

    alpha=0: 纯 SwiGLU（原始行为）
    alpha=1: 纯 ReLU（目标行为，输出自然稀疏）

    混合公式：
        mixed_act = α · relu(gate) + (1-α) · silu(gate)
        H = mixed_act ⊙ up
        output = down_proj(H)
    """

    def __init__(
        self,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        alpha: float = 0.0,
        relu_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        # alpha 用 buffer 存储，可被外部调度器直接修改而无需优化器感知
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        # relu_scale: 方差对齐系数，使 relu(gate) 的方差与 silu(gate) 匹配（现支持 per-channel tensor）
        if isinstance(relu_scale, torch.Tensor):
            self.register_buffer("relu_scale", relu_scale.clone().detach())
        else:
            self.register_buffer("relu_scale", torch.tensor(float(relu_scale)))
        
        # 新增：用于吸收 ReLU 带来的 Mean Shift (初始化为0，校准时赋值)
        self.down_proj_bias = nn.Parameter(torch.zeros(down_proj.out_features, device=down_proj.weight.device, dtype=down_proj.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)   # [B, T, intermediate_size]
        up   = self.up_proj(x)     # [B, T, intermediate_size]

        # 确保 alpha 和 relu_scale 与 gate 的 dtype 一致
        alpha = self.alpha.to(gate.dtype)
        relu_scale = self.relu_scale.to(gate.dtype)

        # SwiGLU 路径（alpha=0 时退化为原始 Llama 3）
        swiglu = F.silu(gate)          # 平滑激活，无自然零点

        # ReLU 路径（alpha=1 时：gate ≤ 0 → 精确 0，天然稀疏）
        # relu_scale 将 relu 的方差对齐到 silu，防止 alpha 增大时激活幅度爆炸
        relu_act = F.relu(gate) * relu_scale

        # 线性混合，grad 对两条路径均可通过
        mixed = alpha * relu_act + (1.0 - alpha) * swiglu

        h = mixed * up  # down_proj 的输入，随 alpha→1 越来越稀疏
        self.last_h = h

        # 核心修复：把均值便宜补偿加载到 down_proj 输出上
        # 不影响 h 的稀疏度，又抵消了残差流里的 Mean Shift
        out = self.down_proj(h)
        if alpha > 0:
            out = out + alpha * self.down_proj_bias
        return out

    @classmethod
    def from_llama_mlp(cls, mlp: nn.Module, alpha: float = 0.0,
                       relu_scale: float = 1.0,) -> "LlamaMLPReluTransition":
        """从 transformers LlamaMLP 原地转换，共享权重（无拷贝）。"""
        return cls(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
            alpha=alpha,
            relu_scale=relu_scale,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. 模型替换工具
# ─────────────────────────────────────────────────────────────────────────────

def estimate_relu_scale_and_bias(model: nn.Module, calib_ids: torch.Tensor,
                                  target_fqns: List[str]) -> Tuple[Dict, Dict]:
    """
    二阶段校准：
    1. 估算每层的独有 relu_scale（方差对齐）
    2. 计算每层 down_proj 的均值偏移 target_bias = E[down(silu) - down(relu*scale)]

    参数：
        model: 待校准的模型
        calib_ids: 校准用的input_ids
        target_fqns: MLP层的完整路径列表（确保顺序一致）
    返回：
        (layer_scales_dict, layer_biases_dict)
    """
    import math

    # ── 根据FQN获取MLP模块（确保顺序一致）────────────────────────
    mlps = []
    for fqn in target_fqns:
        parts = fqn.split(".")
        mod = model
        for p in parts:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            else:
                mod = mod.get_submodule(p)
        mlps.append(mod)

    print(f"  [校准] 准备对 {len(mlps)} 个 MLP 层进行校准...")

    # ── 第一阶段：计算每层 per-channel scale ──
    var_silu_list, var_relu_list = [], []
    def hook_var(module, inp, out):
        x = inp[0]
        with torch.no_grad():
            gate = module.gate_proj(x).float() # [batch, seq, intermediate_size]
            # 计算每层的 per-channel sum of squares (因为均值为 0)
            # 或者直接算 var(dim=(0,1))
            var_silu_list.append(F.silu(gate).var(dim=(0,1)))
            var_relu_list.append(F.relu(gate).var(dim=(0,1)))

    handles = []
    for mod in mlps:
        handles.append(mod.register_forward_hook(hook_var))

    model.eval()
    with torch.no_grad():
        ids = calib_ids[:64] if len(calib_ids) >= 64 else calib_ids
        _ = model(ids)

    for h in handles:
        h.remove()
    handles.clear()

    layer_scales = {}
    if not var_silu_list:
        print("  [校准警告] 未收集到方差数据，使用默认 scale=1.0")
        for i in range(len(mlps)):
            layer_scales[i] = torch.tensor(1.0, device=mlps[i].down_proj.weight.device, dtype=mlps[i].down_proj.weight.dtype)
    else:
        for i in range(len(mlps)):
            # Per-channel scale
            scale = torch.sqrt(var_silu_list[i] / (var_relu_list[i] + 1e-8))
            # 限制极值，避免某些死去或者不活跃神经元产生超大放大系数
            scale = torch.clamp(scale, min=0.1, max=5.0)
            layer_scales[i] = scale.to(device=mlps[i].down_proj.weight.device, dtype=mlps[i].down_proj.weight.dtype)

    print(f"  [校准] 成功计算 {len(layer_scales)} 层的独有 relu_scale。")

    # ── 第二阶段：计算每层补偿 down_bias ──
    layer_biases = {}
    def make_bias_hook(mod_id):
        def hook_bias(module, inp, out):
            x = inp[0]
            with torch.no_grad():
                gate = module.gate_proj(x).float()
                up = module.up_proj(x).float()

                # 取出这层对应的 scale，转换为正确类型
                scale = layer_scales[mod_id].to(module.down_proj.weight.dtype)

                # 计算两种激活下的 down_proj 输出
                # shape: [batch, seq, hidden]
                h_silu = F.silu(gate).to(module.down_proj.weight.dtype) * up.to(module.down_proj.weight.dtype)
                h_relu = F.relu(gate).to(module.down_proj.weight.dtype) * scale * up.to(module.down_proj.weight.dtype)

                out_silu = module.down_proj(h_silu)
                out_relu = module.down_proj(h_relu)

                # 均值偏移：对 batch 和 seq 取平均 -> [hidden]
                diff = (out_silu - out_relu).mean(dim=(0,1))
                layer_biases[mod_id] = diff
        return hook_bias

    for i, mod in enumerate(mlps):
        handles.append(mod.register_forward_hook(make_bias_hook(i)))

    with torch.no_grad():
        _ = model(ids)

    for h in handles:
        h.remove()
    model.train()

    print(f"  [校准] 成功计算 {len(layer_biases)} 层 down_proj_bias 补偿向量。")
    return layer_scales, layer_biases


def replace_all_mlp(model: nn.Module, alpha: float = 0.0,
                    calib_ids: Optional[torch.Tensor] = None) -> int:
    """
    将模型中所有 mlp 子模块替换为 LlamaMLPReluTransition。
    若提供 calib_ids，则通过 calibration 估算 relu_scale 和 relu_bias；
    否则使用理论默认值。
    返回替换的层数。
    """
    # ── 统一获取MLP模块的方式（确保顺序一致）────────────────────────
    target_fqns = [
        fqn
        for fqn, mod in model.named_modules()
        if fqn.endswith(".mlp") and hasattr(mod, "gate_proj")
    ]
    print(f"  找到 {len(target_fqns)} 个 MLP 层待替换")

    if calib_ids is not None:
        print("  估算 relu_scale（方差对齐）...")
        layer_scales, layer_biases = estimate_relu_scale_and_bias(model, calib_ids, target_fqns)
    else:
        # 理论值：gate~N(0,1) 时 scale=0.5，保守估算
        # 获取模型 dtype 和 device
        first_mlp = None
        for fqn, mod in model.named_modules():
            if fqn.endswith(".mlp") and hasattr(mod, "gate_proj"):
                first_mlp = mod
                break
        dtype = first_mlp.gate_proj.weight.dtype if first_mlp else torch.bfloat16
        device = first_mlp.gate_proj.weight.device if first_mlp else "cuda"
        layer_scales = {i: torch.tensor(0.5, device=device, dtype=dtype) for i in range(len(target_fqns))}
        layer_biases = {}
        print(f"  [注意] 未提供 calib_ids，使用理论默认值")

    for idx, fqn in enumerate(target_fqns):
        parts = fqn.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old_mlp = getattr(parent, parts[-1])
        new_mlp = LlamaMLPReluTransition.from_llama_mlp(
            old_mlp, alpha=alpha, relu_scale=layer_scales[idx])

        # 应用校准得到的bias补偿
        if idx in layer_biases:
            new_mlp.down_proj_bias.data = layer_biases[idx].to(device=new_mlp.down_proj_bias.device, dtype=new_mlp.down_proj_bias.dtype)
            print(f"  [层{idx}] 应用 down_proj_bias 补偿，norm={layer_biases[idx].norm():.4f}")

        setattr(parent, parts[-1], new_mlp)
    return len(target_fqns)


def set_alpha(model: nn.Module, alpha: float) -> None:
    """统一设置所有 LlamaMLPReluTransition 的 alpha buffer。"""
    for mod in model.modules():
        if isinstance(mod, LlamaMLPReluTransition):
            mod.alpha.fill_(alpha)


def get_all_relu_mlps(model: nn.Module) -> List[LlamaMLPReluTransition]:
    return [m for m in model.modules() if isinstance(m, LlamaMLPReluTransition)]


# ─────────────────────────────────────────────────────────────────────────────
# 3. 2:4 激活稀疏正则化损失
# ─────────────────────────────────────────────────────────────────────────────

def sparse24_activation_penalty(h: torch.Tensor) -> torch.Tensor:
    """
    2:4 激活稀疏正则化。

    对 down_proj 的输入张量 h（即 mixed * up）的每 4 个相邻元素中，
    对绝对值最小的 2 个施加 L1 惩罚 —— 推动它们趋近 0，
    从而主动构造 2:4 稀疏结构。

    h: [..., K]  任意前缀维度，K 会被 pad 到 4 的倍数
    返回标量 penalty（已归一化）
    """
    x = h.reshape(-1, h.shape[-1])    # [M, K]
    K = x.shape[1]
    pad = (4 - K % 4) % 4
    if pad:
        x = F.pad(x, (0, pad))
    groups = x.reshape(-1, 4)          # [M * K', 4]

    # 每组找绝对值最小的 2 个，直接 L1 惩罚（使它们向 0 靠拢）
    abs_g = groups.abs()
    # topk(k=2, largest=False) → 最小 2 个的索引
    _, bot2_idx = abs_g.topk(2, dim=1, largest=False, sorted=False)
    penalty = groups.gather(1, bot2_idx).abs().mean()
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# 4. 激活稀疏度监控钩子
# ─────────────────────────────────────────────────────────────────────────────

class ActivationSparsityMonitor:
    """
    在 LlamaMLPReluTransition.forward 中挂钩，统计 down_proj 输入（H）的稀疏度。
    用于训练过程中实时观测自然稀疏率的变化。
    """

    def __init__(self) -> None:
        self._handles: list = []
        self.sparsity_records: List[float] = []

    def attach(self, model: nn.Module) -> None:
        for mod in model.modules():
            if isinstance(mod, LlamaMLPReluTransition):
                h = mod.register_forward_hook(self._hook)
                self._handles.append(h)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # def _hook(self, module: LlamaMLPReluTransition, inputs, output) -> None:
    #     # 在 hook 中重新计算 H（不影响梯度图，仅统计）
    #     with torch.no_grad():
    #         x = inputs[0]
    #         gate = module.gate_proj(x)
    #         up   = module.up_proj(x)
    #         swiglu = F.silu(gate)
    #         relu_act = F.relu(gate)
    #         mixed  = module.alpha * relu + (1.0 - module.alpha) * swiglu
    #         h = mixed * up
    #         zero_frac = (h.abs() < 1e-6).float().mean().item()
    #         self.sparsity_records.append(zero_frac)
    def _hook(self, module, inputs, output):
        with torch.no_grad():
            if hasattr(module, 'last_h') and module.last_h is not None:
                h = module.last_h
                zero_frac = (h.abs() < 1e-6).float().mean().item()
                self.sparsity_records.append(zero_frac)

    def mean_sparsity(self) -> float:
        if not self.sparsity_records:
            return 0.0
        val = sum(self.sparsity_records) / len(self.sparsity_records)
        self.sparsity_records.clear()
        return val

import itertools
from datasets import load_dataset, concatenate_datasets, load_from_disk
import glob as glob_module

def load_train_data(tokenizer,
                    seq_len: int = 2048,
                    max_train_tokens: int = 1_000_000_000,
                    dataset_path: str = "/data/sza/local_dataset"):
    """
    离线数据加载：优先使用本地 FineWeb-Edu + UltraChat
    """
    # ── 1. 加载本地 FineWeb-Edu parquet 文件 ──
    parquet_dir = os.path.join(dataset_path, "fineweb-edu", "sample-10BT", "sample", "10BT")
    parquet_files = sorted(glob_module.glob(os.path.join(parquet_dir, "*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"未找到 FineWeb-Edu parquet 文件: {parquet_dir}/*.parquet")

    print(f"\n[数据加载] 找到 {len(parquet_files)} 个 FineWeb-Edu parquet 文件")

    # ── 2. 尝试加载本地 UltraChat ──
    ultrachat_path = os.path.join(dataset_path, "ultrachat_200k")
    has_ultrachat = os.path.exists(ultrachat_path)

    if has_ultrachat:
        print(f"[数据加载] 找到 UltraChat 本地缓存: {ultrachat_path}")
    else:
        print(f"[数据加载] 未找到 UltraChat，将仅使用 FineWeb-Edu")

    # ── 3. 计算数据配比 ──
    fw_ratio = 0.7 if has_ultrachat else 1.0
    fw_tokens = int(max_train_tokens * fw_ratio)

    # ── 4. 加载并处理 FineWeb-Edu ──
    target_files = max(1, fw_tokens // 100_000_000)
    files_to_load = parquet_files[:min(target_files, len(parquet_files))]

    print(f"  加载 FineWeb-Edu: 前 {len(files_to_load)} 个文件 (目标 {fw_tokens/1e6:.0f}M tokens)")

    all_ids = []
    total_tokens = 0

    for i, pq_file in enumerate(files_to_load):
        if total_tokens >= fw_tokens:
            break

        if i % 2 == 0:  # 每2个文件打印一次进度
            print(f"  FineWeb [{i+1}/{len(files_to_load)}] {os.path.basename(pq_file)}...")

        ds = load_dataset("parquet", data_files={"train": pq_file}, split="train")

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=seq_len,
                add_special_tokens=False
            )

        tokenized = ds.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["text"])
        all_tokens = []
        # for item in tokenized:
        #     ids = item["input_ids"]
        #     if len(ids) == seq_len:
        #         all_ids.append(ids)
        #         total_tokens += seq_len
        #         if total_tokens >= fw_tokens:
        #             break
        for item in tokenized:
            all_tokens.extend(item["input_ids"] + [tokenizer.eos_token_id])

        # 按 seq_len 切块
        all_ids = [
            all_tokens[i:i+seq_len]
            for i in range(0, len(all_tokens) - seq_len, seq_len)
        ]
        del ds, tokenized

    print(f"  FineWeb-Edu 完成: {len(all_ids)} 块 ({total_tokens/1e6:.1f}M tokens)")

    # ── 5. 加载并处理 UltraChat (如果可用) ──
    if has_ultrachat:
        chat_tokens = max_train_tokens - total_tokens
        print(f"\n  加载 UltraChat (目标 {chat_tokens/1e6:.0f}M tokens)...")

        try:
            # 尝试从本地加载
            chat_ds = load_from_disk(ultrachat_path)
            if "train_sft" in chat_ds:
                chat_ds = chat_ds["train_sft"]

            # 检查是否已经格式化（已有'text'列）
            if "text" in chat_ds.column_names:
                print("  UltraChat 已格式化，跳过转换步骤")
            elif "messages" in chat_ds.column_names:
                # 需要转换为 Llama-3 格式
                print("  转换 UltraChat 为 Llama-3 格式...")

                def format_llama3_chat(example):
                    text = "<|begin_of_text|>"
                    for msg in example["messages"]:
                        role = msg["role"]
                        content = msg["content"]
                        text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                    return {"text": text}

                chat_ds = chat_ds.map(format_llama3_chat, num_proc=4, remove_columns=chat_ds.column_names)
            else:
                raise ValueError(f"UltraChat 数据集格式未知，列: {chat_ds.column_names}")

            # Tokenize
            def tokenize_chat(examples):
                return tokenizer(examples["text"], truncation=True, max_length=seq_len, add_special_tokens=False)

            chat_tokenized = chat_ds.map(tokenize_chat, batched=True, num_proc=4, remove_columns=["text"])

            chat_count = 0
            for item in chat_tokenized:
                ids = item["input_ids"]
                if len(ids) == seq_len:
                    all_ids.append(ids)
                    total_tokens += seq_len
                    chat_count += 1
                    if total_tokens >= max_train_tokens:
                        break

            print(f"  UltraChat 完成: {chat_count} 块 ({chat_count*seq_len/1e6:.1f}M tokens)")
            del chat_ds, chat_tokenized

        except Exception as e:
            print(f"  [警告] UltraChat 加载失败: {e}")
            print("  将仅使用 FineWeb-Edu")

    # ── 6. 打乱并转换为 tensor ──
    print(f"\n  打乱数据...")
    import random
    random.seed(42)
    random.shuffle(all_ids)

    ids_tensor = torch.tensor(all_ids, dtype=torch.long)
    N = len(ids_tensor)
    print(f"\n[数据加载完成] {N} 块 × {seq_len} tokens = {N * seq_len / 1e6:.1f}M tokens\n")
    return ids_tensor

def load_eval_data(dataset_path: str, tokenizer,
                   sequence_length: int = 2048,
                   stride: int = 512,
                   device: str = "cuda"):
    """
    PPL 评估固定使用 wikitext-103 test split，保证结果可横向比较。
    """
    local_path = os.path.join(dataset_path, "wikitext-103")
    if os.path.exists(local_path):
        try:
            ds = load_from_disk(local_path)
            split = ds["test"] if "test" in ds else None
            if split is not None:
                enc = tokenizer(
                    "\n\n".join([t for t in split["text"] if t.strip()]),
                    return_tensors="pt", max_length=131072, truncation=True,
                )
                enc["input_ids"] = enc["input_ids"].to(device)
                return enc
        except Exception:
            pass

    split = load_dataset("wikitext", "wikitext-103-raw-v1",
                         split="test")
    enc = tokenizer(
        "\n\n".join([t for t in split["text"] if t.strip()]),
        return_tensors="pt", max_length=131072, truncation=True,
    )
    enc["input_ids"] = enc["input_ids"].to(device)
    return enc




# ─────────────────────────────────────────────────────────────────────────────
# 5.5 KL 蒸馏损失
# ─────────────────────────────────────────────────────────────────────────────

def kl_distill_loss(student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    temperature: float = 2.0,
                    chunk_size: int = 256) -> torch.Tensor:
    """
    序列级 KL 蒸馏损失（forward KL: teacher || student）。

    沿 token 维度分块计算，避免一次性将 [B, T, V] float32 张量全展开，
    显著降低峰值显存（对 V=128256 词表尤其重要）。

    公式：
        L_KL = τ² · KL(softmax(t_logits/τ) || log_softmax(s_logits/τ))

    参数：
        student_logits: [B, T, V]
        teacher_logits: [B, T, V]，教师输出（梯度会在此截断）
        temperature:    温度系数
        chunk_size:     每次处理的 token 数，越小显存越省，默认 256
    返回：
        标量 loss（可反传到 student）
    """
    B, T, V = student_logits.shape
    total_kl = torch.tensor(0.0, device=student_logits.device, dtype=torch.float32)
    n_chunks = 0

    for i in range(0, T, chunk_size):
        s_chunk = (student_logits[:, i:i+chunk_size, :] / temperature).float()
        t_chunk = (teacher_logits[:, i:i+chunk_size, :] / temperature).float().detach()

        log_p_s = F.log_softmax(s_chunk, dim=-1)
        p_t     = F.softmax(t_chunk,   dim=-1)

        # reduction="sum" 后手动归一化，等价于 batchmean
        kl_chunk = F.kl_div(log_p_s, p_t, reduction="sum")
        total_kl = total_kl + kl_chunk
        n_chunks += s_chunk.shape[1]

        del s_chunk, t_chunk, log_p_s, p_t

    # 归一化：除以 B*T（batchmean 语义）
    return (total_kl / (B * T)) * (temperature ** 2)

# ─────────────────────────────────────────────────────────────────────────────
# 6. PPL 评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_ppl(model: nn.Module, encodings, sequence_length: int = 2048,
             stride: int = 512) -> float:
    model.eval()
    lls = []
    input_ids = encodings["input_ids"]
    end_loc_last = 0
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - sequence_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i
        chunk = input_ids[:, begin_loc:end_loc]
        target = chunk.clone()
        target[:, :-trg_len] = -100
        out = model(chunk, labels=target)
        lls.append(out.loss * trg_len)
        end_loc_last = end_loc
        if end_loc == input_ids.size(1):
            break
    ppl = float(torch.exp(torch.stack(lls).sum() / end_loc_last))
    model.train()
    return ppl


# ─────────────────────────────────────────────────────────────────────────────
# 7. Alpha 调度器
# ─────────────────────────────────────────────────────────────────────────────

# class AlphaScheduler:
#     """
#     线性 alpha 调度：
#       step  0            → alpha = 0.0  (纯 SwiGLU)
#       step  ramp_steps   → alpha = 1.0  (纯 ReLU)
#       step >ramp_steps   → alpha = 1.0  (保持)
#     """

#     def __init__(self, model: nn.Module, ramp_steps: int) -> None:
#         self.model = model
#         self.ramp_steps = ramp_steps

#     def step(self, current_step: int) -> float:
#         if self.ramp_steps <= 0:
#             alpha = 1.0
#         else:
#             alpha = min(1.0, current_step / self.ramp_steps)
#         set_alpha(self.model, alpha)
#         return alpha
class AlphaScheduler:
    def __init__(self, model, ramp_steps):
        self.model = model
        self.ramp_steps = ramp_steps

    def step(self, current_step):
        if self.ramp_steps <= 0:
            alpha = 1.0
        elif current_step >= self.ramp_steps:
            alpha = 1.0
        else:
            # 余弦调度：前期慢，中期快，后期慢
            progress = current_step / self.ramp_steps
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        set_alpha(self.model, alpha)
        return alpha

# ─────────────────────────────────────────────────────────────────────────────
# 8. 训练主循环
# ─────────────────────────────────────────────────────────────────────────────

def train(args, model: nn.Module, tokenizer, teacher_model: Optional[nn.Module] = None,
          train_ids: Optional[torch.Tensor] = None) -> None:
    device = next(model.parameters()).device  # 自动检测模型所在设备
    print(f"\n[训练配置] 学生模型设备: {device}")
    if teacher_model is not None:
        teacher_device = next(teacher_model.parameters()).device
        print(f"[训练配置] 教师模型设备: {teacher_device}")
    print(f"[训练配置] 数据将在训练时从CPU传输到 {device}\n")

    # ── 冻结非 MLP 参数（可选）──────────────────────────────────────────────
    if not args.train_all_params:
        for name, param in model.named_parameters():
            # 只保留 mlp 层可训练；embedding/norm/attention 全部冻结
            if ".mlp." not in name:
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # ── 数据 ────────────────────────────────────────────────────────────────
    if train_ids is None:
        print("加载训练集...")
        train_ids = load_train_data(
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_train_tokens=args.max_train_tokens,
            dataset_path=args.dataset_path
        )
        print(f"  训练块数: {len(train_ids)}")
    else:
        print(f"使用预加载的训练数据: {len(train_ids)} 块")

    # 数据留在CPU，训练时按需传输到GPU

    print("加载验证集...")
    eval_enc = load_eval_data(args.dataset_path, tokenizer, device=device)

    # ── 优化器 & 调度 ────────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    alpha_scheduler = AlphaScheduler(model, ramp_steps=args.ramp_steps)

    # ── 稀疏度监控 ───────────────────────────────────────────────────────────
    sparsity_monitor = ActivationSparsityMonitor()
    if args.monitor_sparsity:
        sparsity_monitor.attach(model)

    # ── 初始 PPL ────────────────────────────────────────────────────────────
    print("\n计算初始 PPL（alpha=0，纯 SwiGLU）...")
    ppl_init = eval_ppl(model, eval_enc)
    print(f"  初始 PPL: {ppl_init:.4f}")

    # ── 训练循环 ────────────────────────────────────────────────────────────
    model.train()

    # 计算有效batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"\n[梯度累积配置]")
    print(f"  实际 batch_size: {args.batch_size}")
    print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"  有效 batch_size: {effective_batch_size}")
    print(f"  每步处理 tokens: {args.batch_size * args.seq_len:,}")
    print(f"  每次更新处理 tokens: {effective_batch_size * args.seq_len:,}\n")

    step = 0
    total_loss = 0.0
    total_lm_loss = 0.0
    total_ce_loss = 0.0
    total_kl_loss = 0.0
    total_sparse_loss = 0.0
    N = len(train_ids)
    pbar = tqdm(total=args.max_steps, desc="ReLU-ification 微调")

    print(f"\n开始微调：max_steps={args.max_steps}, ramp_steps={args.ramp_steps}, "
          f"sparse_reg={args.sparse_reg}, lr={args.lr}")

    optimizer.zero_grad()  # 初始化梯度

    while step < args.max_steps:
        # ── 梯度累积循环 ─────────────────────────────────────────────────
        alpha = alpha_scheduler.step(step)
        for accumulation_step in range(args.gradient_accumulation_steps):
            # 随机采样一个 batch
            idx = torch.randint(0, N, (args.batch_size,))
            input_ids = train_ids[idx].to(device)          # [B, seq_len]
            labels    = input_ids.clone()

            # alpha 调度
            # alpha = alpha_scheduler.step(step)
            effective_sparse_reg = args.sparse_reg * alpha

            # ── 前向 ─────────────────────────────────────────────────────────
            outputs = model(input_ids, labels=labels)
            lm_loss = outputs.loss

            # ── KL 蒸馏损失（若提供教师模型）────────────────────────────────
            ce_loss_val = lm_loss.item()   # 记录纯 CE loss，供日志使用
            kl_loss_val = 0.0
            if teacher_model is not None and args.distill_lambda > 0:
                # 1. 把输入数据送到教师模型所在的卡 (cuda:1 = 物理卡3)
                input_ids_teacher = input_ids.to("cuda:1")

                with torch.no_grad():
                    # 2. 教师模型在 cuda:1 前向传播
                    teacher_out = teacher_model(input_ids_teacher)

                # 3. 把教师的 logits 拉回到学生模型所在的卡 (cuda:0)
                # 关键优化：立即转移到CPU，避免在GPU上保留完整logits
                teacher_logits = teacher_out.logits.cpu()

                # 4. 释放教师模型的显存
                del teacher_out, input_ids_teacher

                # 5. 计算 KL 散度（teacher_logits在CPU，会按需传输）
                kl_loss = kl_distill_loss(
                    outputs.logits, teacher_logits.to(outputs.logits.device),
                    temperature=args.distill_temp,
                    chunk_size=128  # 更小的chunk size
                )

                # 显式清理显存
                del teacher_logits

                # # 动态蒸馏权重：如果随 alpha 降低到 0，会导致在 alpha=1 (纯 ReLU) 最不稳定时丧失教师引导，发生崩溃。
                # # 应该保持固定的蒸馏权重来规制强行替换导致的分布急剧偏移。
                effective_distill_lambda = args.distill_lambda
                
                lm_loss = (1.0 - effective_distill_lambda) * lm_loss + effective_distill_lambda * kl_loss
                kl_loss_val = kl_loss.item()

            #新增
            sparse_penalty_val = 0.0
            if effective_sparse_reg > 0 and alpha > 0.1:
                relu_mlps = get_all_relu_mlps(model)
                h_list = [m.last_h for m in relu_mlps if hasattr(m, 'last_h') and m.last_h is not None]
                if h_list:
                    sparse_penalty = torch.stack(
                        [sparse24_activation_penalty(h) for h in h_list]
                    ).mean()
                    loss = lm_loss + effective_sparse_reg * sparse_penalty
                    sparse_penalty_val = sparse_penalty.item()
                else:
                    loss = lm_loss
                # 注意：不清除 last_h，让 SparsityMonitor 可以读取
                # last_h 会在下一次 forward 时自然被覆盖
            else:
                loss = lm_loss

            # ── 反向传播（累积梯度）─────────────────────────────────────────
            # 除以累积步数，确保梯度尺度正确
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # 只在累积循环的最后一步记录loss（每个optimizer step记录一次）
            if accumulation_step == args.gradient_accumulation_steps - 1:
                total_loss += loss.item() * args.gradient_accumulation_steps
                total_lm_loss += lm_loss.item()
                total_ce_loss += ce_loss_val
                total_kl_loss += kl_loss_val
                total_sparse_loss += sparse_penalty_val

        # ── 参数更新（累积完成后）─────────────────────────────────────────
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # for m in get_all_relu_mlps(model):
        #     m.last_h = None

        step += 1
        pbar.update(1)

        # ── 日志 ─────────────────────────────────────────────────────────
        if step % args.log_interval == 0:
            avg_loss   = total_loss      / args.log_interval
            avg_lm     = total_lm_loss   / args.log_interval
            avg_ce     = total_ce_loss   / args.log_interval
            avg_kl     = total_kl_loss   / args.log_interval
            avg_sparse = total_sparse_loss / args.log_interval
            sparsity   = sparsity_monitor.mean_sparsity() if args.monitor_sparsity else 0.0
            total_loss = total_lm_loss = total_ce_loss = total_kl_loss = total_sparse_loss = 0.0

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ce":   f"{avg_ce:.4f}",
                "kl":   f"{avg_kl:.4f}",
                "α":    f"{alpha:.3f}",
                "zeros": f"{sparsity:.2%}",
            })

        # ── 周期性 PPL 评估 ───────────────────────────────────────────────
        if step % args.eval_interval == 0:
            ppl = eval_ppl(model, eval_enc)
            effective_lambda = args.distill_lambda if teacher_model else 0.0
            print(f"\n[step {step:5d}] PPL={ppl:.4f}  α={alpha:.3f}  "
                  f"λ_distill={effective_lambda:.3f}  lr={lr_scheduler.get_last_lr()[0]:.2e}")
            model.train()

    pbar.close()

    # ── 关闭监控 ──────────────────────────────────────────────────────────
    if args.monitor_sparsity:
        sparsity_monitor.detach()

    # ── 最终评估 ──────────────────────────────────────────────────────────
    # 获取当前 alpha 值进行评估
    final_alpha = get_all_relu_mlps(model)[0].alpha.item()
    print(f"\n计算最终 PPL（alpha={final_alpha:.3f}）...")
    ppl_final = eval_ppl(model, eval_enc)
    print(f"  初始 PPL: {ppl_init:.4f}")
    print(f"  最终 PPL: {ppl_final:.4f}  (Δ = {ppl_final - ppl_init:+.4f})")

    # ── 激活稀疏率统计 ────────────────────────────────────────────────────
    print("\n测量最终激活稀疏率...")
    _monitor = ActivationSparsityMonitor()
    _monitor.attach(model)
    _ = eval_ppl(model, eval_enc)          # 跑一遍 eval 触发 hook
    final_sparsity = _monitor.mean_sparsity()
    _monitor.detach()
    print(f"  down_proj 输入平均零值比例: {final_sparsity:.2%}")

    # ── 保存 ──────────────────────────────────────────────────────────────
    if args.output_dir:
        print(f"\n保存模型到 {args.output_dir} ...")
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            model.save_pretrained(args.output_dir, safe_serialization=False)
            tokenizer.save_pretrained(args.output_dir)
            print("保存成功。")
        except Exception as e:
            print(f"[警告] 保存失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 参数解析 & main
# ─────────────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Llama 3 SwiGLU → ReLU 渐进式替换微调"
    )
    # 路径
    p.add_argument("--model_id",      type=str, default="/data/sza/model/Meta-Llama-3.1-8B")
    p.add_argument("--dataset_path",  type=str, default="/data/sza/local_dataset")
    p.add_argument("--output_dir",    type=str, default="/data/sza/model/Meta-Llama-3.1-8B-ReluSparse")
    p.add_argument("--device",        type=str, default="cuda")

    # 数据集
    p.add_argument("--dataset_name",     type=str, default="fineweb-edu",
                   choices=["fineweb-edu", "wikitext-103", "wikitext-2"],
                   help="训练数据集：fineweb-edu(推荐) / wikitext-103 / wikitext-2")
    p.add_argument("--max_train_tokens", type=int, default=1_000_000_000,
                   help="最多加载的训练 token 数（50M ≈ 97,000 块×512）")

    # KL 蒸馏
    p.add_argument("--teacher_model_id", type=str, default="/data/sza/model/Meta-Llama-3.1-8B",
                   help="教师模型路径（原始 SwiGLU），为 None 时禁用蒸馏")
    p.add_argument("--distill_lambda",   type=float, default=0.5,
                   help="KL 蒸馏损失权重：L=(1-λ)·CE + λ·KL，推荐 0.3~0.7")
    p.add_argument("--distill_temp",     type=float, default=2.0,
                   help="蒸馏温度 τ，>1 软化分布，推荐 2.0~4.0")

    # 训练超参
    p.add_argument("--max_steps",     type=int,   default=30000,
                   help="总训练步数")
    p.add_argument("--ramp_steps",    type=int,   default=15000,
                   help="alpha 从 0→1 的步数，之后保持 alpha=1")
    p.add_argument("--warmup_steps",  type=int,   default=3000,
                   help="学习率 warmup 步数")
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--batch_size",    type=int,   default=2,
                   help="批次大小（蒸馏时建议≤2以避免OOM）")
    p.add_argument("--gradient_accumulation_steps", type=int, default=16,
                   help="梯度累积步数，有效batch size = batch_size × gradient_accumulation_steps")
    p.add_argument("--seq_len",       type=int,   default=2048,
                   help="训练时的序列长度（蒸馏时建议≤1024）")

    # 稀疏正则化
    p.add_argument("--sparse_reg",    type=float, default=0.001,
                   help="2:4 激活稀疏正则化系数（0=禁用）")

    # Calibration
    p.add_argument("--skip_calibration", action="store_true",
                   help="跳过校准，使用默认 relu_scale=0.5（调试用）")

    # 训练范围
    p.add_argument("--train_all_params", action="store_true",
                   help="全参数微调（默认只训练 MLP 层）")

    # 日志 & 评估
    p.add_argument("--log_interval",  type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    # p.add_argument("--monitor_sparsity", action="store_true", default=True,
    #                help="在训练时监控激活稀疏度")
    p.add_argument("--monitor_sparsity", action=argparse.BooleanOptionalAction,
               default=True, help="监控激活稀疏度（--no-monitor_sparsity 可关闭）")
    return p.parse_args()


def main() -> None:
    args = build_args()

    if torch.cuda.is_available():
        print(f"检测到 {torch.cuda.device_count()} 个可用GPU:")
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
        print(f"环境变量 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    else:
        print("警告：CUDA 不可用，将在 CPU 运行（极慢）")

    print(f"\n加载模型: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 预先加载训练数据（用于calibration和训练）───────────────────────────
    print("\n加载训练数据（用于calibration和训练）...")
    train_ids = load_train_data(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_train_tokens=args.max_train_tokens,
        dataset_path=args.dataset_path
    )
    print(f"  已加载 {len(train_ids)} 块训练数据")

    # 学生模型放在 cuda:0（物理卡1）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print(f"\n学生模型已加载到 cuda:0 ({torch.cuda.get_device_name(0)})")

    # ── 替换所有 MLP（含方差对齐校准）─────────────────────────────────────
    if args.skip_calibration:
        print("  [跳过 Calibration] 使用默认 relu_scale=0.5")
        replaced = replace_all_mlp(model, alpha=0.0, calib_ids=None)
    else:
        print("  Calibration: 从训练数据中随机抽取样本...")
        # 随机抽取64条真实训练数据进行calibration
        calib_indices = torch.randint(0, len(train_ids), (64,))
        _calib_ids = train_ids[calib_indices].to("cuda:0")
        print(f"    抽取了 {len(_calib_ids)} 条真实训练样本 (indices: {calib_indices.tolist()})")

        replaced = replace_all_mlp(model, alpha=0.0, calib_ids=_calib_ids)
        del _calib_ids  # 在这里删除，确保变量存在时才删除

    print(f"  已替换 {replaced} 个 MLP 层为 LlamaMLPReluTransition（初始 alpha=0，纯 SwiGLU）")

    # ── 关键验证：alpha=0 时模型输出应与原始模型一致 ─────────────────────────
    print("\n[验证] 检查 alpha=0 时的模型输出...")
    model.eval()
    with torch.no_grad():
        test_ids = train_ids[:2].to("cuda:0")
        outputs = model(test_ids, labels=test_ids)
        initial_loss = outputs.loss.item()

    print(f"  初始 Loss (alpha=0): {initial_loss:.4f}")
    if initial_loss > 5.0:
        print("  ⚠️ 警告：初始 Loss 异常高！模型可能存在问题。")
        print("  正常的初始 Loss 应该在 2.0-4.0 之间。")
    else:
        print("  ✅ 初始 Loss 正常，模型替换成功！")
    model.train()

    # ── 加载教师模型（可选，用于 KL 蒸馏）──────────────────────────────────
    teacher_model = None
    if args.teacher_model_id is not None:
        print(f"\n加载全精度教师模型 (bfloat16): {args.teacher_model_id}")
        # CUDA_VISIBLE_DEVICES="1,3" 后，cuda:0=物理卡1, cuda:1=物理卡3
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:1"  # 映射到物理卡3（第二张可见卡）
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"  教师模型加载到cuda:1 ({torch.cuda.get_device_name(1)}), 蒸馏权重 λ={args.distill_lambda}, τ={args.distill_temp}")

    # ── 开始训练 ────────────────────────────────────────────────────────────
    train(args, model, tokenizer, teacher_model=teacher_model, train_ids=train_ids)


if __name__ == "__main__":
    main()
