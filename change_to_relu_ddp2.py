from __future__ import annotations
import math
import argparse
import os
import time
import sys
from typing import Dict, List, Optional, Tuple

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
import json
import glob
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
        gate = self.gate_proj(x)
        up   = self.up_proj(x)

        alpha = self.alpha.to(gate.dtype)
        relu_scale = self.relu_scale.to(gate.dtype)

        swiglu = F.silu(gate)
        relu_act = F.relu(gate) * relu_scale
        mixed = alpha * relu_act + (1.0 - alpha) * swiglu

        h = mixed * up

        if self.training:
            with torch.no_grad():
                self._sparsity_frac = (h.abs() < 1e-6).float().mean().item()
            # 仅在训练循环需要稀疏惩罚时应用挂载，且避免 checkpointing 内存泄漏
            coeff = getattr(self, '_penalty_coeff', 0.0)
            if coeff > 0.0:
                h = Sparse24PenaltyFunction.apply(h, coeff)
            self.last_h = None  # 防止之前逻辑可能意外调用的情况，安全起见置为 None
        else:
            self.last_h = None

        out = self.down_proj(h)
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

    print(f"  [校准] 准备对 {len(mlps)} 个 MLP 层进行校准 (低显存模式)...")

    # [应用建议3] 降低校准输入大小
    # 假设 calib_ids shape 类似于 [N, seq_len]
    calib_bs = min(2, len(calib_ids))
    ids = calib_ids[:calib_bs]
    if ids.ndim == 2 and ids.shape[1] > 512:
        ids = ids[:, :512]

    layer_scales = {}
    layer_biases = {}

    model.eval()

    # [应用建议2] 逐层校准，一次只允许 1 层的前向驻留
    with torch.inference_mode():  # [应用建议4] 使用 inference_mode
        for i, mod in enumerate(mlps):
            layer_scale_res = []
            layer_bias_res = []

            def hook_fn(module, inp, out):
                x = inp[0]
                
                # [应用建议1] 移除 .float() 内存膨胀，保持在原有 dtype (bf16) 下前向
                gate = module.gate_proj(x)
                
                # 1) 计算 per-channel scale (仅在归约统计时转 float32 保证精度)
                var_silu = torch.nn.functional.silu(gate).to(torch.float32).var(dim=(0,1))
                var_relu = torch.nn.functional.relu(gate).to(torch.float32).var(dim=(0,1))
                
                scale = torch.sqrt(var_silu / (var_relu + 1e-8))
                scale = torch.clamp(scale, min=0.1, max=5.0)
                scale = scale.to(dtype=module.down_proj.weight.dtype)
                layer_scale_res.append(scale)
                
                # 2) 计算 per-channel bias
                up = module.up_proj(x)
                h_silu = torch.nn.functional.silu(gate) * up
                h_relu = torch.nn.functional.relu(gate) * scale * up
                
                out_silu = module.down_proj(h_silu)
                out_relu = module.down_proj(h_relu)
                
                diff = (out_silu.to(torch.float32) - out_relu.to(torch.float32)).mean(dim=(0,1))
                layer_bias_res.append(diff.to(module.down_proj.weight.dtype))

            # 每次只挂 1 个层的 hook
            handle = mod.register_forward_hook(hook_fn)
            
            # 使用缩小后的输入跑一次完整前向，只提取这一层的统计
            _ = model(ids)
            
            handle.remove()
            
            # 记录本层结果并保存在目标设备
            layer_scales[i] = layer_scale_res[0] if layer_scale_res else torch.tensor(1.0, dtype=mod.down_proj.weight.dtype, device=mod.down_proj.weight.device)
            layer_biases[i] = layer_bias_res[0] if layer_bias_res else torch.zeros((mod.down_proj.weight.shape[0],), dtype=mod.down_proj.weight.dtype, device=mod.down_proj.weight.device)
            
            # 强制清理当前层缓冲，确保下一层开始前显存已释放
            torch.cuda.empty_cache()

    model.train()
    print(f"  [校准] 成功计算 {len(layer_scales)} 层的独有 relu_scale 和 down_proj_bias。")
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


class Sparse24PenaltyFunction(torch.autograd.Function):
    """
    前向恒等返回 h；反向时在 grad_output 上叠加 coeff * ∂(sparse24_activation_penalty)/∂h，
    等价于总损失里增加 coeff * penalty(h)，但不把 penalty 标量并入 loss 张量（与训练循环注释一致）。
    """

    @staticmethod
    def forward(ctx, h: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = float(coeff)
        ctx.save_for_backward(h)
        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h, = ctx.saved_tensors
        coeff = ctx.coeff
        # autograd.Function.backward 默认在 no_grad 下执行，需显式 enable 才能对 penalty 再求导
        with torch.enable_grad():
            h_det = h.detach().requires_grad_(True)
            pen = sparse24_activation_penalty(h_det)
            grad_pen, = torch.autograd.grad(
                pen,
                h_det,
                retain_graph=False,
                create_graph=False,
            )
        return grad_output + coeff * grad_pen, None


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
        if hasattr(module, '_sparsity_frac'):
            self.sparsity_records.append(module._sparsity_frac)

    def mean_sparsity(self) -> float:
        if not self.sparsity_records:
            return 0.0
        val = sum(self.sparsity_records) / len(self.sparsity_records)
        self.sparsity_records.clear()
        return val

import itertools
from datasets import load_dataset, concatenate_datasets, load_from_disk
import glob as glob_module

def load_train_data(pt_file: str, max_train_tokens: int, seq_len: int) -> torch.Tensor:
    """
    离线数据加载：直接使用 prepare_data.py 处理好的 .pt 张量文件
    """
    print(f"\n[数据加载] 正在从本地张量文件 {pt_file} 加载数据...")

    ids_tensor = torch.load(pt_file, weights_only=True, mmap=True)
    N = len(ids_tensor)
    print(f"[数据加载] 成功加载 {N} 块 × {seq_len} tokens = {N * seq_len / 1e6:.1f}M tokens")

    max_blocks = max_train_tokens // seq_len
    if N > max_blocks:
        ids_tensor = ids_tensor[:max_blocks]
        print(f"  按 --max_train_tokens 截断到 {max_blocks} 块 ({max_blocks * seq_len / 1e6:.1f}M tokens)")

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


def chunked_cross_entropy(logits, targets, chunk_size=256):
    B, T, V = logits.shape
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    
    total_loss = torch.tensor(0.0, device=logits.device)
    for i in range(0, T - 1, chunk_size):
        chunk_logits = shift_logits[:, i:i+chunk_size, :].float()
        chunk_labels = shift_labels[:, i:i+chunk_size]
        
        loss_chunk = F.cross_entropy(
            chunk_logits.reshape(-1, V),
            chunk_labels.reshape(-1),
            reduction='sum'
        )
        total_loss += loss_chunk
        
    return total_loss / (B * (T - 1))


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
# 8. 训练主循环 (DDP重写版)
# ─────────────────────────────────────────────────────────────────────────────

def infinite_dataloader(dataloader):
    epoch = 0
    while True:
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1

def train(args, model: nn.Module, tokenizer, teacher_model: Optional[nn.Module] = None,
          train_ids: Optional[torch.Tensor] = None, accelerator=None) -> None:
    
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"\n[训练配置] 学生模型设备: {device}")
        if teacher_model is not None:
            print(f"[训练配置] 教师模型设备: {teacher_model.device}")
        print(f"[训练配置] 数据将在训练时从CPU传输到 {device}\n")

    # ── 冻结非 MLP 参数（可选）──────────────────────────────────────────────
    if not args.train_all_params:
        for name, param in model.named_parameters():
            if ".mlp." not in name:
                param.requires_grad_(False)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        if accelerator.is_main_process:
            print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # ── 数据加载与Dataloader (多卡数据分发) ──────────────────────────────────
    if train_ids is None:
        if accelerator.is_main_process:
            print("加载训练集...")
        with accelerator.main_process_first():
            train_ids = load_train_data(
                pt_file=args.train_data_pt,
                max_train_tokens=args.max_train_tokens,
                seq_len=args.seq_len
            )
    
    dataset = TensorDataset(train_ids)
    # DDP 环境下，accelerator.prepare(dataloader) 会自动插入 DistributedSampler
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    eval_enc = None
    if accelerator.is_main_process:
        print("加载验证集...")
        eval_enc = load_eval_data(args.dataset_path, tokenizer, device=device)

    # ── 优化器 & 调度 ────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr * accelerator.num_processes, weight_decay=0.01, betas=(0.9, 0.95), min_8bit_size=16384
        )
        
        if accelerator.is_main_process:
            print(f"  使用 8-bit AdamW 优化器")
    except ImportError:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr * accelerator.num_processes, weight_decay=0.01, betas=(0.9, 0.95)
        )
        if accelerator.is_main_process:
            print(f"  [警告] 未安装 bitsandbytes，使用标准 AdamW")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    alpha_scheduler = AlphaScheduler(model, ramp_steps=args.ramp_steps)

    # ── 使用 Accelerate 包装模型、优化器、调度器、数据 ───────────────────────
    # 注意：Teacher由于被冻结不参与反向传播，不需要包装，只需保持在 device 即可。
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    data_iterator = infinite_dataloader(dataloader)

    # ── 断点接续 (Resume) 的恢复逻辑 ───────────────────────────────────────
    step = 0
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            if accelerator.is_main_process:
                print(f"\n正在从 {args.resume_from_checkpoint} 恢复训练...")
            # 兼容旧 checkpoint：补齐缺失的非 rank-0 random_states 文件
            rank0_rng = os.path.join(args.resume_from_checkpoint, "random_states_0.pkl")
            if os.path.exists(rank0_rng):
                import shutil
                for ri in range(accelerator.num_processes):
                    target_rng = os.path.join(
                        args.resume_from_checkpoint, f"random_states_{ri}.pkl")
                    if not os.path.exists(target_rng):
                        shutil.copy2(rank0_rng, target_rng)
                        if accelerator.is_main_process:
                            print(f"  [兼容] 复制 random_states_0 -> random_states_{ri}")
            accelerator.load_state(args.resume_from_checkpoint)
            state_file = os.path.join(args.resume_from_checkpoint, "custom_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                    step = state.get("step", 0)

            alpha_scheduler.step(step)

            if accelerator.is_main_process:
                print(f"成功恢复到第 {step} 步，开始接续训练。")
        else:
            if accelerator.is_main_process:
                print(f"[警告] 传入的 checkpoint 路径 {args.resume_from_checkpoint} 不存在，从头开始训练。")

    # ── 稀疏度监控 ───────────────────────────────────────────────────────────
    # 在DDP模式下，monitor挂钩真实模型: accelerator.unwrap_model(model)
    sparsity_monitor = ActivationSparsityMonitor()
    if args.monitor_sparsity:
        sparsity_monitor.attach(accelerator.unwrap_model(model))

    # ── 初始/恢复 PPL ─────────────────────────────────────────────────────
    if accelerator.is_main_process:
        if step == 0:
            print("\n计算初始 PPL（alpha=0，纯 SwiGLU）...")
            ppl_init = eval_ppl(accelerator.unwrap_model(model), eval_enc)
            print(f"  初始 PPL: {ppl_init:.4f}")
        else:
            current_alpha = get_all_relu_mlps(accelerator.unwrap_model(model))[0].alpha.item()
            print(f"\n计算恢复点 PPL（step={step}, alpha={current_alpha:.3f}）...")
            ppl_resume = eval_ppl(accelerator.unwrap_model(model), eval_enc)
            print(f"  恢复点 PPL: {ppl_resume:.4f}")

    # ── 训练循环 ────────────────────────────────────────────────────────────
    model.train()
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    
    if accelerator.is_main_process:
        print(f"\n[分布式多卡 & 梯度累积配置]")
        print(f"  参与设备数: {accelerator.num_processes}")
        print(f"  单卡 batch_size: {args.batch_size}")
        print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
        print(f"  全局有效 batch_size: {effective_batch_size}")
        print(f"  开始微调：max_steps={args.max_steps}, ramp_steps={args.ramp_steps}, sparse_reg={args.sparse_reg}, lr={args.lr * accelerator.num_processes}\n")

    pbar = tqdm(total=args.max_steps, desc="ReLU-ification 微调 (DDP)", disable=not accelerator.is_main_process, initial=step)
    alpha = alpha_scheduler.step(step)

    def _set_penalty_coeff(coeff):
        for mod in accelerator.unwrap_model(model).modules():
            if isinstance(mod, LlamaMLPReluTransition):
                mod._penalty_coeff = coeff
                mod.last_h = None  # clean up legacy just in case

    _set_penalty_coeff(args.sparse_reg if (args.sparse_reg > 0 and alpha > 0.1) else 0.0)
    log_loss, log_ce, log_kl, log_h_sparse = 0.0, 0.0, 0.0, 0.0
    _micro_loss, _micro_ce, _micro_kl = 0.0, 0.0, 0.0

    while step < args.max_steps:
        # Accelerate handles gradient accumulation context automatically
        with accelerator.accumulate(model):
            batch = next(data_iterator)
            # Batch from Dataloader is a list of tensors: [input_ids]
            input_ids = batch[0]
            target_ids = input_ids.clone()
            
            # Forward pass (无 labels，避免 transformers 内部计算爆显存)
            outputs = model(input_ids)
            ce_loss_val = chunked_cross_entropy(outputs.logits, target_ids)
            lm_loss = ce_loss_val
            
            # 手动释放不必要的 graph 对象
            if hasattr(outputs, "hidden_states"):
                del outputs.hidden_states

            # Optional KL Distillation with Teacher
            kl_loss_val = torch.tensor(0.0, device=device)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids)
                s_logits = outputs.logits
                t_logits = teacher_outputs.logits
                kl_val = kl_distill_loss(s_logits, t_logits, temperature=args.distill_temp)
                lm_loss = (1.0 - args.distill_lambda) * ce_loss_val + args.distill_lambda * kl_val
                kl_loss_val = kl_val
                # 尽早释放大张量
                del teacher_outputs, s_logits, t_logits

            loss = lm_loss
            # 注：稀疏惩罚已经在模型前向传播时通过 AddSparsePenaltyWrap 自动处理并融合到了梯度中
            # 此处不再需要手动获取 layer 的 last_h 以及计算外层 graph 惩罚

            accelerator.backward(loss)

            # Gradient clipping (should be synced)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            _micro_loss += loss.detach().float().item()
            _micro_ce += ce_loss_val.detach().float().item()
            _micro_kl += kl_loss_val.detach().float().item()

        if accelerator.sync_gradients:
            step += 1
            pbar.update(1)

            alpha = alpha_scheduler.step(step)
            _set_penalty_coeff(args.sparse_reg if (args.sparse_reg > 0 and alpha > 0.1) else 0.0)

            n_micro = args.gradient_accumulation_steps
            log_loss += _micro_loss / n_micro
            log_ce += _micro_ce / n_micro
            log_kl += _micro_kl / n_micro
            _micro_loss, _micro_ce, _micro_kl = 0.0, 0.0, 0.0
            if args.monitor_sparsity:
                log_h_sparse += sparsity_monitor.mean_sparsity()

            if step % args.log_interval == 0:
                metrics = torch.tensor([log_loss, log_ce, log_kl], device=device)
                metrics = accelerator.reduce(metrics, reduction="mean") / args.log_interval

                avg_loss, avg_ce, avg_kl = metrics[0].item(), metrics[1].item(), metrics[2].item()
                sparsity = (log_h_sparse / args.log_interval) if args.monitor_sparsity else 0.0

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "ce":   f"{avg_ce:.4f}",
                    "kl":   f"{avg_kl:.4f}",
                    "α":    f"{alpha:.3f}",
                    "zeros": f"{sparsity:.2%}",
                })

                log_loss, log_ce, log_kl, log_h_sparse = 0.0, 0.0, 0.0, 0.0

            if step % args.save_interval == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                # save_state 必须所有进程共同调用，各 rank 保存各自的 random_states
                accelerator.save_state(ckpt_dir)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"\n[step {step:5d}] 保存 checkpoint 到 {ckpt_dir} ...")
                    try:
                        with open(os.path.join(ckpt_dir, "custom_state.json"), "w") as f:
                            json.dump({"step": step, "alpha": float(alpha)}, f)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            os.path.join(ckpt_dir, "hf_model"), safe_serialization=False)
                        tokenizer.save_pretrained(os.path.join(ckpt_dir, "hf_model"))
                        print("  Checkpoint 保存成功。")
                    except Exception as e:
                        print(f"  [警告] Checkpoint 保存失败: {e}")

            # ── Eval Evaluation ────────────────────────────────────────
            if getattr(args, "eval_interval", 500) > 0 and step % args.eval_interval == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    ppl = eval_ppl(unwrapped_model, eval_enc)
                    effective_lambda = args.distill_lambda if teacher_model else 0.0
                    print(f"\n[step {step:5d}] PPL={ppl:.4f}  α={alpha:.3f}  "
                          f"λ_distill={effective_lambda:.3f}  lr={lr_scheduler.get_last_lr()[0]:.2e}")
                
                    model.train() # Reset to train mode in main process
                accelerator.wait_for_everyone()

    pbar.close()

    # 关闭监控
    if args.monitor_sparsity:
        sparsity_monitor.detach()

    # 最终评估
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_alpha = get_all_relu_mlps(unwrapped_model)[0].alpha.item()
        print(f"\n计算最终 PPL（alpha={final_alpha:.3f}）...")
        ppl_final = eval_ppl(unwrapped_model, eval_enc)
        print(f"  最终 PPL: {ppl_final:.4f}")

        if args.output_dir:
            print(f"\n保存最终模型到 {args.output_dir} ...")
            os.makedirs(args.output_dir, exist_ok=True)
            try:
                unwrapped_model.save_pretrained(args.output_dir, safe_serialization=False)
                tokenizer.save_pretrained(args.output_dir)
                print("保存成功。")
            except Exception as e:
                print(f"[警告] 保存失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 参数解析 & main
# ─────────────────────────────────────────────────────────────────────────────

def build_args():
    # 抽取出来复用，直接用正则中已经处理好的前面那部分
    pass

def main() -> None:
    # ── DDP/Accelerate 初始化 ──
    # 在最开始解析参数
    p = argparse.ArgumentParser(description="Llama 3 SwiGLU → ReLU 多卡并跑微调")
    p.add_argument("--model_id",      type=str, default="/data/sza/model/Meta-Llama-3.1-8B")
    p.add_argument("--dataset_path",  type=str, default="/data/sza/local_dataset")
    p.add_argument("--output_dir",    type=str, default="/data/sza/model/Meta-Llama-3.1-8B-ReluSparse")
    p.add_argument("--dataset_name",     type=str, default="fineweb-edu", choices=["fineweb-edu", "wikitext-103", "wikitext-2"])
    p.add_argument("--max_train_tokens", type=int, default=1_000_000_000)
    p.add_argument("--teacher_model_id", type=str, default="/data/sza/model/Meta-Llama-3.1-8B")
    p.add_argument("--distill_lambda",   type=float, default=0.5)
    p.add_argument("--distill_temp",     type=float, default=2.0)
    p.add_argument("--max_steps",     type=int,   default=30000)
    p.add_argument("--ramp_steps",    type=int,   default=15000)
    p.add_argument("--warmup_steps",  type=int,   default=3000)
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="单卡的累积步数")
    p.add_argument("--seq_len",       type=int,   default=2048)
    p.add_argument("--sparse_reg",    type=float, default=0.001)
    p.add_argument("--skip_calibration", action="store_true")
    p.add_argument("--train_all_params", action="store_true")
    p.add_argument("--log_interval",  type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--log_file", type=str, default="train_change_relu.log")
    p.add_argument("--monitor_sparsity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="恢复训练的checkpoint目录")
    p.add_argument("--train_data_pt", type=str, default="/data/sza/local_dataset/train_data_4B.pt", help="预处理好的数据张量文件")
    
    args = p.parse_args()

    from accelerate import InitProcessGroupKwargs
    from datetime import timedelta
    # 将 DDP 同步超时时间延长到 3 小时 (10800 秒)，防止主卡加载大量数据时副卡等崩溃
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[timeout_kwargs]
    )

    if accelerator.is_main_process:
        sys.stdout = Logger(args.log_file)
        sys.stderr = sys.stdout
        print(f"\n================ 启动 DDP 训练 (共 {accelerator.num_processes} 卡) ================\n")
    else:
        sys.stdout = open(os.devnull, "w")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print("\n加载训练数据（用于calibration和训练）...")
    
    with accelerator.main_process_first():
        train_ids = load_train_data(
            pt_file=args.train_data_pt,
            max_train_tokens=args.max_train_tokens,
            seq_len=args.seq_len
        )
        
    if accelerator.is_main_process:
        print(f"  已加载 {len(train_ids)} 块训练数据")

    # ========= 分配模型到各卡的正确设备设备上 =========
    # DDP 环境中，每张卡处理一个进程。我们使用 accelerator.local_process_index
    device_map = {"": accelerator.local_process_index}

    # 保证所有的卡等主进程读完数据
    accelerator.wait_for_everyone()

    # 判断是否恢复训练（提前判断，跳过不必要的步骤）
    is_resuming = args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map, use_cache=False,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if accelerator.is_main_process:
        print(f"\n学生模型已加载。已开GC")

    # ── 替换所有 MLP（含方差对齐校准）─────────────────────────────────────
    # 恢复训练时跳过校准，因为 checkpoint 会覆盖 relu_scale 和 down_proj_bias
    if is_resuming:
        if accelerator.is_main_process:
            print("  [恢复训练] 跳过 Calibration，直接替换 MLP 结构...")
        replaced = replace_all_mlp(model, alpha=0.0, calib_ids=None)
    elif args.skip_calibration:
        if accelerator.is_main_process:
            print("  [跳过 Calibration] 使用默认 relu_scale=0.5")
        replaced = replace_all_mlp(model, alpha=0.0, calib_ids=None)
    else:
        if accelerator.is_main_process:
            print("  Calibration: 从训练数据中随机抽取样本...")
        # 取16个真实样本 (单卡算一下就好或者都在自己卡上算)
        calib_indices = torch.randint(0, len(train_ids), (16,))
        calib_ids = train_ids[calib_indices].to(accelerator.device)
        replaced = replace_all_mlp(model, alpha=0.0, calib_ids=calib_ids)

    if accelerator.is_main_process:
        print(f"  已替换 {replaced} 个 MLP 层为 LlamaMLPReluTransition")

    # ── 关键验证（恢复训练时跳过）──────────────────────────────────────
    if not is_resuming:
        if accelerator.is_main_process:
            print("\n[验证] 检查 alpha=0 时的模型输出...")
        model.eval()
        with torch.no_grad():
            test_ids = train_ids[:1].to(accelerator.device)
            out = model(test_ids, labels=test_ids)
            if accelerator.is_main_process:
                print(f"  初始 Loss (alpha=0): {out.loss.item():.4f}")

    # ── 加载教师模型 ───────────────────────────────────────────────────────
    teacher_model = None
    if args.teacher_model_id is not None:
        if accelerator.is_main_process:
            print(f"\n加载教师模型: {args.teacher_model_id}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map, use_cache=False,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    # ── 开始训练 ────────────────────────────────────────────────────────────
    train(args, model, tokenizer, teacher_model=teacher_model, train_ids=train_ids, accelerator=accelerator)

if __name__ == "__main__":
    main()
