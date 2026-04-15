import argparse
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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


class FP8IdentitySemiSparseActivationLinear(nn.Module):
    """
    Runtime activation 2:4 sparsity + FP8 activation/weight Linear (Llama3 MLP down_proj).

    Performance-oriented eval stack (see discussion in repo / five optimizations):
    (1) Fused CUDA row abs-max scale via ``rowwise_abs_max_fp32_scale`` when available.
    (2) Eliminating ``.t().contiguous()`` needs CUTLASS ``LayoutTagD`` / epilogue work in
        ``rowwise_scaled_linear_sparse_cutlass.cuh`` (not done here); call order must stay
        dense ``Xq`` first, sparse ``Wq`` third per ``check_inputs`` in that file.
    (3) Sparsify occupancy: ``MetadataCutlass8bits::kNumWarpsPerCTA`` in ``sparsify24.cu``.
    (4) GEMM tile heuristic: ``m <= 2048`` branch in ``select_config`` (same .cuh).
    (5) Eval loop: ``torch.full_like`` labels + ``torch.inference_mode`` in ``wiki2_eval``.

    ``torch.compile`` on the full model remains optional via ``--compile`` (custom ops
    stay opaque to Inductor across sparsify/GEMM boundaries).
    """

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
        # 避免在 GPU 上与 BF16 权重叠加峰值导致 OOM：CPU 上量化后再搬回 GPU。
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
        x_2d = x.view(-1, orig_shape[-1])  # [M, K]

        if (
            self.use_fused_row_absmax
            and x_2d.is_cuda
            and x_2d.dtype == torch.bfloat16
        ):
            try:
                x_scale_2d = rowwise_abs_max_fp32_scale(x_2d, self._fp8_inv_max)
            except (RuntimeError, NotImplementedError):
                x_scale_2d = (
                    x_2d.abs()
                    .amax(dim=1, keepdim=True)
                    .float()
                    .mul_(self._fp8_inv_max)
                )
        else:
            x_scale_2d = (
                x_2d.abs()
                .amax(dim=1, keepdim=True)
                .float()
                .mul_(self._fp8_inv_max)
            )

        xq_sparse, x_meta = torch.ops.torchao.sparse24_sm90_sparsify(
            x_2d,
            "cutlass",
            "identity",
            "largest_abs",
            dtype=self.activation_dtype,
            scale=x_scale_2d,
        )

        # C++ names: Xq = dense operand (full K), Wq = sparse (K/2); matches tests and
        # rowwise_scaled_linear_sparse_cutlass.cuh::check_inputs.
        out_2d = rowwise_scaled_linear_sparse_cutlass_f8f8(
            self.wq,
            self.w_scale,
            xq_sparse,
            x_meta,
            x_scale_2d.view(-1),
            bias=None,
            out_dtype=torch.bfloat16,
        ).t().contiguous()

        return out_2d.view(*orig_shape[:-1], out_2d.shape[-1])

    @classmethod
    def from_dense(
        cls,
        linear: nn.Linear,
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


def replace_llama_mlp_down_proj_with_activation_sparse(
    model: nn.Module,
    *,
    use_fused_row_absmax: bool = True,
) -> int:
    target_fqns = [
        fqn
        for fqn, module in model.named_modules()
        if isinstance(module, nn.Linear) and fqn.endswith("mlp.down_proj")
    ]

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
    return len(target_fqns)


def wiki2_eval(
    model,
    tokenizer,
    dataset_path,
    sequence_length=2048,
    stride=512,
    verbose=True,
    device="cuda",
):
    print(f"Loading Wikitext-2 dataset from {dataset_path}...")
    tokenizer.pad_token = tokenizer.eos_token

    try:
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, "keys") and "test" in dataset.keys():
            dataset = dataset["test"]
    except Exception:
        test_file = os.path.join(dataset_path, "wiki.test.raw")
        if os.path.exists(test_file):
            dataset = load_dataset("text", data_files={"test": test_file}, split="test")
        else:
            dataset = load_dataset(
                dataset_path,
                "wikitext-2-raw-v1",
                split="test",
                trust_remote_code=True,
            )

    # encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        max_length=131072,
        truncation=True,
    )
    encodings["input_ids"] = encodings["input_ids"].to(device)

    lls = []
    print("Computing Perplexity & Benchmarking...")

    torch.cuda.reset_peak_memory_stats()
    total_tokens_processed = 0
    start_time = time.perf_counter()

    for i in tqdm(range(0, encodings["input_ids"].size(1), stride)):
        begin_loc = max(i + stride - sequence_length, 0)
        end_loc = min(i + stride, encodings["input_ids"].size(1))
        trg_len = end_loc - i

        input_ids = encodings["input_ids"][:, begin_loc:end_loc]
        total_tokens_processed += input_ids.numel()

        # Avoid full clone: only copy the last trg_len labels (rest are ignore_index).
        target_ids = torch.full_like(input_ids, -100)
        target_ids[:, -trg_len:] = input_ids[:, -trg_len:]

        with torch.inference_mode():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

        if end_loc == encodings["input_ids"].size(1):
            break

    end_time = time.perf_counter()
    duration = end_time - start_time
    tokens_per_sec = total_tokens_processed / duration
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024**3)

    ppl = float(torch.exp(torch.stack(lls).sum() / end_loc))

    if verbose:
        print(
            f"\n[Result] PPL: {ppl:.4f} | Speed: {tokens_per_sec:.2f} tokens/s | Peak Mem: {peak_mem_gb:.2f} GB"
        )

    return ppl, tokens_per_sec, peak_mem_gb


def build_args():
    parser = argparse.ArgumentParser(
        description="Llama3 inference: FP8 W+A + activation 2:4 sparsity"
    )
    parser.add_argument("--model_id", type=str, default="/data/sza/model/Meta-Llama-3.1-8B")
    parser.add_argument("--dataset_path", type=str, default="/data/sza/local_dataset")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/sza/model/Meta-Llama-3.1-8B-FP8-ACT24",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for inference",
    )
    parser.add_argument(
        "--no-fused-row-absmax",
        action="store_true",
        help="Disable CUDA fused row abs-max scale (use PyTorch abs+amax instead)",
    )
    return parser.parse_args()


def main():
    args = build_args()
    model_id = args.model_id
    dataset_path = args.dataset_path
    save_path = args.save_path
    device = args.device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Running on GPU: {gpu_name}")
    else:
        print("Warning: CUDA not available, running on CPU (will be slow)")
        gpu_name = "CPU"

    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("\n" + "=" * 50)
    print("Test 1: Baseline (BF16)")
    print("=" * 50)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
    )

    ppl_base, speed_base, mem_base = wiki2_eval(
        model,
        tokenizer,
        dataset_path,
        verbose=False,
        device=device,
    )
    print(
        f"Baseline -> PPL: {ppl_base:.4f}, Speed: {speed_base:.2f} tok/s, Mem: {mem_base:.2f} GB"
    )

    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("Test 2: Quantized (FP8 W8A8) + Activation 2:4 Sparse")
    print("=" * 50)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
    )

    replaced = replace_llama_mlp_down_proj_with_activation_sparse(
        model,
        use_fused_row_absmax=not args.no_fused_row_absmax,
    )
    print(f"Replaced {replaced} MLP down_proj layers with activation 2:4 sparse FP8 kernels")

    print(
        "Applying Float8 Dynamic Activation + Float8 Weight Quantization to remaining Linear layers..."
    )
    # quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        filter_fn=lambda module, fqn: isinstance(module, nn.Linear),
    )

    print(f"Saving quantized model to {save_path} ...")
    try:
        model.save_pretrained(save_path, safe_serialization=False)
        tokenizer.save_pretrained(save_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"[Warning] Save failed (expected for custom tensors): {e}")
        print("Continuing with benchmark...")

    if args.compile:
        print("Compiling model (Mode: max-autotune)...")
        print("Note: First inference steps will be slow due to JIT compilation.")
        model = torch.compile(model, mode="max-autotune")

    ppl_fp8, speed_fp8, mem_fp8 = wiki2_eval(
        model,
        tokenizer,
        dataset_path,
        verbose=False,
        device=device,
    )

    print("\n" + "=" * 50)
    print(f"Summary Result (GPU: {gpu_name})")
    print("=" * 50)
    print(f"{'Metric':<15} | {'Baseline (BF16)':<18} | {'FP8+Act2:4':<18} | {'Diff':<10}")
    print("-" * 65)
    print(f"{'Perplexity':<15} | {ppl_base:<18.4f} | {ppl_fp8:<18.4f} | {ppl_fp8 - ppl_base:+.4f}")
    print(
        f"{'Speed (tok/s)':<15} | {speed_base:<18.2f} | {speed_fp8:<18.2f} | {(speed_fp8 / speed_base - 1) * 100:+.2f}%"
    )
    print(f"{'Peak Mem (GB)':<15} | {mem_base:<18.2f} | {mem_fp8:<18.2f} | {mem_fp8 - mem_base:+.2f}")
    print("=" * 50)
    if args.compile:
        print("* Speed Note: FP8 speed includes JIT compilation overhead for the first few steps.")


if __name__ == "__main__":
    main()
