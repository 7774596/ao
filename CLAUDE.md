# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Install

```bash
# Developer install with CUDA support (requires --no-build-isolation)
USE_CUDA=1 pip install -e . --no-build-isolation

# Developer install with XPU support
USE_XPU=1 pip install -e . --no-build-isolation

# Python-only mode (skip C++ extensions)
USE_CPP=0 pip install -e . --no-build-isolation

# Install dev dependencies
pip install -e .[dev]
```

## Testing

```bash
# Run all tests
pytest test/

# Run specific test file
pytest test/quantization/test_quant_api.py

# Run float8 tests
pytest test/float8/test_base.py

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 pytest test/test_ops.py
```

## Linting

```bash
pip install ruff==0.11.6
ruff check --fix
ruff format .
```

## Architecture Overview

TorchAO is a PyTorch-native library for model optimization (quantization, sparsity, low-precision training). Key design principles: composability with `torch.compile`, tensor subclass-based quantization, and native PyTorch integration.

### Core Directories

- **`torchao/quantization/`**: Main quantization APIs. Entry points: `quantize_()`, `autoquant()`. Configs like `Int4WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig`.
- **`torchao/float8/`**: Float8 training workflow. `convert_to_float8_training()` converts Linear modules to Float8Linear. Recipes: "tensorwise", "rowwise", "rowwise_with_gw_hp".
- **`torchao/dtypes/`**: Custom tensor subclasses. `AffineQuantizedTensor` (base), `NF4Tensor` (QLoRA), layouts in `uintx/` and `floatx/` subdirs.
- **`torchao/sparsity/`**: Sparsity techniques (2:4 semi-structured, block sparsity).
- **`torchao/prototype/`**: Experimental features (MXFP8, AWQ, HQQ, SpinQuant, MoE training).
- **`torchao/csrc/`**: C++/CUDA kernels. CUDA kernels use CUTLASS. CPU kernels for ARM in `csrc/cpu/`.
- **`torchao/kernel/`**: Kernel configuration and autotuning infrastructure.
- **`torchao/optim/`**: Quantized optimizers (AdamW8bit, AdamW4bit, AdamWFp8) and CPU offloading.

### Key Patterns

**Quantization Flow**:
1. Create a config (e.g., `Int4WeightOnlyConfig(group_size=32)`)
2. Apply via `quantize_(model, config)`
3. Compile with `torch.compile(model, mode='max-autotune')`

**Tensor Subclass Pattern**: Quantized weights are `AffineQuantizedTensor` instances that dispatch to specialized kernels via `__torch_dispatch__`. Layouts define packing formats (e.g., `TensorCoreTiledLayout` for int4 tinygemm kernels).

**Float8 Training**: `Float8Linear` wraps standard `nn.Linear`, quantizing activations/weights to float8 on-the-fly during forward/backward.

### Important Files

- `torchao/quantization/quant_api.py`: Main `quantize_()` API and config classes
- `torchao/quantization/quant_primitives.py`: Low-level quantize/dequantize ops
- `torchao/dtypes/affine_quantized_tensor.py`: Base quantized tensor subclass
- `torchao/float8/float8_linear.py`: Float8Linear implementation
- `torchao/ops.py`: Custom op registrations

## Hardware Compatibility

- **H100/B200 (SM90a+)**: Full support including float8 training, int4, MXFP8
- **A100 (SM80)**: int4/int8 quantization, limited float8 support
- **ARM CPU**: Specialized kernels for int4/int8 via `TORCHAO_BUILD_CPU_AARCH64=1`

## Environment Variables

- `USE_CUDA=1`: Build with CUDA support
- `USE_XPU=1`: Build with XPU support
- `USE_CPP=0`: Skip C++ extension builds
- `TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1`: Skip loading .so files (for version compatibility)
- `TORCHAO_BUILD_CPU_AARCH64=1`: Build ARM CPU kernels