#!/bin/bash

# Display which GPUs are visible before running
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
python - <<'PY'
import torch
print("torch visible device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"torch device {i}: {torch.cuda.get_device_name(i)}")
print("default device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
PY