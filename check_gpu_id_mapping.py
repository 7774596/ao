#!/usr/bin/env python3
"""
GPU ID 映射关系查看脚本

显示 nvidia-smi GPU ID 与 PyTorch CUDA 设备 ID 之间的映射关系。
当设置了 CUDA_VISIBLE_DEVICES 环境变量时，两者的编号可能不一致。

用法:
    python check_gpu_id_mapping.py
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4 python check_gpu_id_mapping.py
"""

import os
import subprocess


def get_nvidia_smi_info():
    """通过 nvidia-smi 获取 GPU 信息"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "nvidia_smi_id": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "pci_bus_id": parts[3]
                })
        return gpus
    except Exception as e:
        print(f"无法获取 nvidia-smi 信息: {e}")
        return []


def get_cuda_info():
    """通过 PyTorch 获取 CUDA 设备信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("PyTorch CUDA 不可用")
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "cuda_id": i,
                "name": props.name,
                "uuid": torch.cuda.get_device_properties(i).uuid.hex(),
                "pci_bus_id": torch.cuda.get_device_properties(i).pci_bus_id
            })
        return gpus
    except ImportError:
        print("PyTorch 未安装")
        return []


def get_cuda_visible_devices():
    """获取 CUDA_VISIBLE_DEVICES 环境变量"""
    return os.environ.get("CUDA_VISIBLE_DEVICES", None)


def get_uuid_to_pci_mapping():
    """通过 nvidia-smi 获取 UUID 到 PCI Bus ID 的映射"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        mapping = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid = parts[0]
                pci = parts[1]
                mapping[uuid] = pci
        return mapping
    except Exception:
        return {}


def main():
    print("=" * 70)
    print("GPU ID 映射关系查看工具")
    print("=" * 70)

    # 环境变量
    cuda_visible = get_cuda_visible_devices()
    print(f"\n环境变量:")
    print(f"  CUDA_VISIBLE_DEVICES = {cuda_visible if cuda_visible else '未设置 (使用所有GPU)'}")

    # nvidia-smi 信息
    print(f"\n" + "-" * 70)
    print("nvidia-smi 显示的 GPU:")
    print("-" * 70)

    nvidia_gpus = get_nvidia_smi_info()
    if nvidia_gpus:
        for gpu in nvidia_gpus:
            print(f"  GPU {gpu['nvidia_smi_id']}: {gpu['name']}")
            print(f"           UUID: {gpu['uuid']}")
            print(f"           PCI Bus ID: {gpu['pci_bus_id']}")
    else:
        print("  未检测到 GPU")

    # PyTorch CUDA 信息
    print(f"\n" + "-" * 70)
    print("PyTorch/CUDA 可见的 GPU:")
    print("-" * 70)

    try:
        import torch

        if not torch.cuda.is_available():
            print("  CUDA 不可用")
            return

        print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")

        # 获取 nvidia-smi 的 UUID 映射
        uuid_to_pci = get_uuid_to_pci_mapping()

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # UUID 是 bytes 对象，需要转换
            uuid_bytes = props.uuid
            if isinstance(uuid_bytes, bytes):
                uuid_hex = uuid_bytes.hex()
            else:
                uuid_hex = str(uuid_bytes).replace("-", "").replace("GPU-", "")
            # GPU UUID 格式: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
            full_uuid = f"GPU-{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:]}"

            # 查找对应的 nvidia-smi ID
            nvidia_id = "?"
            for gpu in nvidia_gpus:
                if gpu['uuid'] == full_uuid:
                    nvidia_id = gpu['nvidia_smi_id']
                    break

            print(f"\n  cuda:{i} (PyTorch 设备ID)")
            print(f"    ├─ 名称: {props.name}")
            print(f"    ├─ UUID: {full_uuid}")
            print(f"    ├─ PCI Bus ID: {props.pci_bus_id}")
            print(f"    └─ 对应 nvidia-smi GPU ID: {nvidia_id}")

        # 映射总结
        print(f"\n" + "=" * 70)
        print("映射关系总结:")
        print("=" * 70)
        print(f"\n  {'PyTorch cuda:X':<18} {'nvidia-smi GPU ID':<20} {'PCI Bus ID'}")
        print(f"  {'-'*18} {'-'*20} {'-'*15}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            uuid_bytes = props.uuid
            if isinstance(uuid_bytes, bytes):
                uuid_hex = uuid_bytes.hex()
            else:
                uuid_hex = str(uuid_bytes).replace("-", "").replace("GPU-", "")
            full_uuid = f"GPU-{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-{uuid_hex[16:20]}-{uuid_hex[20:]}"

            nvidia_id = "?"
            for gpu in nvidia_gpus:
                if gpu['uuid'] == full_uuid:
                    nvidia_id = gpu['nvidia_smi_id']
                    break

            print(f"  cuda:{i:<13} GPU {nvidia_id:<17} {props.pci_bus_id}")

        # 实际使用示例
        print(f"\n" + "=" * 70)
        print("代码中的使用方式:")
        print("=" * 70)

        if cuda_visible:
            visible_ids = [int(x.strip()) for x in cuda_visible.split(",")]
            print(f"\n  当前 CUDA_VISIBLE_DEVICES={cuda_visible}")
            print(f"  这意味着物理卡 {visible_ids} 被映射为:")
            for i, phys_id in enumerate(visible_ids):
                print(f"    物理 GPU {phys_id} → cuda:{i}")
            print(f"\n  如果要让学生模型在物理卡{visible_ids[0]}上，教师模型在物理卡{visible_ids[1]}上:")
            print(f"    model_student = ... .to('cuda:0')  # 物理卡 {visible_ids[0]}")
            print(f"    model_teacher = ... .to('cuda:1')  # 物理卡 {visible_ids[1]}")
        else:
            print(f"\n  CUDA_VISIBLE_DEVICES 未设置，所有 GPU 可见")
            print(f"  nvidia-smi GPU ID 与 PyTorch cuda ID 一一对应")

    except ImportError:
        print("  PyTorch 未安装，无法获取 CUDA 信息")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()