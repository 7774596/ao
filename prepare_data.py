#!/usr/bin/env python3
# prepare_data.py
# 离线数据处理脚本，将其预处理为完整的 Pytorch Tensor 并保存。
# 这样在多卡训练时只需直接 torch.load 读取，极大缩短数据加载时间。

import os
import torch
import argparse
import glob as glob_module
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

def prep_train_data(tokenizer, seq_len: int, max_train_tokens: int, dataset_path: str, output_file: str):
    print(f"正在准备数据，最大 Token 数: {max_train_tokens / 1e6:.1f}M，序列长度: {seq_len}")
    
    # ── 1. 加载本地 FineWeb-Edu ──
    parquet_dir = os.path.join(dataset_path, "fineweb-edu", "sample-10BT", "sample", "10BT")
    parquet_files = sorted(glob_module.glob(os.path.join(parquet_dir, "*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"未找到 FineWeb-Edu parquet 文件: {parquet_dir}/*.parquet")

    # ── 2. 加载本地 UltraChat ──
    ultrachat_path = os.path.join(dataset_path, "ultrachat_200k")
    has_ultrachat = os.path.exists(ultrachat_path)

    fw_ratio = 0.7 if has_ultrachat else 1.0
    fw_tokens = int(max_train_tokens * fw_ratio)
    
    target_files = max(1, fw_tokens // 100_000_000)
    files_to_load = parquet_files[:min(target_files, len(parquet_files))]

    all_ids = []
    total_tokens = 0

    print("开始处理 FineWeb-Edu...")
    for i, pq_file in enumerate(files_to_load):
        if total_tokens >= fw_tokens:
            break
        print(f"  处理文件 [{i+1}/{len(files_to_load)}] {os.path.basename(pq_file)}...")
        
        ds = load_dataset("parquet", data_files={"train": pq_file}, split="train")

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=seq_len, add_special_tokens=False)

        tokenized = ds.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["text"])
        all_tokens = []
        for item in tokenized:
            all_tokens.extend(item["input_ids"] + [tokenizer.eos_token_id])
            if len(all_tokens) > (fw_tokens - total_tokens) + seq_len:
                break

        new_blocks = [
            all_tokens[i:i+seq_len]
            for i in range(0, len(all_tokens) - seq_len, seq_len)
        ]
        
        needed_blocks = (fw_tokens - total_tokens) // seq_len
        if needed_blocks > 0:
            new_blocks = new_blocks[:needed_blocks]
            
        all_ids.extend(new_blocks)
        total_tokens += len(new_blocks) * seq_len
        del ds, tokenized

    print(f"FineWeb-Edu 处理完成，收集了 {total_tokens/1e6:.1f}M tokens")

    if has_ultrachat:
        chat_tokens = max_train_tokens - total_tokens
        print(f"开始处理 UltraChat，目标: {chat_tokens/1e6:.1f}M tokens...")
        try:
            chat_ds = load_from_disk(ultrachat_path)
            if "train_sft" in chat_ds:
                chat_ds = chat_ds["train_sft"]

            if "text" not in chat_ds.column_names:
                if "messages" in chat_ds.column_names:
                    def format_llama3_chat(example):
                        text = "<|begin_of_text|>"
                        for msg in example["messages"]:
                            text += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                        return {"text": text}
                    chat_ds = chat_ds.map(format_llama3_chat, num_proc=4, remove_columns=chat_ds.column_names)

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
            print(f"UltraChat 处理完成: {chat_count*seq_len/1e6:.1f}M tokens")
        except Exception as e:
            print(f"[警告] UltraChat 加载失败: {e}")

    print("打乱数据随机性...")
    import random
    random.seed(42)
    random.shuffle(all_ids)

    print("转换为 PyTorch Tensor...")
    ids_tensor = torch.tensor(all_ids, dtype=torch.long)
    
    print(f"保存张量到 {output_file} ...")
    torch.save(ids_tensor, output_file)
    print("全部完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/data/sza/model/Meta-Llama-3.1-8B")
    parser.add_argument("--dataset_path", type=str, default="/data/sza/local_dataset")
    parser.add_argument("--max_train_tokens", type=int, default=4_000_000_000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--output_file", type=str, default="train_data_4B.pt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    prep_train_data(tokenizer, args.seq_len, args.max_train_tokens, args.dataset_path, args.output_file)
