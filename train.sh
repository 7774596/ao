#!/bin/bash

echo "=== 开始微调 ==="
CUDA_VISIBLE_DEVICES="0,1,3,4" accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --main_process_port=29501 \
    change_to_relu_ddp2.py \
    --max_train_tokens 4000000000 \
    --max_steps 60000 \
    --warmup_steps 1000 \
    --ramp_steps 35000 \
    --eval_interval 1000 \
    --lr 3e-5 \
    --distill_lambda 0.7 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_interval 200 \
    --log_interval 500 \
    --output_dir "/data/sza/model/Meta-Llama-3.1-8B-ReluSparse-Test" \
    --log_file "train_change_relu.log" \
    --resume_from_checkpoint "/data/sza/model/Meta-Llama-3.1-8B-ReluSparse-Test/checkpoint-7000"

if [ $? -ne 0 ]; then
    echo "验证失败！请检查错误。"
    exit 1
fi

echo "=== 验证通过！ ==="
exit 0

# 恢复训练
# accelerate launch --multi_gpu --num_processes=4 change_to_relu_ddp2.py \
#     --batch_size 1 --gradient_accumulation_steps 8 \
#     --resume_from_checkpoint "/data/sza/model/Meta-Llama-3.1-8B-ReluSparse-Test/checkpoint-5000"
# 暂停训练
# pkill -STOP -f change_to_relu_ddp2.py
# 恢复训练
# pkill -CONT -f change_to_relu_ddp2.py
# 强制停止
# pkill -9 -f change_to_relu_ddp2.py