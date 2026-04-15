#!/bin/bash
pkill -9 -f change_to_relu_ddp2.py
# pkill -9 -f accelerate
# unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
# --multi_gpu \
echo "=== 开始微调 ==="
# CUDA_VISIBLE_DEVICES="0,1,3,4" accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --mixed_precision=bf16 \
#     --main_process_port=29501 \
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    change_to_relu_ddp2.py \
    --max_train_tokens 4000000000 \
    --max_steps 60000 \
    --warmup_steps 50 \
    --ramp_steps 35000 \
    --eval_interval 200 \
    --lr 5e-6 \
    --distill_lambda 0.7 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_interval 200 \
    --log_interval 200 \
    --sparse_reg 0 \
    --output_dir "/data/sza/model/Meta-Llama-3.1-8B-ReluSparse-Test" \
    --log_file "train_change_relu_model_only.log" \
    --resume_from_checkpoint "/data/sza/model/Meta-Llama-3.1-8B-ReluSparse-Test/checkpoint-14400" \
    --resume_model_only
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