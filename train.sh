#!/bin/bash

# echo "=== 开始小步数快速验证 ==="
python change_to_relu.py \
    --max_train_tokens 4000000000 \
    --max_steps 60000 \
    --warmup_steps 1000 \
    --ramp_steps 35000 \
    --eval_interval 1000 \
    --lr 3e-5 \
    --distill_lambda 0.7 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --log_interval 500 \
    --output_dir "/data/sza/model/Meta-Llama-3.1-8B-ReluSparse-Test"

if [ $? -ne 0 ]; then
    echo "验证失败！请检查错误。"
    exit 1
fi

echo "=== 验证通过！ ==="
exit 0


