#!/bin/bash
# test_resume_smoke.sh
# 用于验证断点续训 (resume) 逻辑

export CUDA_VISIBLE_DEVICES=0,1  # 仅使用前两张卡做快速测试
OUTPUT_DIR="/data/sza/model/test-resume-output"

echo "========== 阶段 1：首轮训练 (运行到 step 4，每 2 步保存) =========="
accelerate launch --multi_gpu --num_processes 2 change_to_relu_ddp.py \
    --max_train_tokens 2000000 \
    --max_steps 4 \
    --save_interval 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --skip_calibration \
    --output_dir $OUTPUT_DIR \
    --log_file "test_resume_stage1.log"

echo ""
echo "========== 阶段 2：断点续训 (从 checkpoint-2 恢复) =========="
# 这里继续设置 max_steps 4，预期代码只会运行 step 3 和 step 4 就停止
accelerate launch --multi_gpu --num_processes 2 change_to_relu_ddp.py \
    --max_train_tokens 2000000 \
    --max_steps 4 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --skip_calibration \
    --resume_from_checkpoint "$OUTPUT_DIR/checkpoint-2" \
    --output_dir "${OUTPUT_DIR}-resumed" \
    --log_file "test_resume_stage2.log"

echo "完成！请检查终端输出以及 test_resume_stage2.log 中的 step 是否从 2 开始。"