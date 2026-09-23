#!/bin/bash
# VLLM 服务启动脚本

#cd /data0/xj/Slime_MS/llm_ms_slime

# 使用 verl 环境
#PYTHON="/root/lushuo/anaconda3/envs/verl/bin/python"

# GPU 配置 - 使用前4张卡
#export CUDA_VISIBLE_DEVICES=0

# 配置
export VLLM_USE_MODELSCOPE=true
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # 或尝试 TORCH_SDPA

# 模型路径
MODEL_PATH="/root/autodl-tmp/shiyan/output/Qwen3.5-4B-grpo"
# 或使用 gemma 模型作为备选（非视觉模型）
# MODEL_PATH="/data3/xj/pre_model/gemma-4-31B-it"

# 启动参数
PORT=8000
TP_SIZE=1  # 张量并行大小（根据可用GPU数量调整）
MAX_MODEL_LEN=32768  # 最大模型长度

# 启动命令
echo "Starting vLLM service..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel: $TP_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo ""
echo "Command:"
echo "$PYTHON -m vllm.entrypoints.cli.main serve $MODEL_PATH --port $PORT --tensor-parallel-size $TP_SIZE --max-model-len $MAX_MODEL_LEN --reasoning-parser qwen3"
echo ""

$PYTHON -m vllm.entrypoints.cli.main serve $MODEL_PATH \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --reasoning-parser qwen3 \
    --trust-remote-code

# nohup bash start_vllm.sh > vllm_1.log 2>&1 &

