# VLLM 环境设置指南

## 1. 环境状态

### 已完成：
- [x] `verl` 环境检查 (Python 3.10, PyTorch 2.10.0+cu128)
- [x] vLLM 0.19.0 (nightly) 安装
- [x] openai 客户端安装
- [x] modelscope 安装
- [x] `vllm_service.py` 代码编写与测试导入

### 待完成：
- [ ] flash-attn 编译安装（需要 30-60 分钟，与 PyTorch 2.10/CUDA 12.8 兼容）

## 2. 手动完成 flash-attn 安装

由于编译需要较长时间，请在终端手动运行：

```bash
# 激活 verl 环境
source /root/lushuo/anaconda3/bin/activate verl

# 安装 flash-attn（从源码编译，约 30-60 分钟）
export MAX_JOBS=8
pip install flash-attn --no-cache-dir --no-build-isolation

# 验证安装
python -c "import flash_attn; print(flash_attn.__version__)"
```

## 3. 启动 VLLM 服务

flash-attn 安装完成后，启动服务：

```bash
# 方法 1: 使用提供的脚本
cd /data0/xj/Slime_MS/llm_ms_slime/casmi_pipeline
bash start_vllm.sh

# 方法 2: 手动启动
export VLLM_USE_MODELSCOPE=true
python -m vllm.entrypoints.cli.main serve /data3/xj/pre_model/Qwen3.5-27B \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

## 4. 测试代码

服务启动后，测试调用：

```python
from casmi_pipeline.vllm_service import call_vllm_service, call_vllm_service_simple

# 测试调用
response = call_vllm_service("你好，请介绍一下自己")
print(response.choices[0].message.content)

# 简化调用
content = call_vllm_service_simple("你好")
print(content)
```

## 5. 文件说明

- `vllm_service.py` - VLLM 服务调用客户端
- `start_vllm.sh` - 服务启动脚本
- `SETUP.md` - 本说明文件

## 6. 故障排除

### 问题：flash-attn CUDA 版本不兼容
**解决**：确保从源码重新编译 flash-attn，使其与当前 PyTorch 版本匹配。

### 问题：模型加载失败
**解决**：检查模型路径是否正确，并确保 `VLLM_USE_MODELSCOPE=true` 已设置。

### 问题：显存不足
**解决**：减小 `--max-model-len` 参数值，或减小 `--tensor-parallel-size` 使用更多 GPU。
