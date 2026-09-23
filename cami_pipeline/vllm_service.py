# -*- coding: utf-8 -*-
# @Time    : 2026/4/6  09:54
# @Author  : psi

"""
部署命令:
# 1. 安装 vllm (在 verl 环境下)
# uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# 2. 启动 vllm 服务 (使用 ModelScope 镜像)
# VLLM_USE_MODELSCOPE=true vllm serve /data3/xj/pre_model/Qwen3.5-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3

# 3. 环境准备 (在 verl 环境下)
# pip install openai socksio
"""

from openai import OpenAI

model_file_path = ["/data3/xj/pre_model/Qwen3.5-4B-grpo"][0]

# 全局客户端实例（延迟初始化）
_client = None


def get_client():
    """获取或创建 OpenAI 客户端（延迟初始化）"""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="http://localhost:8000/v1",  # vLLM 默认端口
            api_key="not-needed"  # vLLM 本地部署不需要 API key
        )
    return _client


def call_vllm_service(message, model_name=model_file_path, max_tokens=4096):
    """
    调用本地 vLLM 服务进行对话

    Args:
        message: 用户输入的消息 (str 或 list)
        model_name: 模型名称，默认为 Qwen3.5-27B
        max_tokens: 最大生成token数（注意：输入+输出不能超过模型上下文长度）

    Returns:
        chat_response: OpenAI API 响应对象
    """
    # 处理消息格式
    if isinstance(message, str):
        messages = [{"role": "user", "content": message}]
    else:
        messages = message

    client = get_client()

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.95,
        # presence_penalty=1.5,
        # extra_body={
        #     "top_k": 20,
        # },
    )

    return chat_response


def call_vllm_service_simple(message, model_name=None):
    """
    简化版调用，直接返回回复内容
    """
    if model_name is None:
        model_name = model_file_path
    response = call_vllm_service(message, model_name=model_name)
    return response.choices[0].message.content


def call_vllm_service_stream(message, model_name=None):
    """
    流式调用，逐字返回
    """
    if model_name is None:
        model_name = model_file_path

    if isinstance(message, str):
        messages = [{"role": "user", "content": message}]
    else:
        messages = message

    client = get_client()

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    import sys

    # 测试导入
    print("Testing import...")
    print(f"Python: {sys.executable}")

    # 检查服务是否可用
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        print("✓ vLLM service is running on port 8000")
    except Exception as e:
        print(f"✗ vLLM service not available: {e}")
        print("\n请先启动服务:")
        print("VLLM_USE_MODELSCOPE=true vllm serve /data3/xj/pre_model/Qwen3.5-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3")
        sys.exit(1)

    # 运行测试
    print("\nRunning test...")
    messages = [
        {"role": "user", "content": '你好'},
    ]

    try:
        # 使用较小的 max_tokens 进行测试（输入+输出不能超过32768）
        chat_response = call_vllm_service(messages, model_name=model_file_path, max_tokens=1024)
        print("\nChat response:", chat_response)
        print("\nContent:", chat_response.choices[0].message.content)
        print("\n✓ Test passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
