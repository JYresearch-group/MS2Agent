import base64
import hashlib
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import requests
from casmi_pipeline.vllm_service import call_vllm_service
from casmi_pipeline.io_utils import write_json_atomic


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError("No JSON object found", text, 0)


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return parse_first_json_object(cleaned)


def extract_message_content(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError(f"No choices found in response: {response_json}")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def image_file_to_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_llm_messages(
    system_prompt: str,
    user_prompt: str,
    image_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if image_path is None:
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_file_to_data_uri(image_path)}},
                ],
            }
        )
    return messages


def make_cache_key(model: str, messages: Sequence[Dict[str, Any]]) -> str:
    serializable = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((model + "\n" + serializable).encode("utf-8")).hexdigest()


def call_chat_completion(
    session: requests.Session,
    api_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, Any]],
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": list(messages),
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            if 'openrouter' in api_url:
                payload = {"model": model, "messages": list(messages), "reasoning": {"enabled": True}}
                response = session.post(url=api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
            elif 'deepseek' in api_url:
                payload = {"model": model, "messages": list(messages), "thinking": {"type": "enabled"}}
                response = session.post(url=api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
            elif 'aliyuncs' in api_url:
                from openai import OpenAI
                client = OpenAI(
                    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                completion = client.chat.completions.create(
                    model=model,  # 您可以按需更换为其它深度思考模型
                    messages=messages,
                    extra_body={"enable_thinking": True},
                    stream=False
                )
                completion11 = json.loads(completion.to_json())
                return completion11
            elif "localhost:8000" in api_url:
                chat_response = call_vllm_service(list(messages))
                xxx1 = json.loads(chat_response.json())
                return xxx1
            else:
                response = session.post(api_url, headers=headers, json=payload, timeout=timeout)

            if response.status_code >= 500 or response.status_code == 429:
                raise requests.HTTPError(
                    f"Transient HTTP error {response.status_code}: {response.text[:500]}",
                    response=response,
                )
            response.raise_for_status()
            # print(response.json())
            # os._exit(0)
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {exc}") from exc
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("Unreachable retry logic")


def cached_llm_json(
    session: requests.Session,
    cache_dir: Path,
    api_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, Any]],
    temperature: float,
    timeout: int,
    max_retries: int,
    force: bool,
) -> Tuple[Dict[str, Any], str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = make_cache_key(model, messages)
    cache_path = cache_dir / f"{cache_key}.json"

    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached["parsed_json"], cached["raw_content"]

    response_json = call_chat_completion(
        session=session,
        api_url=api_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    raw_content = extract_message_content(response_json)
    parsed_json = extract_json_object(raw_content)

    write_json_atomic(
        cache_path,
        {
            "response_json": response_json,
            "raw_content": raw_content,
            "parsed_json": parsed_json,
        },
    )

    return parsed_json, raw_content
