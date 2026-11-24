"""Optional server-side Ark adapter; credentials never enter prompts or browser responses."""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def load_config(data):
    path = Path(data) / "ark-config.json"
    if not path.exists():
        return {"enabled": False}
    config = json.loads(path.read_text())
    return config


def validate(config):
    if not config.get("api_key") or not config.get("model"):
        raise ValueError("方舟配置缺少 API Key 或模型/接入点 ID")
    u = urllib.parse.urlparse(config.get("base_url", DEFAULT_BASE))
    if (
        u.scheme != "https"
        or u.hostname != "ark.cn-beijing.volces.com"
        or u.username
        or u.password
        or u.query
        or u.fragment
    ):
        raise ValueError("请配置火山方舟官方 HTTPS Base URL")
    if u.path.rstrip("/") not in ("/api/v3", "/api/coding/v3"):
        raise ValueError("方舟 Base URL 路径无效")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("云端接口发生重定向，已停止发送凭据")


def generate(config, messages, schema=None, planning=False, opener=None, vision=False):
    config = dict(config)
    if vision:
        if not config.get("vision_model"):
            raise ValueError("请先配置支持图片输入的方舟 vision_model；可与聊天模型相同")
        config["model"] = config["vision_model"]
    validate(config)
    wire = [{"role": m["role"], "content": m["content"]} for m in messages]
    for original, message in zip(messages, wire):
        if original.get("images"):
            message["content"] = [{"type": "text", "text": original["content"]}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}
                for img in original["images"]
            ]
    payload = {
        "model": config["model"],
        "messages": wire,
        "stream": False,
        "temperature": 0.2 if planning else 0.65,
        "max_tokens": 2048,
    }
    if schema:
        # JSON mode is more widely supported than strict JSON Schema; validate in caller.
        wire.insert(
            0,
            {
                "role": "system",
                "content": "只输出符合以下 JSON Schema 的 JSON 对象，不输出代码围栏或说明："
                + json.dumps(schema, ensure_ascii=False),
            },
        )
        payload["response_format"] = {"type": "json_object"}
    if config.get("disable_thinking"):
        payload["thinking"] = {"type": "disabled"}
    endpoint = "/chat/completions"
    responses = config.get("api_type") == "responses"
    if responses:
        endpoint = "/responses"
        inputs = []
        for message in wire:
            content = message["content"]
            if isinstance(content, list):
                content = [
                    {"type": "input_image", "image_url": part["image_url"]["url"]}
                    if part["type"] == "image_url" else
                    {"type": "input_text", "text": part["text"]}
                    for part in content
                ]
            inputs.append({"role": message["role"], "content": content})
        payload = {
            "model": config["model"], "input": inputs, "store": False,
            "stream": False, "max_output_tokens": 4096,
        }
        if schema:
            payload["text"] = {"format": {"type": "json_object"}}
        if config.get("disable_thinking"):
            payload["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(
        config.get("base_url", DEFAULT_BASE).rstrip("/") + endpoint,
        json.dumps(payload).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config["api_key"],
        },
    )
    opener = opener or urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as e:
        # Never echo upstream bodies: they can contain request data or credentials.
        error_code = None
        try:
            body = json.loads(e.read(16384))
            if isinstance(body, dict) and isinstance(body.get('error'), dict):
                error_code = body['error'].get('code')
        except (ValueError, TypeError, OSError):
            pass
        e.close()
        messages_by_status = {
            401: "密钥无效或已过期",
            403: "无权使用该模型",
            404: "模型 ID 或接口地址不正确",
            429: "额度不足或请求频率受限",
        }
        reason = '账户欠费，请在方舟控制台补足余额后重试' if error_code == 'AccountOverdueError' else messages_by_status.get(e.code, 'HTTP ' + str(e.code))
        raise ValueError(
            "方舟 API 请求失败："
            + reason
        ) from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("方舟 API 连接失败或超时，请检查网络") from None
    try:
        if responses:
            if result.get("status") != "completed":
                raise ValueError("方舟 Responses 未完成生成，请重试或检查输出上限")
            content = "".join(
                part["text"] for item in result.get("output", [])
                if item.get("type") == "message" and item.get("role") == "assistant"
                for part in item.get("content", []) if part.get("type") == "output_text"
            )
            choice = {"finish_reason": result["status"]}
        else:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("方舟未返回有效回复")
        if choice.get("finish_reason") == "length":
            raise ValueError("方舟输出被长度上限截断，请调整配置")
    except (KeyError, IndexError, TypeError):
        raise ValueError("方舟未返回有效回复") from None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("方舟未返回有效回复")
    return content, {
        "provider": "volcengine_ark",
        "model": result.get("model", config["model"]),
        "request_id": result.get("id"),
        "usage": result.get("usage"),
        "done_reason": choice.get("finish_reason"),
    }
