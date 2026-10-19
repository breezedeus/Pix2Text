"""Exercise the real LiteLLM adapters with HTTP intercepted after serialization."""
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import httpx
import pytest
from PIL import Image


SPEC = spec_from_file_location(
    "minimax_vlm_wire", Path(__file__).parents[1] / "pix2text" / "vlm_api.py"
)
vlm_api = module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_api)


@pytest.mark.parametrize("host", ["api.minimax.io", "api.minimaxi.com"])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
def test_minimax_vlm_serialized_requests(monkeypatch, host, protocol, model):
    anthropic = protocol == "anthropic"
    base = f"https://{host}/{'anthropic' if anthropic else 'v1'}"
    captures = []

    def send(client, request, **kwargs):
        captures.append(request)
        body = json.loads(request.content)
        assert str(request.url) == base + ("/v1/messages" if anthropic else "/chat/completions")
        assert body["model"] == model
        assert request.headers["x-api-key" if anthropic else "authorization"] == (
            "test-key" if anthropic else "Bearer test-key"
        )
        assert body["max_tokens"] == 1024
        for key in ("thinking", "reasoning_effort", "output_config", "service_tier", "reasoning_split"):
            assert key not in body
        content = body["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Read the image"}
        if anthropic:
            assert content[1]["type"] == "image"
            assert content[1]["source"]["type"] == "base64"
            assert content[1]["source"]["data"]
            payload = {
                "id": "msg_test", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": "recognized text"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        else:
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            payload = {
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "reasoning_content": "private", "content": "recognized text",
                }}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    vlm = vlm_api.Vlm(model_name=f"{protocol}/{model}", api_key="test-key", api_base=base)
    result = vlm(Image.new("RGB", (4, 4)), prompt="Read the image", parsing_func=None, max_tokens=1024)
    assert result == {"text": "recognized text", "score": 0.0}
    assert len(captures) == 1


def test_non_minimax_model_and_per_call_options_are_preserved(monkeypatch):
    captures = []

    def completion(**kwargs):
        captures.append(kwargs)
        return [{"choices": [{"message": {"content": "text"}}]}]

    monkeypatch.setattr(vlm_api, "batch_completion", completion)
    vlm = vlm_api.Vlm(model_name="openai/custom-model", api_key="test-key", api_base="https://saved.example/v1")
    vlm(Image.new("RGB", (4, 4)), parsing_func=None, api_base="https://override.example/v1", thinking={"type": "enabled", "budget_tokens": 8192})
    assert captures[0]["model"] == "openai/custom-model"
    assert captures[0]["api_base"] == "https://override.example/v1"
    assert captures[0]["thinking"] == {"type": "enabled", "budget_tokens": 8192}
