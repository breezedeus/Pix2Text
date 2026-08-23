from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "pix2text" / "vlm_api.py"
SPEC = spec_from_file_location("vlm_api", MODULE_PATH)
vlm_api = module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_api)


def test_minimax_openai_compatible_endpoint(monkeypatch):
    calls = []

    def fake_batch_completion(**kwargs):
        calls.append(kwargs)
        return [
            {
                "choices": [
                    {
                        "message": {"content": "recognized text"},
                        "logprobs": None,
                    }
                ]
            }
        ]

    def fake_encode_image(*args, **kwargs):
        return "image"

    monkeypatch.setattr(vlm_api, "batch_completion", fake_batch_completion)
    monkeypatch.setattr(vlm_api, "encode_image", fake_encode_image)

    vlm = vlm_api.Vlm(
        model_name="openai/MiniMax-M3",
        api_key="test-key",
        api_base="https://api.minimax.io/v1",
    )

    result = vlm("image.png", parsing_func=None)
    assert result == {"text": "recognized text", "score": 0.0}
    assert calls[-1]["api_base"] == "https://api.minimax.io/v1"

    vlm(
        "image.png",
        parsing_func=None,
        api_base="https://api.minimaxi.com/v1",
    )
    assert calls[-1]["api_base"] == "https://api.minimaxi.com/v1"
