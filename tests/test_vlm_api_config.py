from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pix2text.pix_to_text import (
    prepare_table_ocr_engine,
    prepare_text_formula_ocr_engine,
)
from pix2text.text_formula_ocr import VlmTextFormulaOCR

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


def test_vlm_text_formula_ocr_forwards_call_options():
    calls = []

    class FakeVlm:
        def __call__(self, imgs, **kwargs):
            calls.append(kwargs)
            return [{"text": "recognized text", "score": 0.0} for _ in imgs]

    ocr = VlmTextFormulaOCR(vlm=FakeVlm())

    result = ocr.recognize_text(
        ["image.png"],
        rec_config={"temperature": 0.1},
        api_base="https://api.minimaxi.com/v1",
    )

    assert result == ["recognized text"]
    assert calls == [
        {"temperature": 0.1, "api_base": "https://api.minimaxi.com/v1"}
    ]

    result = ocr.recognize_formula(
        ["image.png"],
        rec_config={"api_base": "https://api.minimax.io/v1"},
        api_base="https://api.minimaxi.com/v1",
    )

    assert result == ["recognized text"]
    assert calls[-1] == {"api_base": "https://api.minimaxi.com/v1"}


def test_api_base_is_forwarded_from_pix2text_config():
    api_base = "https://api.minimax.io/v1"

    text_formula_ocr = prepare_text_formula_ocr_engine(
        {
            "model_type": "VlmTextFormulaOCR",
            "model_name": "openai/MiniMax-M3",
            "api_key": "test-key",
            "api_base": api_base,
        },
        enable_formula=True,
        device="cpu",
    )
    table_ocr = prepare_table_ocr_engine(
        {
            "model_type": "VlmTableOCR",
            "model_name": "openai/MiniMax-M3",
            "api_key": "test-key",
            "api_base": api_base,
        },
        device="cpu",
        text_formula_ocr=text_formula_ocr,
    )

    assert text_formula_ocr.vlm.api_base == api_base
    assert table_ocr.vlm.api_base == api_base
