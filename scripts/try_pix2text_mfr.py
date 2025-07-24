# coding: utf-8
#! pip install pillow transformers optimum[onnxruntime]
from PIL import Image
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import VisionEncoderDecoderModel

def test_tokenizer_consistency(processor, test_strings=None):
    """
    测试Tokenizer的编码和解码是否一致
    
    Args:
        processor: TrOCRProcessor实例
        test_strings (list): 要测试的字符串列表
    """
    if test_strings is None:
        test_strings = [
            # "Hello, world!",
            # "你好，世界！",
            # "12345",
            # "1 + 1 = 2",
            # "The quick brown fox jumps over the lazy dog.",
            # "测试一下中文和English混合的情况",
            # "\mathcal{L}_{\mathrm{e y e l i d}} \,=\sum_{t=1}^{T} \sum_{v=1}^{V} \mathcal{M}_{v}^{\mathrm{( e y e l i d )}} \left( \left\| \hat{h}_{t, v}-x_{t, v} \right\|^{2} \right)",
            "\\hat { N } _ { 3 } = \\sum \\sp f _ { j = 1 } a _ { j } \\sp { \\dagger } a _ { j } .",
        ]

    print("\n" + "="*50)
    print("Testing Tokenizer Consistency")
    print("="*50)

    all_passed = True
    for text in test_strings:
        # 编码
        encoded = processor.tokenizer.encode_plus(text, return_tensors="pt")
        outs = processor.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=512,
        )["input_ids"]
        input_ids = encoded["input_ids"][0]
        breakpoint()

        # 解码
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=True)

        # 比较
        is_match = (text == decoded)
        if not is_match:
            all_passed = False

        print(f"\nOriginal: {repr(text)}")
        print(f"Encoded: {input_ids.tolist()}")
        print(f"Decoded: {repr(decoded)}")
        print(f"Match: {is_match}")

    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed! Tokenizer encoding and decoding are consistent.")
    else:
        print("❌ Some tests failed. Tokenizer encoding and decoding are not consistent.")
    print("="*50 + "\n")

model = 'breezedeus/pix2text-mfr'
model = 'models/checkpoint-683356'
processor = TrOCRProcessor.from_pretrained(model)

# 测试Tokenizer的编码和解码是否一致
# test_tokenizer_consistency(processor)

# model = ORTModelForVision2Seq.from_pretrained(model, use_cache=False)

model = VisionEncoderDecoderModel.from_pretrained(model)

image_fps = [
    # 'https://github.com/breezedeus/Pix2Text/blob/main/docs/examples/formula.jpg?raw=true',
    'docs/examples/formula.jpg',
    # '/Users/king/Documents/WhatIHaveDone/Test/syndoc/output-latex/sqrt_tex/150-cmbright.jpg'
    # 'examples/0000186.png',
]
images = [Image.open(fp).convert('RGB') for fp in image_fps]
pixel_values = processor(images=images, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f'generated_ids: {generated_ids}, \ngenerated text: {generated_text}')
