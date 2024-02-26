# coding: utf-8
#! pip install pillow transformers optimum[onnxruntime]
from PIL import Image
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

processor = TrOCRProcessor.from_pretrained('breezedeus/pix2text-mfr')
model = ORTModelForVision2Seq.from_pretrained('breezedeus/pix2text-mfr', use_cache=False)

image_fps = [
    # 'https://github.com/breezedeus/Pix2Text/blob/main/docs/examples/formula.jpg?raw=true',
    'docs/examples/formula.jpg',
    # 'examples/0000186.png',
]
images = [Image.open(fp).convert('RGB') for fp in image_fps]
pixel_values = processor(images=images, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f'generated_ids: {generated_ids}, \ngenerated text: {generated_text}')

