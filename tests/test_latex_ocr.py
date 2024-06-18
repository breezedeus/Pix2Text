# coding: utf-8
import os
import os.path
import time

from pix2text import set_logger, read_img
from pix2text.latex_ocr import *

logger = set_logger()


def test_download_model():
    latex_ocr = LatexOCR()

    image_fps = [
        'docs/examples/formula.jpg',
        'docs/examples/math-formula-42.png',
    ]
    start_time = time.time()
    outs = latex_ocr.recognize(image_fps)
    logger.info(f'average cost time: {(time.time() - start_time) / len(image_fps):.4f} seconds')
    for img, out in zip(image_fps, outs):
        logger.info(f'- image: {img}, out: \n\t{out}')


def test_infer_with_transformers():
    from PIL import Image
    from transformers import TrOCRProcessor
    from optimum.onnxruntime import ORTModelForVision2Seq

    model_dir = os.path.expanduser('~/.pix2text/1.1/mfr-pro-onnx')
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = ORTModelForVision2Seq.from_pretrained(model_dir, use_cache=False)

    image_fps = [
        'docs/examples/formula.jpg',
        'docs/examples/math-formula-42.png',
    ]
    images = [read_img(fp, return_type='Image') for fp in image_fps]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    # print(f'pixel_values', pixel_values)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f'generated_ids: {generated_ids}, \ngenerated text: {generated_text}')


def test_infer():
    more_model_configs = {}
    latex_ocr = LatexOCR(more_model_configs=more_model_configs)

    image_fps = [
        'docs/examples/formula.jpg',
        'docs/examples/math-formula-42.png',
    ]
    start_time = time.time()
    outs = latex_ocr.recognize(image_fps, batch_size=2)
    logger.info(f'average cost time: {(time.time() - start_time) / len(image_fps):.4f} seconds')
    for img, out in zip(image_fps, outs):
        logger.info(f'- image: {img}, out: \n\t{out}')