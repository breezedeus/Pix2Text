# coding: utf-8
import time

from pix2text import set_logger
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
