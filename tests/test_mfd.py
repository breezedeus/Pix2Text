# coding: utf-8

from pix2text import set_logger
from pix2text import MathFormulaDetector

logger = set_logger()


def test_formula_detector():
    det = MathFormulaDetector()
    image_fps = [
        'docs/examples/mixed.jpg',
        'docs/examples/vietnamese.jpg',
    ]
    outs = det.detect(image_fps[0])
    print(outs)
    outs = det.detect(image_fps)
    # outs = det.detect(image_fps, visualize=True, save=True)
    print(outs)

