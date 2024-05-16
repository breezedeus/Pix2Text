# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
import string
from typing import Sequence, List, Optional

import numpy as np
import cv2


def clip(x, min_value, max_value):
    return min(max(x, min_value), max_value)


class TextOcrEngine:
    """Text OCR Engine Wrapper"""

    name = 'unknown'

    def __init__(self, languages: Sequence[str], ocr_engine):
        self.languages = languages
        self.ocr_engine = ocr_engine

    def detect_only(self, img: np.ndarray, **kwargs):
        """
        Only detect the texts from the input image.
        Args:
            img (np.ndarray): RGB image with shape: (height, width, 3)
            kwargs: more configs

        Returns:
            Dict[str, List[dict]]: The dictionary contains the following keys:
               * 'detected_texts': list, each element stores the information of a detected box, recorded in a dictionary, including the following values:
                   'position': The rectangular box corresponding to the detected text; np.ndarray, shape: (4, 2), representing the coordinates (x, y) of the 4 points of the box;

                 Example:
                   {'detected_texts':
                       [{'position': array([[416,  77],
                                       [486,  13],
                                       [800, 325],
                                       [730, 390]], dtype=int32),
                        },
                        ...
                       ]
                   }
        """
        pass

    def recognize_only(self, img: np.ndarray, **kwargs):
        """
        Only recognize the texts for cropped images, which are from bboxes detected by detect_only.
        Args:
            img (): RGB image with shape [height, width] or [height, width, channel].
                    channel should be 1 (gray image) or 3 (RGB formatted color image). scaled in [0, 255];
            kwargs: more configs

        Returns:
            dict, with keys:
                - 'text' (str): The recognized text
                - 'score' (float): The score of the recognition result (confidence level), ranging from `[0, 1]`; the higher the score, the more reliable it is

            Example:
            ```
             {'score': 0.8812797665596008,
              'text': 'Current Line'}
            ```
        """
        pass

    def ocr(self, img: np.ndarray, rec_config: Optional[dict] = None, **kwargs):
        """
        Detect texts first, and then recognize the texts for detected bbox patches.
        Args:
            img (np.ndarray): RGB image with shape [height, width] or [height, width, channel].
                    channel should be 1 (gray image) or 3 (RGB formatted color image). scaled in [0, 255];
            rec_config (Optional[dict]): The config for recognition
            kwargs: more configs

        Returns:
            list of detected texts, which element is a dict, with keys:
                - 'text' (str): The recognized text
                - 'score' (float): The score of the recognition result (confidence level), ranging from `[0, 1]`; the higher the score, the more reliable it is
                - 'position' (np.ndarray): 4 x 2 array, representing the coordinates (x, y) of the 4 points of the box

            Example:
            ```
             [{'score': 0.88,
              'text': 'Line 1',
              'position': array([[146, 22],
                                 [179, 22],
                                 [179, 60],
                                 [146, 60]], dtype=int32)
              },
              {'score': 0.78,
              'text': 'Line 2'
              'position': array([[641, 115],
                                 [1180, 115],
                                 [1180, 244],
                                 [641, 244]], dtype=int32)
              }]
            ```
        """

        pass


class CnOCREngine(TextOcrEngine):
    name = 'cnocr'

    def detect_only(self, img: np.ndarray, **kwargs):
        outs = self.ocr_engine.det_model.detect(img, **kwargs)
        for out in outs['detected_texts']:
            out['position'] = out.pop('box')
        return outs

    def recognize_only(self, img: np.ndarray, **kwargs):
        try:
            return self.ocr_engine.ocr_for_single_line(img)
        except:
            return {'text': '', 'score': 0.0}

    def ocr(self, img: np.ndarray, rec_config: Optional[dict] = None, **kwargs) -> str:
        rec_config = rec_config or {}
        outs = self.ocr_engine.ocr(img, **rec_config)
        return outs


class EasyOCREngine(TextOcrEngine):
    name = 'easyocr'

    def detect_only(self, img: np.ndarray, **kwargs):
        if 'resized_shape' in kwargs:
            kwargs.pop('resized_shape')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]
        horizontal_list, free_list = self.ocr_engine.detect(img, **kwargs)
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        bboxes = []
        for x1x2_y1y2 in horizontal_list:
            xmin, xmax, ymin, ymax = x1x2_y1y2
            xmin = clip(xmin, 0, width)
            xmax = clip(xmax, 0, width)
            ymin = clip(ymin, 0, height)
            ymax = clip(ymax, 0, height)
            box = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            bboxes.append({'position': box})
        for bbox in free_list:
            if bbox:
                bboxes.append({'position': np.array(bbox)})
        return {'detected_texts': bboxes}

    def recognize_only(self, img: np.ndarray, **kwargs) -> dict:
        out = {'text': '', 'score': 0.0}
        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            result = self.ocr_engine.recognize(img, **kwargs)
            if result:
                out = {'text': result[0][1], 'score': result[0][2]}
        except:
            pass
        return out

    def ocr(
        self, img: np.ndarray, rec_config: Optional[dict] = None, **kwargs
    ) -> List[dict]:
        rec_config = rec_config or {}
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.ocr_engine.readtext(img, **rec_config)
        outs = []
        for result in results:
            outs.append(
                {'text': result[1], 'score': result[2], 'position': np.array(result[0])}
            )
        return outs


def prepare_ocr_engine(languages: Sequence[str], ocr_engine_config):
    if len(set(languages).difference({'en', 'ch_sim'})) == 0:
        from cnocr import CnOcr

        if 'ch_sim' not in languages and 'cand_alphabet' not in ocr_engine_config:  # only recognize english characters
            ocr_engine_config['cand_alphabet'] = string.printable
        ocr_engine = CnOcr(**ocr_engine_config)
        engine_wrapper = CnOCREngine(languages, ocr_engine)
    else:
        try:
            from easyocr import Reader
        except:
            raise ImportError('Please install easyocr first: pip install easyocr')
        gpu = False
        if 'context' in ocr_engine_config:
            context = ocr_engine_config.pop('context').lower()
            gpu = 'gpu' in context or 'cuda' in context
        ocr_engine = Reader(lang_list=list(languages), gpu=gpu, **ocr_engine_config)
        engine_wrapper = EasyOCREngine(languages, ocr_engine)
    return engine_wrapper
