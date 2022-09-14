# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

import os
from glob import glob
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
from cnocr import CnOcr, ImageClassifier

from .consts import IMAGE_TYPES, LATEX_CONFIG_FP, MODEL_VERSION, CLF_MODEL_URL_FMT
from .latex_ocr import LatexOCR
from .utils import data_dir, get_model_file

logger = logging.getLogger(__name__)


DEFAULT_CONFIGS = {
    'clf': {
        'base_model_name': 'mobilenet_v2',
        'categories': IMAGE_TYPES,
        'transform_configs': {
            'crop_size': [150, 450],
            'resize_size': 160,
            'resize_max_size': 1000,
        },
        'model_dir': Path(data_dir()) / 'clf',
        'model_fp': None,  # 如果指定，直接使用此模型文件
    },
    'general': {},
    'english': {'det_model_name': 'en_PP-OCRv3_det', 'rec_model_name': 'en_PP-OCRv3'},
    'formula': {
        'config': LATEX_CONFIG_FP,
        'checkpoint': Path(data_dir()) / 'formular' / 'weights.pth',
        'no_resize': False,
    },
    'thresholds': {
        'formula2general': 0.65,  # 如果识别为 `formula` 类型，但阈值小于此值，则改为 `general` 类型
        'english2general': 0.75,  # 如果识别为 `english` 类型，但阈值小于此值，则改为 `general` 类型
    },
}


class Pix2Text(object):
    MODEL_FILE_PREFIX = 'pix2text-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        *,
        clf_config: Dict[str, Any] = None,
        general_config: Dict[str, Any] = None,
        english_config: Dict[str, Any] = None,
        formula_config: Dict[str, Any] = None,
        thresholds: Dict[str, Any] = None,
        device: str = 'cpu',  # ['cpu', 'cuda', 'gpu']
        **kwargs,
    ):
        """

        Args:
            clf_config (dict): 分类模型对应的配置信息；默认为 `None`，表示使用默认配置
            general_config (dict): 通用模型对应的配置信息；默认为 `None`，表示使用默认配置
            english_config (dict): 英文模型对应的配置信息；默认为 `None`，表示使用默认配置
            formula_config (dict): 公式识别模型对应的配置信息；默认为 `None`，表示使用默认配置
            thresholds (dict): 识别阈值对应的配置信息；默认为 `None`，表示使用默认配置
            device (str): 使用什么资源进行计算，支持 `['cpu', 'cuda', 'gpu']`；默认为 `cpu`
            **kwargs (): 预留的其他参数；目前未被使用
        """
        if device.lower() == 'gpu':
            device = 'cuda'
        self.device = device
        thresholds = thresholds or DEFAULT_CONFIGS['thresholds']
        self.thresholds = deepcopy(thresholds)

        (
            clf_config,
            general_config,
            english_config,
            formula_config,
        ) = self._prepare_configs(
            clf_config, general_config, english_config, formula_config, device
        )
        _clf_config = deepcopy(clf_config)
        _clf_config.pop('model_dir')
        _clf_config.pop('model_fp')
        self.image_clf = ImageClassifier(**_clf_config)

        self.general_ocr = CnOcr(**general_config)
        self.english_ocr = CnOcr(**english_config)
        self.latex_model = LatexOCR(formula_config)

        self._assert_and_prepare_clf_model(clf_config)

    def _prepare_configs(
        self, clf_config, general_config, english_config, formula_config, device
    ):
        def _to_default(_conf, _def_val):
            if _conf is None:
                _conf = _def_val
            return _conf

        clf_config = _to_default(clf_config, DEFAULT_CONFIGS['clf'])
        general_config = _to_default(general_config, DEFAULT_CONFIGS['general'])
        general_config['context'] = device
        english_config = _to_default(english_config, DEFAULT_CONFIGS['english'])
        english_config['context'] = device
        formula_config = _to_default(formula_config, DEFAULT_CONFIGS['formula'])
        formula_config['device'] = device
        return clf_config, general_config, english_config, formula_config

    def _assert_and_prepare_clf_model(self, clf_config):
        model_file_prefix = '{}-{}'.format(
            self.MODEL_FILE_PREFIX, clf_config['base_model_name']
        )
        model_dir = clf_config['model_dir']
        model_fp = clf_config['model_fp']

        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        fps = glob(os.path.join(model_dir, model_file_prefix) + '*.ckpt')
        if len(fps) > 1:
            raise ValueError(
                'multiple .ckpt files are found in %s, not sure which one should be used'
                % model_dir
            )
        elif len(fps) < 1:
            logger.warning('no .ckpt file is found in %s' % model_dir)
            url = CLF_MODEL_URL_FMT % clf_config['base_model_name']
            get_model_file(url, model_dir)  # download the .zip file and unzip
            fps = glob(os.path.join(model_dir, model_file_prefix) + '*.ckpt')

        model_fp = fps[0]
        self.image_clf.load(model_fp, self.device)

    @classmethod
    def from_config(cls, total_configs: Optional[dict] = None, device: str = 'cpu'):
        total_configs = total_configs or DEFAULT_CONFIGS
        return cls(
            clf_config=total_configs.get('clf', dict()),
            general_config=total_configs.get('general', dict()),
            english_config=total_configs.get('english', dict()),
            formula_config=total_configs.get('formula', dict()),
            thresholds=total_configs.get('thresholds', DEFAULT_CONFIGS['thresholds']),
            device=device,
        )

    def __call__(self, img: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        return self.recognize(img)

    def recognize(self, img: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`

        Returns: a dict, with keys:
           `image_type`: 图像类别；
           `text`: 识别出的文字或Latex公式

        """
        if isinstance(img, Image.Image):
            _img = torch.tensor(np.asarray(img.convert('RGB')))
            res = self.image_clf.predict_images([_img])[0]
        else:
            res = self.image_clf.predict_images([img])[0]
        logger.debug('CLF Result: %s', res)
        image_type = res[0]
        if res[1] < self.thresholds['formula2general'] and res[0] == 'formula':
            image_type = 'general'
        if res[1] < self.thresholds['english2general'] and res[0] == 'english':
            image_type = 'general'
        if image_type == 'formula':
            result = self._latex(img)
        else:
            result = self._ocr(img, image_type)

        return {'image_type': image_type, 'text': result}

    def _ocr(self, image, image_type):
        ocr_model = self.english_ocr if image_type == 'english' else self.general_ocr
        result = ocr_model.ocr(image)
        texts = [_one['text'] for _one in result]
        result = '\n'.join(texts)
        return result

    def _latex(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        out = self.latex_model(image)
        return str(out)


if __name__ == '__main__':
    from .utils import set_logger

    logger = set_logger(log_level='DEBUG')

    p2t = Pix2Text()
    img = 'docs/examples/english.jpg'
    out = p2t.recognize(Image.open(img))
    logger.info(out)
