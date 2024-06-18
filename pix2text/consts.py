# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
import os
import logging
from collections import OrderedDict
from copy import copy, deepcopy
from typing import Set, Tuple, Dict, Any, Optional

from .__version__ import __version__

logger = logging.getLogger(__name__)

# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '1.0.*'，对应的 MODEL_VERSION 都是 '1.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
DOWNLOAD_SOURCE = os.environ.get('PIX2TEXT_DOWNLOAD_SOURCE', 'HF')

CN_OSS_ENDPOINT = (
        "https://sg-models.oss-cn-beijing.aliyuncs.com/pix2text/%s/" % MODEL_VERSION
)


def format_model_info(info: dict) -> dict:
    out_dict = copy(info)
    out_dict['cn_oss'] = CN_OSS_ENDPOINT
    return out_dict


class AvailableModels(object):
    P2T_SPACE = '__pix2text__'

    FREE_MODELS = OrderedDict(
        {
            ('mfr', 'onnx'): {
                'filename': 'p2t-mfr-onnx.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfr',
                'local_model_id': 'mfr-onnx',
            },
            ('mfd', 'onnx'): {
                'filename': 'p2t-mfd-onnx.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfd',
                'local_model_id': 'mfd-onnx',
            },
        }
    )

    PAID_MODELS = OrderedDict(
        {
            ('mfr', 'pytorch'): {
                'filename': 'p2t-mfr-pytorch.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfr-pytorch',
                'local_model_id': 'mfr-pytorch',
            },
            ('mfr-pro', 'onnx'): {
                'filename': 'p2t-mfr-pro-onnx.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfr-pro',
                'local_model_id': 'mfr-pro-onnx',
            },
            ('mfr-pro', 'pytorch'): {
                'filename': 'p2t-mfr-pro-pytorch.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfr-pro-pytorch',
                'local_model_id': 'mfr-pro-pytorch',
            },
            ('mfr-plus', 'onnx'): {
                'filename': 'p2t-mfr-plus-onnx.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfr-plus',
                'local_model_id': 'mfr-plus-onnx',
            },
            ('mfr-plus', 'pytorch'): {
                'filename': 'p2t-mfr-plus-pytorch.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfr-plus-pytorch',
                'local_model_id': 'mfr-plus-pytorch',
            },
            ('mfd', 'pytorch'): {
                'filename': 'p2t-mfd-pytorch.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfd-pytorch',
                'local_model_id': 'mfd-pytorch',
            },
            ('mfd-advanced', 'onnx'): {
                'filename': 'p2t-mfd-advanced-onnx.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfd-advanced',
                'local_model_id': 'mfd-advanced-onnx',
            },
            ('mfd-advanced', 'pytorch'): {
                'filename': 'p2t-mfd-advanced-pytorch.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfd-advanced-pytorch',
                'local_model_id': 'mfd-advanced-pytorch',
            },
            ('mfd-pro', 'onnx'): {
                'filename': 'p2t-mfd-pro-onnx.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfd-pro',
                'local_model_id': 'mfd-pro-onnx',
            },
            ('mfd-pro', 'pytorch'): {
                'filename': 'p2t-mfd-pro-pytorch.zip',  # download the file from CN OSS
                'hf_model_id': 'breezedeus/pix2text-mfd-pro-pytorch',
                'local_model_id': 'mfd-pro-pytorch',
            },
        }
    )

    P2T_MODELS = deepcopy(FREE_MODELS)
    P2T_MODELS.update(PAID_MODELS)
    OUTER_MODELS = {}

    def all_models(self) -> Set[Tuple[str, str]]:
        return set(self.P2T_MODELS.keys()) | set(self.OUTER_MODELS.keys())

    def __contains__(self, model_name_backend: Tuple[str, str]) -> bool:
        return model_name_backend in self.all_models()

    def register_models(self, model_dict: Dict[Tuple[str, str], Any], space: str):
        assert not space.startswith('__')
        for key, val in model_dict.items():
            if key in self.P2T_MODELS or key in self.OUTER_MODELS:
                logger.warning(
                    'model %s has already existed, and will be ignored' % key
                )
                continue
            val = deepcopy(val)
            val['space'] = space
            self.OUTER_MODELS[key] = val

    def get_space(self, model_name, model_backend) -> Optional[str]:
        if (model_name, model_backend) in self.P2T_MODELS:
            return self.P2T_SPACE
        elif (model_name, model_backend) in self.OUTER_MODELS:
            return self.OUTER_MODELS[(model_name, model_backend)]['space']
        return self.P2T_SPACE

    def get_info(self, model_name, model_backend) -> Optional[dict]:
        if (model_name, model_backend) in self.P2T_MODELS:
            info = self.P2T_MODELS[(model_name, model_backend)]
        elif (model_name, model_backend) in self.OUTER_MODELS:
            info = self.OUTER_MODELS[(model_name, model_backend)]
        else:
            logger.warning(
                'no url is found for model %s' % ((model_name, model_backend),)
            )
            return None
        info = format_model_info(info)
        return info


AVAILABLE_MODELS = AvailableModels()
