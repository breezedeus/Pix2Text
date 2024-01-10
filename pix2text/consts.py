# coding: utf-8
# Copyright (C) 2022-2024, [Breezedeus](https://github.com/breezedeus).
import os
import logging
from collections import OrderedDict
from copy import copy, deepcopy
from pathlib import Path
from typing import Set, Tuple, Dict, Any, Optional

from .__version__ import __version__

logger = logging.getLogger(__name__)

# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '0.1.*'，对应的 MODEL_VERSION 都是 '0.1'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
DOWNLOAD_SOURCE = os.environ.get('PIX2TEXT_DOWNLOAD_SOURCE', 'CN')

# 图片分类模型对应的类别标签
IMAGE_TYPES = ('general', 'english', 'formula')

# LATEX_OCR 使用的配置信息
LATEX_CONFIG_FP = Path(__file__).parent.absolute() / 'latex_config.yaml'

# 模型下载根地址
HF_HUB_REPO_ID = "breezedeus/cnstd-cnocr-models"
HF_HUB_SUBFOLDER = "models/pix2text/%s" % MODEL_VERSION
PAID_HF_HUB_REPO_ID = "breezedeus/paid-models"
PAID_HF_HUB_SUBFOLDER = "cnocr/%s" % MODEL_VERSION
CN_OSS_ENDPOINT = (
        "https://sg-models.oss-cn-beijing.aliyuncs.com/pix2text/%s/" % MODEL_VERSION
)


def format_model_info(info: dict, is_paid_model=False) -> dict:
    out_dict = copy(info)

    if is_paid_model:
        repo_id = PAID_HF_HUB_REPO_ID
        subfolder = PAID_HF_HUB_SUBFOLDER
    else:
        repo_id = HF_HUB_REPO_ID
        subfolder = HF_HUB_SUBFOLDER
        out_dict['cn_oss'] = CN_OSS_ENDPOINT
    out_dict.update(
        {'repo_id': repo_id, 'subfolder': subfolder,}
    )
    return out_dict


class AvailableModels(object):
    P2T_SPACE = '__cnocr__'

    FREE_MODELS = OrderedDict(
        {
            ('mfr', 'pytorch'): {
                'url': 'mfr-origin.zip',
                'fn': 'weights.pth',
            },
            ('resizer', 'pytorch'): {
                'url': 'resizer.zip',
                'fn': 'image_resizer.pth',
            },
        }
    )

    PAID_MODELS = OrderedDict(
        {
            ('mfr-pro', 'pytorch'): {
                'url': 'mfr-20230702.zip',
                'fn': 'p2t-mfr-20230702.pth',
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
        is_paid_model = False
        if (model_name, model_backend) in self.P2T_MODELS:
            info = self.P2T_MODELS[(model_name, model_backend)]
            is_paid_model = (model_name, model_backend) in self.PAID_MODELS
        elif (model_name, model_backend) in self.OUTER_MODELS:
            info = self.OUTER_MODELS[(model_name, model_backend)]
        else:
            logger.warning(
                'no url is found for model %s' % ((model_name, model_backend),)
            )
            return None
        info = format_model_info(info, is_paid_model=is_paid_model)
        return info


AVAILABLE_MODELS = AvailableModels()
