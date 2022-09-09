# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

from pathlib import Path

from .__version__ import __version__
from .category_mapping import CATEGORY_MAPPINGS


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '0.1.*'，对应的 MODEL_VERSION 都是 '0.1'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])

# 图片分类模型对应的类别标签
IMAGE_TYPES = ('general', 'english', 'formula')

# LATEX_OCR 使用的配置信息
LATEX_CONFIG_FP = Path(__file__).parent.absolute() / 'latex_config.yaml'

# 模型下载根地址
ROOT_URL = (
    'https://huggingface.co/breezedeus/cnstd-cnocr-models/resolve/main/models/pix2text/%s/'
    % MODEL_VERSION
)

CLF_MODEL_URL_FMT = ROOT_URL + '%s.zip'
