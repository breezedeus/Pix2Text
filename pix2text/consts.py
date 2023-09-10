# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

from pathlib import Path

from .__version__ import __version__


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '0.1.*'，对应的 MODEL_VERSION 都是 '0.1'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])

# 图片分类模型对应的类别标签
IMAGE_TYPES = ('general', 'english', 'formula')

# LATEX_OCR 使用的配置信息
LATEX_CONFIG_FP = Path(__file__).parent.absolute() / 'latex_config.yaml'

# 模型下载根地址
HF_HUB_REPO_ID = "breezedeus/cnstd-cnocr-models"
HF_HUB_SUBFOLDER = "models/pix2text/%s" % MODEL_VERSION


def format_hf_hub_url(url: str) -> dict:
    return {
        'repo_id': HF_HUB_REPO_ID,
        'subfolder': HF_HUB_SUBFOLDER,
        'filename': url,
    }


CLF_MODEL_URL_FMT = '%s.zip'
