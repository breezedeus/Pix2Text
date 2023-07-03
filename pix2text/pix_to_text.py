# coding: utf-8
# Copyright (C) 2022-2023, [Breezedeus](https://www.breezedeus.com).

import os
from glob import glob
import logging
from itertools import chain
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from copy import deepcopy, copy

from PIL import Image
import numpy as np
import torch
from cnocr import CnOcr, ImageClassifier
from cnstd.utils import get_model_file
from cnstd import LayoutAnalyzer
from cnstd.yolov7.consts import CATEGORY_DICT

from .utils import sort_boxes, rotated_box_to_horizontal, is_valid_box, list2box
from cnstd.yolov7.general import xyxy24p, box_partial_overlap

from .consts import (
    IMAGE_TYPES,
    LATEX_CONFIG_FP,
    MODEL_VERSION,
    CLF_MODEL_URL_FMT,
    format_hf_hub_url,
)
from .latex_ocr import LatexOCR
from .utils import (
    data_dir,
    read_img,
    save_layout_img,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIGS = {
    'analyzer': {'model_name': 'mfd'},
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
        'checkpoint': Path(data_dir()) / 'formula' / 'weights.pth',
        'no_resize': False,
    },
    'thresholds': {  # 用于clf场景
        'formula2general': 0.65,  # 如果识别为 `formula` 类型，但阈值小于此值，则改为 `general` 类型
        'english2general': 0.75,  # 如果识别为 `english` 类型，但阈值小于此值，则改为 `general` 类型
    },
}


class Pix2Text(object):
    MODEL_FILE_PREFIX = 'pix2text-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        *,
        analyzer_config: Dict[str, Any] = None,
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
            analyzer_config (dict): Analyzer模型对应的配置信息；默认为 `None`，表示使用默认配置
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
            analyzer_config,
            clf_config,
            general_config,
            english_config,
            formula_config,
        ) = self._prepare_configs(
            analyzer_config,
            clf_config,
            general_config,
            english_config,
            formula_config,
            device,
        )

        self.analyzer = LayoutAnalyzer(**analyzer_config)

        _clf_config = deepcopy(clf_config)
        _clf_config.pop('model_dir')
        _clf_config.pop('model_fp')
        self.image_clf = ImageClassifier(**_clf_config)

        self.general_ocr = CnOcr(**general_config)
        self.english_ocr = CnOcr(**english_config)
        self.latex_model = LatexOCR(formula_config)

        self._assert_and_prepare_clf_model(clf_config)

    def _prepare_configs(
        self,
        analyzer_config,
        clf_config,
        general_config,
        english_config,
        formula_config,
        device,
    ):
        def _to_default(_conf, _def_val):
            if not _conf:
                _conf = _def_val
            return _conf

        analyzer_config = _to_default(analyzer_config, DEFAULT_CONFIGS['analyzer'])
        analyzer_config['device'] = device
        clf_config = _to_default(clf_config, DEFAULT_CONFIGS['clf'])
        general_config = _to_default(general_config, DEFAULT_CONFIGS['general'])
        general_config['context'] = device
        english_config = _to_default(english_config, DEFAULT_CONFIGS['english'])
        english_config['context'] = device
        formula_config = _to_default(formula_config, DEFAULT_CONFIGS['formula'])
        formula_config['device'] = device
        return (
            analyzer_config,
            clf_config,
            general_config,
            english_config,
            formula_config,
        )

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
            url = format_hf_hub_url(CLF_MODEL_URL_FMT % clf_config['base_model_name'])
            get_model_file(url, model_dir)  # download the .zip file and unzip
            fps = glob(os.path.join(model_dir, model_file_prefix) + '*.ckpt')

        model_fp = fps[0]
        self.image_clf.load(model_fp, self.device)

    @classmethod
    def from_config(cls, total_configs: Optional[dict] = None, device: str = 'cpu'):
        total_configs = total_configs or DEFAULT_CONFIGS
        return cls(
            analyzer_config=total_configs.get('analyzer', dict()),
            clf_config=total_configs.get('clf', dict()),
            general_config=total_configs.get('general', dict()),
            english_config=total_configs.get('english', dict()),
            formula_config=total_configs.get('formula', dict()),
            thresholds=total_configs.get('thresholds', DEFAULT_CONFIGS['thresholds']),
            device=device,
        )

    def __call__(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        return self.recognize(img, **kwargs)

    def recognize(
        self, img: Union[str, Path, Image.Image], use_analyzer: bool = True, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        对图片先做版面分析，然后再识别每块中包含的信息。在版面分析未识别出内容时，则把整个图片作为整体进行识别。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            use_analyzer (bool): whether to use the analyzer (MFD or Layout) to analyze the image; Default: `True`
            kwargs ():
                * resized_shape (int): 把图片宽度resize到此大小再进行处理；默认值为 `700`
                * save_analysis_res (str): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储
                * embed_sep (tuple): embedding latex的前后缀；只针对使用 `MFD` 时才有效；默认值为 `(' $', '$ ')`
                * isolated_sep (tuple): isolated latex的前后缀；只针对使用 `MFD` 时才有效；默认值为 `('$$\n', '\n$$')`

        Returns: a list of dicts, with keys:
           `type`: 图像类别；
           `text`: 识别出的文字或Latex公式
           `postion`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]

        """
        out = None
        if use_analyzer:
            if self.analyzer._model_name == 'mfd':
                out = self.recognize_by_mfd(img, **kwargs)
            else:
                out = self.recognize_by_layout(img, **kwargs)
        if not out:
            out = self.recognize_by_clf(img, **kwargs)
        return out

    def recognize_by_clf(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        把整张图片作为一整块进行识别。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`

        Returns: a list of dicts, with keys:
           `type`: 图像类别；
           `text`: 识别出的文字或Latex公式
           `position`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]

        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        width, height = img0.size
        _img = torch.tensor(np.asarray(img0))
        res = self.image_clf.predict_images([_img])[0]
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

        box = xyxy24p([0, 0, width, height], np.array)

        if kwargs.get('save_analysis_res'):
            out = [{'type': image_type, 'score': res[1], 'position': box}]
            save_layout_img(img0, IMAGE_TYPES, out, kwargs.get('save_analysis_res'))

        return [{'type': image_type, 'text': result, 'position': box}]

    def recognize_by_mfd(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        对图片先做MFD 或 版面分析，然后再识别每块中包含的信息。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): 把图片宽度resize到此大小再进行处理；默认值为 `608`
                * save_analysis_res (str): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储
                * embed_sep (tuple): embedding latex的前后缀；默认值为 `(' $', '$ ')`
                * isolated_sep (tuple): isolated latex的前后缀；默认值为 `('$$\n', '\n$$')`

        Returns: a list of ordered (top to bottom, left to right) dicts,
            with each dict representing one detected box, containing keys:
           `type`: 图像类别；Optional: 'text', 'isolated', 'embedding'
           `text`: 识别出的文字或Latex公式
           `position`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]
           `line_number`: box 所在行号（第一行 `line_number==0`），值相同的box表示它们在同一行

        """
        # 对于大图片，把图片宽度resize到此大小；宽度比此小的图片，其实不会放大到此大小，
        # 具体参考：cnstd.yolov7.layout_analyzer.LayoutAnalyzer._preprocess_images 中的 `letterbox` 行
        resized_shape = kwargs.get('resized_shape', 608)
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        w, h = img0.size
        ratio = resized_shape / w
        resized_shape = (int(h * ratio), resized_shape)  # (H, W)
        analyzer_outs = self.analyzer(img0.copy(), resized_shape=resized_shape)
        logger.debug('MFD Result: %s', analyzer_outs)
        embed_sep = kwargs.get('embed_sep', (' $', '$ '))
        isolated_sep = kwargs.get('isolated_sep', ('$$\n', '\n$$'))

        mf_out = []
        for box_info in analyzer_outs:
            box = box_info['box']
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop_patch = img0.crop((xmin, ymin, xmax, ymax))
            patch_out = self._latex(crop_patch)
            sep = isolated_sep if box_info['type'] == 'isolated' else embed_sep
            text = sep[0] + patch_out + sep[1]
            mf_out.append({'type': box_info['type'], 'text': text, 'position': box})

        img = np.array(img0.copy())
        # 把公式部分mask掉，然后对其他部分进行OCR
        for box_info in analyzer_outs:
            if box_info['type'] in ('isolated', 'embedding'):
                box = box_info['box']
                xmin, ymin = max(0, int(box[0][0]) - 1), max(0, int(box[0][1]) - 1)
                xmax, ymax = (
                    min(img0.size[0], int(box[2][0]) + 1),
                    min(img0.size[1], int(box[2][1]) + 1),
                )
                img[ymin:ymax, xmin:xmax, :] = 255

        box_infos = self.general_ocr.det_model.detect(img)

        def _to_iou_box(ori):
            return torch.tensor([ori[0][0], ori[0][1], ori[2][0], ori[2][1]]).unsqueeze(
                0
            )

        total_text_boxes = []
        for crop_img_info in box_infos['detected_texts']:
            # crop_img_info['box'] 可能是一个带角度的矩形框，需要转换成水平的矩形框
            hor_box = rotated_box_to_horizontal(crop_img_info['box'])
            if not is_valid_box(hor_box, min_height=8, min_width=2):
                continue
            line_box = _to_iou_box(hor_box)
            embed_mfs = []
            for box_info in mf_out:
                if box_info['type'] == 'embedding':
                    box2 = _to_iou_box(box_info['position'])
                    if float(box_partial_overlap(line_box, box2).squeeze()) > 0.7:
                        embed_mfs.append(
                            {
                                'position': box2[0].int().tolist(),
                                'text': box_info['text'],
                                'type': box_info['type'],
                            }
                        )

            ocr_boxes = self._split_line_image(line_box, embed_mfs)
            total_text_boxes.extend(ocr_boxes)

        outs = copy(mf_out)
        for box in total_text_boxes:
            crop_patch = torch.tensor(np.asarray(img0.crop(box['position'])))
            part_res = self._ocr_for_single_line(crop_patch, 'general')
            if part_res['text']:
                box['position'] = list2box(*box['position'])
                box['text'] = part_res['text']
                outs.append(box)

        outs = sort_boxes(outs, key='position')
        logger.debug(outs)
        outs = self._post_process(outs)

        outs = list(chain(*outs))
        if kwargs.get('save_analysis_res'):
            save_layout_img(
                img0,
                ('text', 'isolated', 'embedding'),
                outs,
                kwargs.get('save_analysis_res'),
            )

        return outs

    @classmethod
    def _post_process(cls, outs):
        for line_boxes in outs:
            if (
                len(line_boxes) > 1
                and line_boxes[-1]['type'] == 'text'
                and line_boxes[-2]['type'] != 'text'
            ):
                if line_boxes[-1]['text'].lower() == 'o':
                    line_boxes[-1]['text'] = '。'
        return outs

    @classmethod
    def _split_line_image(cls, line_box, embed_mfs):
        # 利用embedding formula所在位置，把单行文字图片切割成多个小段，之后这些小段会分别扔进OCR进行文字识别
        line_box = line_box[0]
        if not embed_mfs:
            return [{'position': line_box.int().tolist(), 'type': 'text'}]
        embed_mfs.sort(key=lambda x: x['position'][0])

        outs = []
        start = int(line_box[0])
        xmax, ymin, ymax = int(line_box[2]), int(line_box[1]), int(line_box[-1])
        for mf in embed_mfs:
            _xmax = min(xmax, int(mf['position'][0]) + 1)
            if start + 8 < _xmax:
                outs.append({'position': [start, ymin, _xmax, ymax], 'type': 'text'})
            start = int(mf['position'][2])
        if start < xmax:
            outs.append({'position': [start, ymin, xmax, ymax], 'type': 'text'})
        return outs

    def recognize_by_layout(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        对图片先做版面分析，然后再识别每块中包含的信息。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): 把图片宽度resize到此大小再进行处理；默认值为 `700`
                * save_analysis_res (str): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储

        Returns: a list of dicts, with keys:
           `type`: 图像类别；
           `text`: 识别出的文字或Latex公式
           `position`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]

        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        resized_shape = kwargs.get('resized_shape', 500)
        layout_out = self.analyzer(img0.copy(), resized_shape=resized_shape)
        logger.debug('Layout Analysis Result: %s', layout_out)

        out = []
        for box_info in layout_out:
            box = box_info['box']
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop_patch = img0.crop((xmin, ymin, xmax, ymax))
            if box_info['type'] == 'Equation':
                image_type = 'formula'
                patch_out = self._latex(crop_patch)
            else:
                crop_patch = torch.tensor(np.asarray(crop_patch))
                res = self.image_clf.predict_images([crop_patch])[0]
                image_type = res[0]
                if res[0] == 'formula':
                    image_type = 'general'
                elif (
                    res[1] < self.thresholds['english2general'] and res[0] == 'english'
                ):
                    image_type = 'general'
                patch_out = self._ocr(crop_patch, image_type)
            out.append({'type': image_type, 'text': patch_out, 'position': box})

        if kwargs.get('save_analysis_res'):
            save_layout_img(
                img0,
                CATEGORY_DICT['layout'],
                layout_out,
                kwargs.get('save_analysis_res'),
                key='box',
            )

        return out

    def _ocr_for_single_line(self, image, image_type):
        ocr_model = self.english_ocr if image_type == 'english' else self.general_ocr
        try:
            return ocr_model.ocr_for_single_line(image)
        except:
            return {'text': '', 'score': 0.0}

    def _ocr(self, image, image_type):
        ocr_model = self.english_ocr if image_type == 'english' else self.general_ocr
        result = ocr_model.ocr(image)
        texts = [_one['text'] for _one in result]
        result = '\n'.join(texts)
        return result

    def _latex(self, image):
        if isinstance(image, (str, Path)):
            image = read_img(image, return_type='Image')
        out = self.latex_model(image)
        return str(out)


if __name__ == '__main__':
    from .utils import set_logger

    logger = set_logger(log_level='DEBUG')

    p2t = Pix2Text()
    img = 'docs/examples/english.jpg'
    img = read_img(img, return_type='Image')
    out = p2t.recognize(img)
    logger.info(out)
