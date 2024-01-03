# coding: utf-8
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import logging
from itertools import chain
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Sequence
from copy import copy

from PIL import Image
import numpy as np
import torch
from cnstd import LayoutAnalyzer
from cnstd.yolov7.consts import CATEGORY_DICT
from cnstd.yolov7.general import box_partial_overlap

from .utils import (
    sort_boxes,
    merge_adjacent_bboxes,
    adjust_line_height,
    rotated_box_to_horizontal,
    is_valid_box,
    list2box,
)
from .ocr_engine import prepare_ocr_engine

from .consts import (
    LATEX_CONFIG_FP,
    MODEL_VERSION,
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
    'text': {},
    'formula': {
        'config': LATEX_CONFIG_FP,
        'checkpoint': Path(data_dir()) / 'formula' / 'weights.pth',
        'no_resize': False,
    },
}


class Pix2Text(object):
    MODEL_FILE_PREFIX = 'pix2text-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        *,
        languages: Union[str, Sequence[str]] = ('en', 'ch_sim'),
        analyzer_config: Dict[str, Any] = None,
        text_config: Dict[str, Any] = None,
        formula_config: Dict[str, Any] = None,
        device: str = 'cpu',  # ['cpu', 'cuda', 'gpu']
        **kwargs,
    ):
        """

        Args:
            analyzer_config (dict): Analyzer模型对应的配置信息；默认为 `None`，表示使用默认配置
            text_config (dict): 文本识别模型对应的配置信息；默认为 `None`，表示使用默认配置
            formula_config (dict): 公式识别模型对应的配置信息；默认为 `None`，表示使用默认配置
            device (str): 使用什么资源进行计算，支持 `['cpu', 'cuda', 'gpu']`；默认为 `cpu`
            **kwargs (): 预留的其他参数；目前未被使用
        """
        if device.lower() == 'gpu':
            device = 'cuda'
        self.device = device

        analyzer_config, text_config, formula_config = self._prepare_configs(
            analyzer_config, text_config, formula_config, device,
        )

        self.analyzer = LayoutAnalyzer(**analyzer_config)

        self.text_ocr = prepare_ocr_engine(languages, text_config)
        self.latex_model = LatexOCR(formula_config)

    def _prepare_configs(
        self,
        analyzer_config,
        text_config,
        formula_config,
        device,
    ):
        def _to_default(_conf, _def_val):
            if not _conf:
                _conf = _def_val
            return _conf

        analyzer_config = _to_default(analyzer_config, DEFAULT_CONFIGS['analyzer'])
        analyzer_config['device'] = device
        text_config = _to_default(text_config, DEFAULT_CONFIGS['text'])
        text_config['context'] = device
        formula_config = _to_default(formula_config, DEFAULT_CONFIGS['formula'])
        formula_config['device'] = device
        return (
            analyzer_config,
            text_config,
            formula_config,
        )

    @classmethod
    def from_config(cls, total_configs: Optional[dict] = None, device: str = 'cpu'):
        total_configs = total_configs or DEFAULT_CONFIGS
        return cls(
            analyzer_config=total_configs.get('analyzer', dict()),
            text_config=total_configs.get('text', dict()),
            formula_config=total_configs.get('formula', dict()),
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
                * det_bbox_max_expand_ratio (float): 对检测出的文本 bbox，扩展其高度。此值表示相对于原始 bbox 高度来说的上下最大扩展比率

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
        return out

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
                * det_bbox_max_expand_ratio (float): 对检测出的文本 bbox，扩展其高度。此值表示相对于原始 bbox 高度来说的上下最大扩展比率

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

        box_infos = self.text_ocr.detect_only(img)

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
            box['position'] = list2box(*box['position'])
            outs.append(box)
        outs = sort_boxes(outs, key='position')
        outs = [merge_adjacent_bboxes(bboxes) for bboxes in outs]
        max_expand_ratio = kwargs.get('det_bbox_max_expand_ratio', 0.2)
        outs = adjust_line_height(outs, img0.size[1], max_expand_ratio=max_expand_ratio)

        for line_boxes in outs:
            for box in line_boxes:
                if box['type'] != 'text':
                    continue
                bbox = box['position']
                xmin, ymin, xmax, ymax = (
                    int(bbox[0][0]),
                    int(bbox[0][1]),
                    int(bbox[2][0]),
                    int(bbox[2][1]),
                )
                crop_patch = np.array(img0.crop((xmin, ymin, xmax, ymax)))
                part_res = self.text_ocr.recognize_only(crop_patch)
                box['text'] = part_res['text']

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
            image_type = box_info['type']
            if image_type == 'Equation':
                patch_out = self._latex(crop_patch)
            else:
                patch_out = self._ocr(crop_patch)
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

    def _ocr(self, image):
        result = self.text_ocr.ocr(image)
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
