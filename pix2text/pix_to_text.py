# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
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
    select_device,
    prepare_imgs,
)
from .ocr_engine import prepare_ocr_engine

from .consts import MODEL_VERSION
from .latex_ocr import LatexOCR
from .utils import (
    read_img,
    save_layout_img,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIGS = {
    'analyzer': {'model_name': 'mfd'},
    'text': {},
    'formula': {},
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
        device: str = None,
        **kwargs,
    ):
        """
        Args:
            languages (str or Sequence[str]): The language code(s) of the text to be recognized; defaults to `('en', 'ch_sim')`.
            analyzer_config (dict): Configuration information for the Analyzer model; defaults to `None`, which means using the default configuration.
            text_config (dict): Configuration information for the Text OCR model; defaults to `None`, which means using the default configuration.
            formula_config (dict): Configuration information for Math Formula OCR model; defaults to `None`, which means using the default configuration.
            device (str, optional): What device to use for computation, supports `['cpu', 'cuda', 'gpu', 'mps']`; defaults to None, which selects the device automatically.
            **kwargs (): Reserved for other parameters; not currently used.
        """
        self.device = select_device(device)

        analyzer_config, text_config, formula_config = self._prepare_configs(
            analyzer_config, text_config, formula_config, self.device,
        )

        self.analyzer = LayoutAnalyzer(**analyzer_config)

        self.text_ocr = prepare_ocr_engine(languages, text_config)
        self.latex_model = LatexOCR(**formula_config)

    def _prepare_configs(
        self, analyzer_config, text_config, formula_config, device,
    ):
        def _to_default(_conf, _def_val):
            if not _conf:
                _conf = _def_val
            return _conf

        analyzer_config = _to_default(analyzer_config, DEFAULT_CONFIGS['analyzer'])
        # FIXME
        analyzer_config['device'] = device if device != 'mps' else 'cpu'
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
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Analyze the layout of the image, and then recognize the information contained in each section.

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): Resize the image width to this size for processing; default value is `608`
                * save_analysis_res (str): Save the analysis result image in this file; default is `None`, which means not to save
                * embed_sep (tuple): Prefix and suffix for embedding latex; only effective when using `MFD`; default is `(' $', '$ ')`
                * isolated_sep (tuple): Prefix and suffix for isolated latex; only effective when using `MFD`; default is `('$$\n', '\n$$')`
                * det_bbox_max_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height
                * mfr_batch_size (int): batch size for MFR; default value is `1`

        Returns: a list of dicts, with keys:
           `type`: The category of the image
           `text`: The recognized text or Latex formula
           `position`: Position information of the block, `np.ndarray`, with a shape of [4, 2]

        """
        if self.analyzer._model_name == 'mfd':
            out = self.recognize_by_mfd(img, **kwargs)
        else:
            out = self.recognize_by_layout(img, **kwargs)
        return out

    def recognize_by_mfd(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform Mathematical Formula Detection (MFD) on the image, and then recognize the information contained in each section.

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): Resize the image width to this size for processing; default value is `608`
                * save_analysis_res (str): Save the parsed result image in this file; default value is `None`, which means not to save
                * embed_sep (tuple): Prefix and suffix for embedding latex; default value is `(' $', '$ ')`
                * isolated_sep (tuple): Prefix and suffix for isolated latex; default value is `('$$\n', '\n$$')`
                * det_bbox_max_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.2`
                * mfr_batch_size (int): batch size for MFR; default value is `1`

        Returns: a list of ordered (top to bottom, left to right) dicts,
            with each dict representing one detected box, containing keys:
           `type`: The category of the image; Optional: 'text', 'isolated', 'embedding'
           `text`: The recognized text or Latex formula
           `position`: Position information of the block, `np.ndarray`, with shape of [4, 2]
           `line_number`: The line number of the box (first line `line_number==0`), boxes with the same value indicate they are on the same line

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

        crop_patches = []
        for box_info in analyzer_outs:
            box = box_info['box']
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop_patch = img0.crop((xmin, ymin, xmax, ymax))
            crop_patches.append(crop_patch)

        mfr_batch_size = kwargs.get('mfr_batch_size', 1)
        mf_results = self.latex_model.recognize(crop_patches, batch_size=mfr_batch_size)
        assert len(mf_results) == len(analyzer_outs)

        mf_outs = []
        for box_info, patch_out in zip(analyzer_outs, mf_results):
            sep = isolated_sep if box_info['type'] == 'isolated' else embed_sep
            text = sep[0] + patch_out + sep[1]
            mf_outs.append(
                {'type': box_info['type'], 'text': text, 'position': box_info['box']}
            )

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
            for box_info in mf_outs:
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

        outs = copy(mf_outs)
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
        Perform Layout Analysis (LA) on the image, then recognize the information contained in each section.

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): Resize the image width to this size for processing; default value is `500`
                * save_analysis_res (str): Save the parsed result image in this file; default value is `None`, which means not to save

        Returns: a list of dicts, with keys:
           `type`: The category of the image;
           `text`: The recognized text or Latex formula
           `position`: Position information of the block, `np.ndarray`, with a shape of [4, 2]

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
                patch_out = self.recognize_formula(crop_patch)
            else:
                patch_out = self.recognize_text(crop_patch)
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

    def recognize_text(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Recognize a pure Text Image.
        Args:
            imgs (Union[str, Path, Image.Image], List[str], List[Path], List[Image.Image]): The image or list of images
            kwargs (): Other parameters for `text_ocr.ocr()`

        Returns:
        Returns: Text str or list of text strs

        """
        is_single_image = False
        if isinstance(imgs, (str, Path, Image.Image)):
            imgs = [imgs]
            is_single_image = True

        input_imgs = prepare_imgs(imgs)

        outs = []
        for image in input_imgs:
            result = self.text_ocr.ocr(np.array(image), **kwargs)
            texts = [_one['text'] for _one in result]
            result = '\n'.join(texts)
            outs.append(result)

        if is_single_image:
            return outs[0]
        return outs

    def recognize_formula(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        batch_size: int = 1,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Recognize pure Math Formula images to LaTeX Expressions
        Args:
            imgs (Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]): The image or list of images
            batch_size (int): The batch size
            **kwargs (): Special model parameters. Not used for now

        Returns: The LaTeX Expression or list of LaTeX Expressions

        """
        return self.latex_model.recognize(imgs, batch_size=batch_size, **kwargs)


if __name__ == '__main__':
    from .utils import set_logger

    logger = set_logger(log_level='DEBUG')

    p2t = Pix2Text()
    img = 'docs/examples/english.jpg'
    img = read_img(img, return_type='Image')
    out = p2t.recognize(img)
    logger.info(out)
