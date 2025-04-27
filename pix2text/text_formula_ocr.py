# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import logging
import re
from itertools import chain
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from copy import copy

from PIL import Image
import numpy as np
import torch
from cnstd.utils import box_partial_overlap
from spellchecker import SpellChecker

from .utils import (
    custom_deepcopy,
    sort_boxes,
    merge_adjacent_bboxes,
    adjust_line_height,
    adjust_line_width,
    rotated_box_to_horizontal,
    is_valid_box,
    list2box,
    select_device,
    prepare_imgs,
    merge_line_texts,
    remove_overlap_text_bbox,
    y_overlap,
)
from .ocr_engine import prepare_ocr_engine, TextOcrEngine
from .formula_detector import MathFormulaDetector
from .latex_ocr import LatexOCR
from .utils import (
    read_img,
    save_layout_img,
)

logger = logging.getLogger(__name__)


DEFAULT_CONFIGS = {
    'mfd': {},
    'text': {},
    'formula': {},
}
# see: https://pypi.org/project/pyspellchecker
CHECKER_SUPPORTED_LANGUAGES = {
    'en',
    'es',
    'fr',
    'pt',
    'de',
    'it',
    'ru',
    'ar',
    'eu',
    'lv',
    'nl',
}


class TextFormulaOCR(object):
    def __init__(
        self,
        *,
        text_ocr: Optional[TextOcrEngine] = None,
        mfd: Optional[Any] = None,
        latex_ocr: Optional[LatexOCR] = None,
        spellchecker: Optional[SpellChecker] = None,
        enable_formula: bool = True,
        **kwargs,
    ):
        """
        Recognize text and formula from an image.
        Args:
            text_ocr (Optional[TextOcrEngine]): Text OCR engine; defaults to `None`.
            mfd (Optional[Any]): Math Formula Detector; defaults to `None`.
            latex_ocr (Optional[LatexOCR]): Latex OCR engine; defaults to `None`.
            spellchecker (Optional[SpellChecker]): Spell Checker; defaults to `None`.
            enable_formula (bool): Whether to enable the capability of Math Formula Detection (MFD) and Recognition (MFR); defaults to `True`.
            **kwargs ():
        """
        if text_ocr is None:
            text_config = custom_deepcopy(DEFAULT_CONFIGS['text'])
            device = select_device(device=None)
            text_config['context'] = device
            logger.warning(
                f'text_ocr must not be None. Using default text_ocr engine instead, with config: {text_config}.'
            )
            text_ocr = prepare_ocr_engine(
                languages=('en', 'ch_sim'), ocr_engine_config=text_config
            )
        # if mfd is None or latex_ocr is None:
        #     default_ocr = TextFormulaOCR.from_config()
        #     mfd = default_ocr.mfd if mfd is None else mfd
        #     text_ocr = default_ocr.text_ocr if text_ocr is None else text_ocr
        #     latex_ocr = default_ocr.latex_ocr if latex_ocr is None else latex_ocr
        #     del default_ocr

        self.text_ocr = text_ocr
        self.mfd = mfd
        self.latex_ocr = latex_ocr
        self.spellchecker = spellchecker
        self.enable_formula = enable_formula

    @classmethod
    def from_config(
        cls,
        total_configs: Optional[dict] = None,
        enable_formula: bool = True,
        enable_spell_checker: bool = True,
        device: str = None,
        **kwargs,
    ):
        """
        Args:
            total_configs (dict): Configuration information for Pix2Text; defaults to `None`, which means using the default configuration. Usually the following keys are used:

                * languages (str or Sequence[str]): The language code(s) of the text to be recognized; defaults to `('en', 'ch_sim')`.
                * mfd (dict): Configuration information for the Analyzer model; defaults to `None`, which means using the default configuration.
                * text (dict): Configuration information for the Text OCR model; defaults to `None`, which means using the default configuration.
                * formula (dict): Configuration information for Math Formula OCR model; defaults to `None`, which means using the default configuration.
            enable_formula (bool): Whether to enable the capability of Math Formula Detection (MFD) and Recognition (MFR); defaults to True.
            enable_spell_checker (bool): Whether to enable the capability of Spell Checker; defaults to True.
            device (str, optional): What device to use for computation, supports `['cpu', 'cuda', 'gpu', 'mps']`; defaults to None, which selects the device automatically.
            **kwargs (): Reserved for other parameters; not currently used.
        """
        total_configs = total_configs or DEFAULT_CONFIGS
        languages = total_configs.get('languages', ('en', 'ch_sim'))
        text_config = total_configs.get('text', dict())
        mfd_config = total_configs.get('mfd', dict())
        formula_config = total_configs.get('formula', dict())

        device = select_device(device)
        mfd_config, text_config, formula_config = cls.prepare_configs(
            mfd_config, text_config, formula_config, device,
        )

        text_ocr = prepare_ocr_engine(languages, text_config)

        if enable_formula:
            mfd = MathFormulaDetector(**mfd_config)
            latex_ocr = LatexOCR(**formula_config)
        else:
            mfd = None
            latex_ocr = None

        spellchecker = None
        if enable_spell_checker:
            checker_languages = set(languages) & CHECKER_SUPPORTED_LANGUAGES
            if checker_languages:
                spellchecker = SpellChecker(language=checker_languages)

        return cls(
            text_ocr=text_ocr,
            mfd=mfd,
            latex_ocr=latex_ocr,
            spellchecker=spellchecker,
            enable_formula=enable_formula,
            **kwargs,
        )

    @classmethod
    def prepare_configs(
        cls, mfd_config, text_config, formula_config, device,
    ):
        def _to_default(_conf, _def_val):
            if not _conf:
                _conf = custom_deepcopy(_def_val)
            return custom_deepcopy(_conf)

        mfd_config = _to_default(mfd_config, DEFAULT_CONFIGS['mfd'])
        mfd_config['device'] = device
        text_config = _to_default(text_config, DEFAULT_CONFIGS['text'])
        text_config['context'] = device
        formula_config = _to_default(formula_config, DEFAULT_CONFIGS['formula'])
        formula_config['device'] = device
        return (
            mfd_config,
            text_config,
            formula_config,
        )

    @property
    def languages(self):
        return self.text_ocr.languages

    def __call__(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        return self.recognize(img, **kwargs)

    def recognize(
        self, img: Union[str, Path, Image.Image], return_text: bool = True, **kwargs
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Perform Mathematical Formula Detection (MFD) on the image, and then recognize the information contained in each section.

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            return_text (bool): Whether to return only the recognized text; default value is `True`
            kwargs ():
                * contain_formula (bool): If `True`, the image will be recognized as a mixed image (text and formula). If `False`, it will be recognized as a text; default value is `True`
                * resized_shape (int): Resize the image width to this size for processing; default value is `768`
                * save_analysis_res (str): Save the parsed result image in this file; default value is `None`, which means not to save
                * mfr_batch_size (int): batch size for MFR; When running on GPU, this value is suggested to be set to greater than 1; default value is `1`
                * embed_sep (tuple): Prefix and suffix for embedding latex; only effective when `return_text` is `True`; default value is `(' $', '$ ')`
                * isolated_sep (tuple): Prefix and suffix for isolated latex; only effective when `return_text` is `True`; default value is two-dollar signs
                * line_sep (str): The separator between lines of text; only effective when `return_text` is `True`; default value is a line break
                * auto_line_break (bool): Automatically line break the recognized text; only effective when `return_text` is `True`; default value is `True`
                * det_text_bbox_max_width_expand_ratio (float): Expand the width of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.3`
                * det_text_bbox_max_height_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.2`
                * embed_ratio_threshold (float): The overlap threshold for embed formulas and text lines; default value is `0.6`.
                    When the overlap between an embed formula and a text line is greater than or equal to this threshold,
                    the embed formula and the text line are considered to be on the same line;
                    otherwise, they are considered to be on different lines.
                * formula_rec_kwargs (dict): generation arguments passed to formula recognizer `latex_ocr`; default value is `{}`

        Returns: a str when `return_text` is `True`, or a list of ordered (top to bottom, left to right) dicts when `return_text` is `False`,
            with each dict representing one detected box, containing keys:

                * `type`: The category of the image; Optional: 'text', 'isolated', 'embedding'
                * `text`: The recognized text or Latex formula
                * `score`: The confidence score [0, 1]; the higher, the more confident
                * `position`: Position information of the block, `np.ndarray`, with shape of [4, 2]
                * `line_number`: The line number of the box (first line `line_number==0`), boxes with the same value indicate they are on the same line

        """
        resized_shape = kwargs.get('resized_shape', 768)
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        w, h = img0.size
        ratio = resized_shape / w
        resized_shape = (int(h * ratio), resized_shape)  # (H, W)
        # logger.debug('MFD Result: %s', analyzer_outs)
        analyzer_outs = []
        crop_patches = []
        mf_results = []
        enable_formula = kwargs.get('contain_formula', True) and self.enable_formula
        if enable_formula and self.mfd is not None and self.latex_ocr is not None:
            analyzer_outs = self.mfd(img0.copy(), resized_shape=resized_shape)
            for mf_box_info in analyzer_outs:
                box = mf_box_info['box']
                xmin, ymin, xmax, ymax = (
                    int(box[0][0]),
                    int(box[0][1]),
                    int(box[2][0]),
                    int(box[2][1]),
                )
                crop_patch = img0.crop((xmin, ymin, xmax, ymax))
                crop_patches.append(crop_patch)

            mfr_batch_size = kwargs.get('mfr_batch_size', 1)
            formula_rec_kwargs = kwargs.get('formula_rec_kwargs', {})
            mf_results = self.latex_ocr.recognize(
                crop_patches, batch_size=mfr_batch_size, **formula_rec_kwargs
            )

        assert len(mf_results) == len(analyzer_outs)

        mf_outs = []
        for mf_box_info, patch_out in zip(analyzer_outs, mf_results):
            text = patch_out['text']
            mf_outs.append(
                {
                    'type': mf_box_info['type'],
                    'text': text,
                    'position': mf_box_info['box'],
                    'score': patch_out['score'],
                }
            )

        masked_img = np.array(img0.copy())
        # 把公式部分mask掉，然后对其他部分进行OCR
        for mf_box_info in analyzer_outs:
            if mf_box_info['type'] in ('isolated', 'embedding'):
                box = mf_box_info['box']
                xmin, ymin = max(0, int(box[0][0]) - 1), max(0, int(box[0][1]) - 1)
                xmax, ymax = (
                    min(img0.size[0], int(box[2][0]) + 1),
                    min(img0.size[1], int(box[2][1]) + 1),
                )
                masked_img[ymin:ymax, xmin:xmax, :] = 255
        masked_img = Image.fromarray(masked_img)

        text_box_infos = self.text_ocr.detect_only(
            np.array(img0), resized_shape=resized_shape
        )
        box_infos = []
        for line_box_info in text_box_infos['detected_texts']:
            # crop_img_info['box'] 可能是一个带角度的矩形框，需要转换成水平的矩形框
            _text_box = rotated_box_to_horizontal(line_box_info['position'])
            if not is_valid_box(_text_box, min_height=8, min_width=2):
                continue
            box_infos.append({'position': _text_box})
        max_width_expand_ratio = kwargs.get('det_text_bbox_max_width_expand_ratio', 0.3)
        if self.text_ocr.name == 'cnocr':
            box_infos: list[dict] = adjust_line_width(
                text_box_infos=box_infos,
                formula_box_infos=mf_outs,
                img_width=img0.size[0],
                max_expand_ratio=max_width_expand_ratio,
            )
        box_infos = remove_overlap_text_bbox(box_infos, mf_outs)

        def _to_iou_box(ori):
            return torch.tensor([ori[0][0], ori[0][1], ori[2][0], ori[2][1]]).unsqueeze(
                0
            )

        embed_ratio_threshold = kwargs.get('embed_ratio_threshold', 0.6)
        total_text_boxes = []
        for line_box_info in box_infos:
            _line_box = _to_iou_box(line_box_info['position'])
            _embed_mfs = []
            for mf_box_info in mf_outs:
                if mf_box_info['type'] == 'embedding':
                    _mf_box = _to_iou_box(mf_box_info['position'])
                    overlap_area_ratio = float(
                        box_partial_overlap(_line_box, _mf_box).squeeze()
                    )
                    if overlap_area_ratio >= embed_ratio_threshold or (
                        overlap_area_ratio > 0
                        and y_overlap(line_box_info, mf_box_info, key='position')
                        > embed_ratio_threshold
                    ):
                        _embed_mfs.append(
                            {
                                'position': _mf_box[0].int().tolist(),
                                'text': mf_box_info['text'],
                                'type': mf_box_info['type'],
                            }
                        )

            ocr_boxes = self._split_line_image(_line_box, _embed_mfs)
            total_text_boxes.extend(ocr_boxes)

        outs = copy(mf_outs)
        for box in total_text_boxes:
            box['position'] = list2box(*box['position'])
            outs.append(box)
        outs = sort_boxes(outs, key='position')
        outs = [merge_adjacent_bboxes(bboxes) for bboxes in outs]
        max_height_expand_ratio = kwargs.get(
            'det_text_bbox_max_height_expand_ratio', 0.2
        )
        outs = adjust_line_height(
            outs, img0.size[1], max_expand_ratio=max_height_expand_ratio
        )

        for line_idx, line_boxes in enumerate(outs):
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
                crop_patch = np.array(masked_img.crop((xmin, ymin, xmax, ymax)))
                part_res = self.text_ocr.recognize_only(crop_patch)
                box['text'] = part_res['text']
                box['score'] = part_res['score']
            outs[line_idx] = [box for box in line_boxes if box['text'].strip()]

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

        if return_text:
            embed_sep = kwargs.get('embed_sep', (' $', '$ '))
            isolated_sep = kwargs.get('isolated_sep', ('$$\n', '\n$$'))
            line_sep = kwargs.get('line_sep', '\n')
            auto_line_break = kwargs.get('auto_line_break', True)
            outs = merge_line_texts(
                outs,
                auto_line_break,
                line_sep,
                embed_sep,
                isolated_sep,
                self.spellchecker,
            )

        return outs

    def _post_process(self, outs):
        match_pairs = [
            (',', ',，'),
            ('.', '.。'),
            ('?', '?？'),
        ]
        formula_tag = '^[（\(]\d+(\.\d+)*[）\)]$'

        def _match(a1, a2):
            matched = False
            for b1, b2 in match_pairs:
                if a1 in b1 and a2 in b2:
                    matched = True
                    break
            return matched

        for idx, line_boxes in enumerate(outs):
            if (
                any([_lang in ('ch_sim', 'ch_tra') for _lang in self.languages])
                and len(line_boxes) > 1
                and line_boxes[-1]['type'] == 'text'
                and line_boxes[-2]['type'] != 'text'
            ):
                if line_boxes[-1]['text'].lower() == 'o':
                    line_boxes[-1]['text'] = '。'
            if len(line_boxes) > 1:
                # 去掉边界上多余的标点
                for _idx2, box in enumerate(line_boxes[1:]):
                    if (
                        box['type'] == 'text'
                        and line_boxes[_idx2]['type'] == 'embedding'
                    ):  # if the current box is text and the previous box is embedding
                        if _match(line_boxes[_idx2]['text'][-1], box['text'][0]) and (
                            not line_boxes[_idx2]['text'][:-1].endswith('\\')
                            and not line_boxes[_idx2]['text'][:-1].endswith(r'\end')
                        ):
                            line_boxes[_idx2]['text'] = line_boxes[_idx2]['text'][:-1]
                # 把 公式 tag 合并到公式里面去
                for _idx2, box in enumerate(line_boxes[1:]):
                    if (
                        box['type'] == 'text'
                        and line_boxes[_idx2]['type'] == 'isolated'
                    ):  # if the current box is text and the previous box is embedding
                        if y_overlap(line_boxes[_idx2], box, key='position') > 0.9:
                            if re.match(formula_tag, box['text']):
                                # 去掉开头和结尾的括号
                                tag_text = box['text'][1:-1]
                                line_boxes[_idx2]['text'] = line_boxes[_idx2][
                                    'text'
                                ] + ' \\tag{{{}}}'.format(tag_text)
                                new_xmax = max(
                                    line_boxes[_idx2]['position'][2][0],
                                    box['position'][2][0],
                                )
                                line_boxes[_idx2]['position'][1][0] = line_boxes[_idx2][
                                    'position'
                                ][2][0] = new_xmax
                                box['text'] = ''

            outs[idx] = [box for box in line_boxes if box['text'].strip()]
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
            if _xmax >= xmax:
                break
        if start < xmax:
            outs.append({'position': [start, ymin, xmax, ymax], 'type': 'text'})
        return outs

    def recognize_text(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        return_text: bool = True,
        rec_config: Optional[dict] = None,
        **kwargs,
    ) -> Union[str, List[str], List[Any], List[List[Any]]]:
        """
        Recognize a pure Text Image.
        Args:
            imgs (Union[str, Path, Image.Image], List[str], List[Path], List[Image.Image]): The image or list of images
            return_text (bool): Whether to return only the recognized text; default value is `True`
            rec_config (Optional[dict]): The config for recognition
            kwargs (): Other parameters for `text_ocr.ocr()`

        Returns: Text str or list of text strs when `return_text` is True;
                `List[Any]` or `List[List[Any]]` when `return_text` is False, with the same length as `imgs` and the following keys:

                    * `position`: Position information of the block, `np.ndarray`, with a shape of [4, 2]
                    * `text`: The recognized text
                    * `score`: The confidence score [0, 1]; the higher, the more confident

        """
        is_single_image = False
        if isinstance(imgs, (str, Path, Image.Image)):
            imgs = [imgs]
            is_single_image = True

        input_imgs = prepare_imgs(imgs)

        outs = []
        for image in input_imgs:
            result = self.text_ocr.ocr(np.array(image), rec_config=rec_config, **kwargs)
            if return_text:
                texts = [_one['text'] for _one in result]
                result = '\n'.join(texts)
            outs.append(result)

        if kwargs.get('save_analysis_res'):
            save_layout_img(
                input_imgs[0], ['text'], outs[0], kwargs.get('save_analysis_res'),
            )

        if is_single_image:
            return outs[0]
        return outs

    def recognize_formula(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        batch_size: int = 1,
        return_text: bool = True,
        rec_config: Optional[dict] = None,
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Recognize pure Math Formula images to LaTeX Expressions
        Args:
            imgs (Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]): The image or list of images
            batch_size (int): The batch size
            return_text (bool): Whether to return only the recognized text; default value is `True`
            rec_config (Optional[dict]): The config for recognition
            **kwargs (): Special model parameters. Not used for now

        Returns: The LaTeX Expression or list of LaTeX Expressions;
                str or List[str] when `return_text` is True;
                Dict[str, Any] or List[Dict[str, Any]] when `return_text` is False, with the following keys:

                    * `text`: The recognized LaTeX text
                    * `score`: The confidence score [0, 1]; the higher, the more confident

        """
        if not self.enable_formula:
            raise RuntimeError('Formula recognition is not enabled')
        if self.latex_ocr is None:
            raise RuntimeError('`latex_ocr` model MUST NOT be None')
        outs = self.latex_ocr.recognize(
            imgs, batch_size=batch_size, rec_config=rec_config, **kwargs
        )
        if return_text:
            if isinstance(outs, dict):
                outs = outs['text']
            elif isinstance(outs, list):
                outs = [one['text'] for one in outs]

        return outs


# 基于 Vlm 实现一个 TextFormulaOCR 的子类
class VlmTextFormulaOCR(TextFormulaOCR):
    def __init__(
        self,
        *,
        vlm: Optional[Any] = None,
        spellchecker: Optional[SpellChecker] = None,
        **kwargs,
    ):
        """
        Recognize text and formula from an image.
        Args:
            vlm (Optional[Any]): VLM model; defaults to `None`.
            spellchecker (Optional[SpellChecker]): Spell Checker; defaults to `None`.
            **kwargs (): not used for now.
        """
        if vlm is None:
            raise ValueError('vlm must not be None')
        self.vlm = vlm
        self.spellchecker = spellchecker

    @classmethod
    def from_config(
        cls,
        total_configs: Optional[dict] = None,
        enable_spell_checker: bool = True,
        **kwargs,
    ):
        """
        Args:
            total_configs (dict): Configuration information for VlmTextFormulaOCR; defaults to `None`, which means using the default configuration. Usually the following keys are used:
                * languages (str or Sequence[str]): The language code(s) of the text to be recognized; defaults to `('en', 'ch_sim')`.
            enable_spell_checker (bool): Whether to enable the capability of Spell Checker; defaults to True.
            **kwargs (): Reserved for other parameters: 
                * model_name (str): The name of the VLM model; defaults to `None`, which means using the default model.
                * api_key (str): The API key for the VLM model; defaults to `None`, which means using the default API key.
        """
        from .vlm_api import Vlm

        total_configs = total_configs or {}
        # Combine configs with any additional kwargs
        all_kwargs = kwargs.copy()
        if total_configs:
            all_kwargs.update(total_configs)
        
        vlm = Vlm(
            model_name=all_kwargs.pop("model_name", None),
            api_key=all_kwargs.pop("api_key", None),
        )

        spellchecker = None
        if enable_spell_checker:
            languages = total_configs.get('languages', ('en', 'ch_sim'))
            checker_languages = set(languages) & CHECKER_SUPPORTED_LANGUAGES
            if checker_languages:
                spellchecker = SpellChecker(language=checker_languages)

        return cls(
            vlm=vlm,
            spellchecker=spellchecker,
            **all_kwargs
        )

    def recognize(
        self, img: Union[str, Path, Image.Image], return_text: bool = True, **kwargs
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Perform Mathematical Formula Detection (MFD) on the image, and then recognize the information contained in each section.
        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            return_text (bool): Whether to return only the recognized text; default value is `True`
            kwargs (): Other parameters for `vlm.__call__()`,
                * `prompt`: The prompt for the VLM model

        Returns: a str when `return_text` is `True`, or a list of ordered (top to bottom, left to right) dicts when `return_text` is `False`,
            with each dict representing one detected box, containing keys:

                * `type`: The category of the image; Optional: 'text', 'isolated', 'embedding'
                * `text`: The recognized text or Latex formula
                * `score`: The confidence score [0, 1]; the higher, the more confident
                * `position`: Position information of the block, `np.ndarray`, with shape of [4, 2]
                * `line_number`: The line number of the box (first line `line_number==0`), boxes with the same value indicate they are on the same line

        """
        resized_shape = kwargs.get('resized_shape', 768)
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        w, h = img0.size
        result = self.vlm(img_path=img0, auto_resize=True, **kwargs)
        if return_text:
            return result["text"]

        result["type"] = "text"
        result["position"] = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        result["line_number"] = 0
        return [result]

    def recognize_text(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        return_text: bool = True,
        rec_config: Optional[dict] = None,
        **kwargs,
    ) -> Union[str, List[str], List[Any], List[List[Any]]]:
        return self._recognize_batch(imgs, res_type='text', return_text=return_text, rec_config=rec_config)

    def recognize_formula(
        self,
        imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
        batch_size: int = 1,
        return_text: bool = True,
        rec_config: Optional[dict] = None,
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Recognize pure Math Formula images to LaTeX Expressions
        Args:
            imgs (Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]): The image or list of images
            batch_size (int): The batch size. Useless here
            return_text (bool): Whether to return only the recognized text; default value is `True`
            rec_config (Optional[dict]): The config for recognition
            **kwargs (): Special model parameters. Not used for now

        Returns: The LaTeX Expression or list of LaTeX Expressions;
                str or List[str] when `return_text` is True;
                Dict[str, Any] or List[Dict[str, Any]] when `return_text` is False, with the following keys:

                    * `text`: The recognized LaTeX text
                    * `score`: The confidence score [0, 1]; the higher, the more confident

        """
        return self._recognize_batch(imgs, res_type='formula', return_text=return_text, rec_config=rec_config)

    def _recognize_batch(self, imgs, *, res_type, return_text = True, rec_config = None):
        rec_config = rec_config or {}
        if isinstance(imgs, (str, Path, Image.Image)):
            result = self.recognize(imgs, return_text, **rec_config)
            if not return_text:
                result = result[0]
            return result

        results = self.vlm(imgs, **rec_config)
        if return_text:
            results = [one['text'] for one in results]
        else:
            for img, result in zip(imgs, results):
                if isinstance(img, Image.Image):
                    w, h = img.size
                else:
                    with read_img(img, return_type='Image') as img0:
                        w, h = img0.size

                result["type"] = res_type 
                result["position"] = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
                result["line_number"] = 0
        return results
    

if __name__ == '__main__':
    from .utils import set_logger

    logger = set_logger(log_level='DEBUG')

    p2t = TextFormulaOCR()
    img = 'docs/examples/english.jpg'
    img = read_img(img, return_type='Image')
    out = p2t.recognize(img)
    logger.info(out)
