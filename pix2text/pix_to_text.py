# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
from PIL import Image

from .utils import (
    select_device,
    box2list,
    read_img, add_img_margin,
)
from .layout_parser import LayoutParser, ElementType
from .doc_xl_layout import DocXLayoutParser
from .text_formula_ocr import TextFormulaOCR
from .table_ocr import TableOCR
from .page_elements import Element, Page

logger = logging.getLogger(__name__)


class Pix2Text(object):
    # MODEL_FILE_PREFIX = 'pix2text-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        *,
        layout_parser: Optional[LayoutParser] = None,
        text_formula_ocr: Optional[TextFormulaOCR] = None,
        table_ocr: Optional[TableOCR] = None,
        **kwargs,
    ):
        if layout_parser is None:
            device = select_device(None)
            # layout_parser = LayoutParser.from_config(None, device=device)
            layout_parser = DocXLayoutParser.from_config(None, device=device)
        if text_formula_ocr is None:
            device = select_device(None)
            text_formula_ocr = TextFormulaOCR.from_config(
                None, enable_formula=True, device=device
            )
        self.layout_parser = layout_parser
        self.text_formula_ocr = text_formula_ocr
        self.table_ocr = table_ocr

    @classmethod
    def from_config(
        cls,
        total_configs: Optional[dict] = None,
        enable_formula: bool = True,
        enable_table: bool = True,
        device: str = None,
        **kwargs,
    ):
        """

        Args:
            total_configs (dict):
            enable_formula ():
            enable_table ():
            device ():
            **kwargs ():

        Returns:

        """
        total_configs = total_configs or {}
        layout_config = total_configs.get('layout', None)
        text_formula_config = total_configs.get('text_formula', None)
        table_config = total_configs.get('table', None)

        # layout_parser = LayoutParser.from_config(layout_config, device=device)
        layout_parser = DocXLayoutParser.from_config(layout_config, device=device)
        text_formula_ocr = TextFormulaOCR.from_config(
            text_formula_config, enable_formula=enable_formula, device=device
        )
        if enable_table:
            table_ocr = TableOCR.from_config(
                text_formula_ocr.text_ocr, text_formula_ocr.spellchecker, table_config, device=device
            )
        else:
            table_ocr = None

        return cls(
            layout_parser=layout_parser,
            text_formula_ocr=text_formula_ocr,
            table_ocr=table_ocr,
            **kwargs,
        )

    def __call__(self, img: Union[str, Path, Image.Image], **kwargs) -> Page:
        return self.recognize_page(img, page_id='0', **kwargs)

    def recognize_page(
        self,
        img: Union[str, Path, Image.Image],
        page_id: str,
        # return_text: bool = True,
        **kwargs,
    ) -> Page:
        """
        Analyze the layout of the image, and then recognize the information contained in each section.

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            page_id (str): page id
            kwargs ():
                * resized_shape (int): Resize the image width to this size for processing; default value is `768`
                * save_layout_res (str): Save the layout result image in this file; default is `None`, which means not to save
                * mfr_batch_size (int): batch size for MFR; When running on GPU, this value is suggested to be set to greater than 1; default value is `1`
                * embed_sep (tuple): Prefix and suffix for embedding latex; only effective when `return_text` is `True`; default value is `(' $', '$ ')`
                * isolated_sep (tuple): Prefix and suffix for isolated latex; only effective when `return_text` is `True`; default value is `('$$\n', '\n$$')`
                * line_sep (str): The separator between lines of text; only effective when `return_text` is `True`; default value is `'\n'`
                * auto_line_break (bool): Automatically line break the recognized text; only effective when `return_text` is `True`; default value is `True`
                * det_text_bbox_max_width_expand_ratio (float): Expand the width of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.3`
                * det_text_bbox_max_height_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.2`
                * embed_ratio_threshold (float): The overlap threshold for embed formulas and text lines; default value is `0.6`.
                    When the overlap between an embed formula and a text line is greater than or equal to this threshold,
                    the embed formula and the text line are considered to be on the same line;
                    otherwise, they are considered to be on different lines.
                * table_as_image (bool): If `True`, the table will be recognized as an image; default value is `False`
                * formula_rec_kwargs (dict): generation arguments passed to formula recognizer `latex_ocr`; default value is `{}`

        Returns: a str when `return_text` is `True`; or a list of ordered (top to bottom, left to right) dicts when `return_text` is `False`,
            with each dict representing one detected box, containing keys:
               * `type`: The category of the image; Optional: 'text', 'isolated', 'embedding'
               * `text`: The recognized text or Latex formula
               * `score`: The confidence score [0, 1]; the higher, the more confident
               * `position`: Position information of the block, `np.ndarray`, with shape of [4, 2]
               * `line_number`: The line number of the box (first line `line_number==0`), boxes with the same value indicate they are on the same line

        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')

        resized_shape = kwargs.get('resized_shape', 768)
        table_as_image = kwargs.get('table_as_image', False)
        layout_out = self.layout_parser.parse(
            img0.copy(),
            resized_shape=resized_shape,
            table_as_image=table_as_image,
            **kwargs,
        )

        outs = []
        for _id, box_info in enumerate(layout_out):
            box = box2list(box_info['position'])
            crop_patch = img0.crop(box)
            crop_width, crop_height = crop_patch.size
            image_type = box_info['type']
            score = 1.0
            if image_type in (ElementType.TEXT, ElementType.TITLE):
                _resized_shape = resized_shape
                while crop_width > 1.5 * _resized_shape and _resized_shape < 2048:
                    _resized_shape = min(int(1.5 * _resized_shape), 2048)
                padding_patch = add_img_margin(crop_patch, left_right_margin=20, top_bottom_margin=20)
                _out = self.text_formula_ocr.recognize(
                    padding_patch, return_text=False, resized_shape=_resized_shape, save_analysis_res=f'{_id}-{image_type.name}.png', **kwargs
                )
                text, meta = None, _out
                score = float(np.mean([x['score'] for x in _out]))
            elif image_type == ElementType.TABLE:
                xmin, ymin, xmax, ymax = box
                img_width, img_height = img0.size
                table_expansion_margin = 10
                xmin, ymin = max(0, xmin - table_expansion_margin), max(0, ymin - table_expansion_margin)
                xmax, ymax = min(img_width, xmax + table_expansion_margin), min(img_height, ymax + table_expansion_margin)
                box = (xmin, ymin, xmax, ymax)
                crop_patch = img0.crop(box)
                _out = self.table_ocr.recognize(
                    crop_patch,
                    out_cells=True,
                    out_markdown=True,
                    out_html=True,
                    **kwargs,
                )
                text, meta = None, _out
            elif image_type == ElementType.FORMULA:
                _out = self.text_formula_ocr.recognize_formula(
                    crop_patch, return_text=False, **kwargs
                )
                score = _out['score']
                text, meta = None, _out
            elif image_type == ElementType.FIGURE:
                text, meta = '', None
            else:
                image_type = ElementType.UNKNOWN
                text, meta = '', None

            outs.append(
                Element(
                    id=f'{page_id}-{_id}',
                    box=box,
                    meta=meta,
                    text=text,
                    type=image_type,
                    score=score,
                    total_img=img0,
                    spellchecker=self.text_formula_ocr.spellchecker,
                    configs=kwargs,
                )
            )

        return Page(id=page_id, elements=outs, config=kwargs)

    def recognize_text_formula(
            self,
            imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
            return_text: bool = True,
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
        return self.text_formula_ocr.recognize(
            imgs, return_text, **kwargs
        )

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
        return self.text_formula_ocr.recognize_text(
            imgs, return_text, rec_config, **kwargs
        )

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
        return self.text_formula_ocr.recognize_formula(
            imgs, batch_size, return_text, rec_config, **kwargs
        )


if __name__ == '__main__':
    from .utils import set_logger

    logger = set_logger(log_level='DEBUG')

    p2t = Pix2Text()
    img = 'docs/examples/english.jpg'
    img = read_img(img, return_type='Image')
    out = p2t.recognize(img)
    logger.info(out)
