# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
import logging
import io
import os
from copy import deepcopy
from functools import cmp_to_key
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Literal

import numpy as np
from PIL import Image
import fitz  # PyMuPDF

from .utils import (
    select_device,
    box2list,
    read_img,
    add_img_margin,
    get_background_color,
    x_overlap,
    list2box,
    merge_line_texts,
)
from .layout_parser import LayoutParser, ElementType
from .doc_xl_layout import DocXLayoutParser
from .text_formula_ocr import TextFormulaOCR
from .table_ocr import TableOCR
from .page_elements import Element, Page, Document

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
        """
        Initialize the Pix2Text object.
        Args:
            layout_parser (LayoutParser): The layout parser object; default value is `None`, which means to create a default one
            text_formula_ocr (TextFormulaOCR): The text and formula OCR object; default value is `None`, which means to create a default one
            table_ocr (TableOCR): The table OCR object; default value is `None`, which means not to recognize tables
            **kwargs (dict): Other arguments, currently not used
        """
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
        Create a Pix2Text object from the configuration.
        Args:
            total_configs (dict): The total configuration; default value is `None`, which means to use the default configuration.
                If not None, it should contain the following keys:

                    * `layout`: The layout parser configuration
                    * `text_formula`: The TextFormulaOCR configuration
                    * `table`: The table OCR configuration
            enable_formula (bool): Whether to enable formula recognition; default value is `True`
            enable_table (bool): Whether to enable table recognition; default value is `True`
            device (str): The device to run the model; optional values are 'cpu', 'gpu' or 'cuda';
                default value is `None`, which means to select the device automatically
            **kwargs (dict): Other arguments

        Returns: a Pix2Text object

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
                text_formula_ocr.text_ocr,
                text_formula_ocr.spellchecker,
                table_config,
                device=device,
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

    def recognize(
        self,
        img: Union[str, Path, Image.Image],
        file_type: Literal[
            'pdf', 'page', 'text_formula', 'formula', 'text'
        ] = 'text_formula',
        **kwargs,
    ) -> Union[Document, Page, str, List[str], List[Any], List[List[Any]]]:
        """
        Recognize the content of the image or pdf file according to the specified type.
        It will call the corresponding recognition function `.recognize_{img_type}()` according to the `img_type`.
        Args:
            img (Union[str, Path, Image.Image]): The image/pdf file path or `Image.Image` object
            file_type (str):  Supported file types: 'pdf', 'page', 'text_formula', 'formula', 'text'
            **kwargs (dict): Arguments for the corresponding recognition function

        Returns: recognized results

        """
        rec_func = getattr(self, f'recognize_{file_type}', None)
        if rec_func is None:
            raise ValueError(f'Unsupported file type: {file_type}')
        return rec_func(img, **kwargs)

    def recognize_pdf(
        self,
        pdf_fp: Union[str, Path],
        pdf_number: int = 0,
        pdf_id: Optional[str] = None,
        page_numbers: Optional[List[int]] = None,
        **kwargs,
    ) -> Document:
        """
        recognize a pdf file
        Args:
            pdf_fp (Union[str, Path]): pdf file path
            pdf_number (int): pdf number
            pdf_id (str): pdf id
            page_numbers (List[int]): page numbers to recognize; default is `None`, which means to recognize all pages
            kwargs (dict): Optional keyword arguments. The same as `recognize_page`

        Returns: a Document object. Use `doc.to_markdown('output-dir')` to get the markdown output of the recognized document.

        """
        pdf_id = pdf_id or str(pdf_number)

        doc = fitz.open(pdf_fp, filetype='pdf')
        if page_numbers is None:
            page_numbers = list(range(len(doc)))
        outs = []
        for page_num in range(len(doc)):
            if page_num not in page_numbers:
                continue
            page = doc.load_page(page_num)
            # convert to image
            pix = page.get_pixmap(dpi=300)
            # convert the pixmap to bytes
            img_data = pix.tobytes(output='jpg', jpg_quality=200)
            # Create a PIL Image from the raw image data
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            page_id = str(page_num)
            page_kwargs = deepcopy(kwargs)
            if kwargs.get('save_debug_res'):
                page_kwargs['save_debug_res'] = os.path.join(
                    kwargs['save_debug_res'], f'{pdf_id}-{page_id}'
                )
            outs.append(
                self.recognize_page(
                    image, page_number=page_num, page_id=page_id, **page_kwargs
                )
            )
        return Document(
            number=pdf_number,
            id=pdf_id,
            pages=outs,
            spellchecker=self.text_formula_ocr.spellchecker,
            config=kwargs,
        )

    def recognize_page(
        self,
        img: Union[str, Path, Image.Image],
        page_number: int = 0,
        page_id: Optional[str] = None,
        **kwargs,
    ) -> Page:
        """
        Analyze the layout of the image, and then recognize the information contained in each section.

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            page_number (str): page number; default value is `0`
            page_id (str): page id; default value is `None`, which means to use the `str(page_number)`
            kwargs ():
                * resized_shape (int): Resize the image width to this size for processing; default value is `768`
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
                * table_as_image (bool): If `True`, the table will be recognized as an image (don't parse the table content as text) ; default value is `False`
                * title_contain_formula (bool): If `True`, the title of the page will be recognized as a mixed image (text and formula). If `False`, it will be recognized as a text; default value is `False`
                * text_contain_formula (bool): If `True`, the text of the page will be recognized as a mixed image (text and formula). If `False`, it will be recognized as a text; default value is `True`
                * formula_rec_kwargs (dict): generation arguments passed to formula recognizer `latex_ocr`; default value is `{}`
                * save_debug_res (str): if `save_debug_res` is set, the directory to save the debug results; default value is `None`, which means not to save

        Returns: a Page object. Use `page.to_markdown('output-dir')` to get the markdown output of the recognized page.
        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')

        page_id = page_id or str(page_number)
        kwargs['embed_sep'] = kwargs.get('embed_sep', (' $', '$ '))
        kwargs['isolated_sep'] = kwargs.get('isolated_sep', ('$$\n', '\n$$'))
        kwargs['line_sep'] = kwargs.get('line_sep', '\n')
        kwargs['auto_line_break'] = kwargs.get('auto_line_break', True)
        kwargs['title_contain_formula'] = kwargs.get('title_contain_formula', False)
        kwargs['text_contain_formula'] = kwargs.get('text_contain_formula', True)
        resized_shape = kwargs.get('resized_shape', 768)
        kwargs['resized_shape'] = resized_shape
        layout_kwargs = deepcopy(kwargs)
        layout_kwargs['resized_shape'] = resized_shape
        layout_kwargs['table_as_image'] = kwargs.get('table_as_image', False)
        layout_out, column_meta = self.layout_parser.parse(
            img0.copy(), **layout_kwargs,
        )

        debug_dir = None
        if kwargs.get('save_debug_res', None):
            debug_dir = Path(kwargs.get('save_debug_res'))
            debug_dir.mkdir(exist_ok=True, parents=True)

        outs = []
        for _id, box_info in enumerate(layout_out):
            image_type = box_info['type']
            if image_type == ElementType.IGNORED:
                continue
            box = box2list(box_info['position'])
            crop_patch = img0.crop(box)
            crop_width, _ = crop_patch.size
            score = 1.0
            if image_type in (ElementType.TEXT, ElementType.TITLE):
                _resized_shape = resized_shape
                while crop_width > 1.5 * _resized_shape and _resized_shape < 2048:
                    _resized_shape = min(int(1.5 * _resized_shape), 2048)
                padding_patch = add_img_margin(
                    crop_patch, left_right_margin=30, top_bottom_margin=30
                )
                text_formula_kwargs = deepcopy(kwargs)
                text_formula_kwargs['resized_shape'] = _resized_shape
                text_formula_kwargs['save_analysis_res'] = (
                    debug_dir / f'{_id}-{image_type.name}.png' if debug_dir else None
                )
                if image_type == ElementType.TITLE:
                    text_formula_kwargs['contain_formula'] = kwargs[
                        'title_contain_formula'
                    ]
                if image_type == ElementType.TEXT:
                    text_formula_kwargs['contain_formula'] = kwargs[
                        'text_contain_formula'
                    ]
                text_formula_kwargs['return_text'] = False
                _out = self.text_formula_ocr.recognize(
                    padding_patch, **text_formula_kwargs,
                )
                text, meta = None, _out
                score = float(np.mean([x['score'] for x in _out]))
            elif image_type == ElementType.TABLE:
                xmin, ymin, xmax, ymax = box
                img_width, img_height = img0.size
                table_expansion_margin = 10
                xmin, ymin = (
                    max(0, xmin - table_expansion_margin),
                    max(0, ymin - table_expansion_margin),
                )
                xmax, ymax = (
                    min(img_width, xmax + table_expansion_margin),
                    min(img_height, ymax + table_expansion_margin),
                )
                box = (xmin, ymin, xmax, ymax)
                crop_patch = img0.crop(box)
                save_analysis_res = (
                    debug_dir / f'{_id}-{image_type.name}.png' if debug_dir else None
                )
                table_kwargs = deepcopy(kwargs)
                table_kwargs['save_analysis_res'] = save_analysis_res
                _out = self.table_ocr.recognize(
                    crop_patch,
                    out_cells=True,
                    out_markdown=True,
                    out_html=True,
                    **table_kwargs,
                )
                text, meta = None, _out
            elif image_type == ElementType.FORMULA:
                formula_kwargs = deepcopy(kwargs)
                formula_kwargs['return_text'] = False
                _out = self.text_formula_ocr.recognize_formula(
                    crop_patch, **formula_kwargs
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
                    isolated=box_info['isolated'],
                    col_number=box_info['col_number'],
                    type=image_type,
                    score=score,
                    total_img=img0,
                    spellchecker=self.text_formula_ocr.spellchecker,
                    configs=kwargs,
                )
            )

        remaining_blocks = self._parse_remaining(
            img0, layout_out, column_meta, debug_dir, **kwargs
        )
        for box_info in remaining_blocks:
            outs.append(
                Element(
                    id=f'{page_id}-{len(outs)}-remaining',
                    box=box2list(box_info['position']),
                    meta=None,
                    text=box_info['text'],
                    isolated=False,
                    col_number=box_info['col_number'],
                    type=ElementType.TEXT
                    if box_info['type'] != 'isolated'
                    else ElementType.FORMULA,
                    score=box_info['score'],
                    total_img=img0,
                    spellchecker=self.text_formula_ocr.spellchecker,
                    configs=kwargs,
                )
            )
        return Page(
            number=page_number,
            id=page_id,
            elements=outs,
            spellchecker=self.text_formula_ocr.spellchecker,
            config=kwargs,
        )

    def _parse_remaining(self, img0, layout_out, column_meta, debug_dir, **kwargs):
        masked_img = np.array(img0.copy())
        bg_color = get_background_color(img0)
        # 把layout parser 已解析出的部分mask掉，然后对其他部分进行OCR
        for _box_info in layout_out:
            xmin, ymin, xmax, ymax = box2list(_box_info['position'])
            masked_img[ymin:ymax, xmin:xmax, :] = bg_color
        masked_img = Image.fromarray(masked_img)

        text_formula_kwargs = deepcopy(kwargs)
        text_formula_kwargs['return_text'] = False
        save_analysis_res = debug_dir / f'layout-remaining.png' if debug_dir else None
        text_formula_kwargs['save_analysis_res'] = save_analysis_res
        _out = self.text_formula_ocr.recognize(masked_img, **text_formula_kwargs,)
        min_text_length = kwargs.get('min_text_length', 4)
        _out = [_o for _o in _out if len(_o['text']) >= min_text_length]
        # guess which column the box belongs to
        for _box_info in _out:
            overlap_vals = []
            for col_number, col_info in column_meta.items():
                overlap_val = x_overlap(_box_info, col_info, key='position')
                overlap_vals.append([col_number, overlap_val])
            overlap_vals.sort(key=lambda x: (x[1], x[0]), reverse=True)
            match_col_number = overlap_vals[0][0]
            _box_info['col_number'] = match_col_number

        if len(_out) < 2:
            return _out

        def _compare(box_info1, box_info2):
            if box_info1['col_number'] != box_info2['col_number']:
                return box_info1['col_number'] < box_info2['col_number']
            else:
                return box_info1['position'][0, 1] < box_info2['position'][0, 1]

        _out = sorted(_out, key=cmp_to_key(_compare))

        begin_idx = 0
        end_idx = 1

        new_blocks = []
        while end_idx <= len(_out):
            while (
                end_idx < len(_out)
                and _out[end_idx]['col_number'] == _out[begin_idx]['col_number']
            ):
                end_idx += 1
            col_outs = _out[begin_idx:end_idx]
            begin_idx = end_idx
            end_idx += 1
            if len(col_outs) < 2:
                new_blocks.append(col_outs[0])
            else:
                new_blocks.extend(
                    _separate_blocks(
                        col_outs, self.text_formula_ocr.spellchecker, **kwargs
                    )
                )

        return new_blocks

    def recognize_text_formula(
        self, img: Union[str, Path, Image.Image], return_text: bool = True, **kwargs,
    ) -> Union[str, List[str], List[Any], List[List[Any]]]:
        """
        Analyze the layout of the image, and then recognize the information contained in each section.

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            return_text (bool): Whether to return the recognized text; default value is `True`
            kwargs ():
                * resized_shape (int): Resize the image width to this size for processing; default value is `768`
                * save_analysis_res (str): Save the mfd result image in this file; default is `None`, which means not to save
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
        return self.text_formula_ocr.recognize(img, return_text, **kwargs)

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


def _separate_blocks(col_outs, spellchecker, **kwargs):
    out_blocks = []

    def _merge_lines(cur_block_lines):
        if len(cur_block_lines) < 2:
            return cur_block_lines[0]
        ymin = cur_block_lines[0]['position'][0, 1]
        ymax = cur_block_lines[-1]['position'][3, 1]
        xmin = min([_b['position'][0, 0] for _b in cur_block_lines])
        xmax = max([_b['position'][3, 0] for _b in cur_block_lines])
        position = list2box(xmin, ymin, xmax, ymax)
        score = np.mean([_b['score'] for _b in cur_block_lines])
        col_number = cur_block_lines[0]['col_number']
        # text = smart_join([_b['text'] for _b in cur_block_lines], spellchecker)
        text = merge_line_texts(
            cur_block_lines,
            auto_line_break=kwargs['auto_line_break'],
            line_sep=kwargs['line_sep'],
            embed_sep=kwargs['embed_sep'],
            isolated_sep=kwargs['isolated_sep'],
            spellchecker=spellchecker,
        )

        return {
            'type': 'text',
            'text': text,
            'position': position,
            'score': score,
            'col_number': col_number,
            'line_number': len(out_blocks),
        }

    cur_block_lines = [col_outs[0]]
    for _box_info in col_outs[1:]:
        cur_height = (
            cur_block_lines[-1]['position'][3, 1]
            - cur_block_lines[-1]['position'][0, 1]
        )
        if (
            _box_info['position'][0, 1] - cur_block_lines[-1]['position'][3, 1]
            < cur_height
        ):
            # 当前行与下一行的间距少于一行的行高，则认为它们在相同的block
            cur_block_lines.append(_box_info)
        else:
            # merge lines
            merged_line = _merge_lines(cur_block_lines)
            out_blocks.append(merged_line)

            cur_block_lines = [_box_info]

    if len(cur_block_lines) > 0:
        merged_line = _merge_lines(cur_block_lines)
        out_blocks.append(merged_line)

    return out_blocks


if __name__ == '__main__':
    from .utils import set_logger

    logger = set_logger(log_level='DEBUG')

    p2t = Pix2Text()
    img = 'docs/examples/english.jpg'
    img = read_img(img, return_type='Image')
    out = p2t.recognize(img)
    logger.info(out)
