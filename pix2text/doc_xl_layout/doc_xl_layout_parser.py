# coding: utf-8
# Adapted from https://github.com/AlibabaResearch/AdvancedLiterateMachinery
import json
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import logging
from typing import Union, List, Dict, Any, Optional

import numpy as np
from PIL import Image

from .opts import opts
from .huntie_subfield import Huntie_Subfield
from .detectors.detector_factory import detector_factory
from .wrapper import wrap_result
from ..consts import MODEL_VERSION
from ..layout_parser import LayoutParser, ElementType
from ..utils import (
    select_device,
    read_img,
    data_dir,
    save_layout_img,
    clipbox,
    overlap,
    box2list,
    x_overlap,
    merge_boxes,
)

logger = logging.getLogger(__name__)

CATEGORIES = {
    "title": 0,
    "figure": 1,
    "plain text": 2,
    "header": 3,
    "page number": 4,
    "footnote": 5,
    "footer": 6,
    "table": 7,
    "table caption": 8,
    "figure caption": 9,
    "equation": 10,
    "full column": 11,
    "sub column": 12,
}
CATEGORY_MAPPING = [''] * len(CATEGORIES)
for cate, idx in CATEGORIES.items():
    CATEGORY_MAPPING[idx] = cate


class DocXLayoutOutput:
    def __init__(self, layout_detection_info, subfield_detection_info, message=''):
        self.layout_detection_info = layout_detection_info
        self.subfield_detection_info = subfield_detection_info
        self.message = message

    def to_json(self):
        return wrap_result(
            self.layout_detection_info, self.subfield_detection_info, CATEGORY_MAPPING
        )


class DocXLayoutParser(LayoutParser):
    ignored_types = {'footnote', 'footer', 'page number'}
    type_mappings = {
        'title': ElementType.TITLE,
        'figure': ElementType.FIGURE,
        'plain text': ElementType.TEXT,
        'header': ElementType.TEXT,
        'table': ElementType.TABLE,
        'table caption': ElementType.TEXT,
        'figure caption': ElementType.TEXT,
        'equation': ElementType.FORMULA,
    }
    # types that are isolated and usually don't cross different columns. They should not be merged with other elements
    is_isolated = {'header', 'table caption', 'figure caption', 'equation'}

    def __init__(
        self,
        device: str = None,
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        **kwargs,
    ):
        if model_fp is None:
            model_fp = self._prepare_model_files(root, None)
        new_params = {
            'task': 'ctdet_subfield',
            'arch': 'dlav0subfield_34',
            'input_res': 768,
            'num_classes': 13,
            'scores_thresh': kwargs.get('scores_thresh', 0.35),
            'load_model': str(model_fp),
            'debug': kwargs.get('debug', 0),
        }

        opt = opts().parse(new_params)
        opt = opts().update_dataset_info_and_set_heads(opt, Huntie_Subfield)
        opt.device = select_device(device)

        Detector = detector_factory[opt.task]
        detector = Detector(opt)
        self.detector = detector
        self.opt = opt
        logger.debug("DocXLayoutParser parameters %s", self.opt)

    @classmethod
    def from_config(cls, configs: Optional[dict] = None, device: str = None, **kwargs):
        configs = configs or {}
        device = select_device(device)
        # configs['device'] = device if device != 'mps' else 'cpu'

        return cls(
            device=device,
            model_fp=configs.get('model_fp', None),
            root=configs.get('root', data_dir()),
            **configs,
        )

    def _prepare_model_files(self, root, model_info):
        model_root_dir = Path(root) / MODEL_VERSION
        model_dir = model_root_dir / 'layout-parser'
        model_fp = model_dir / 'DocXLayout_231012.pth'
        if model_fp.exists():
            return model_fp
        if model_dir.exists():
            shutil.rmtree(str(model_dir))
        model_dir.mkdir(parents=True)
        download_cmd = f'huggingface-cli download --repo-type model --resume-download --local-dir-use-symlinks False breezedeus/pix2text-layout --local-dir {model_dir}'
        os.system(download_cmd)
        if not model_fp.exists():  # download failed above
            if model_dir.exists():
                shutil.rmtree(str(model_dir))
            os.system('HF_ENDPOINT=https://hf-mirror.com ' + download_cmd)
        return model_fp

    def convert_eval_format(self, all_bboxes, opt):
        layout_detection_items = []
        subfield_detection_items = []
        for cls_ind in all_bboxes:
            for box in all_bboxes[cls_ind]:
                if box[8] < opt.scores_thresh:
                    continue
                pts = np.round(box).tolist()[:8]
                score = box[8]
                category_id = box[9]
                # direction_id = box[10]
                # secondary_id = box[11]
                detection = {
                    "category_id": int(category_id),
                    # "secondary_id": int(secondary_id),
                    # "direction_id": int(direction_id),
                    "poly": pts,
                    "score": float("{:.2f}".format(score)),
                }
                if cls_ind in (12, 13):
                    subfield_detection_items.append(detection)
                else:
                    layout_detection_items.append(detection)
        return layout_detection_items, subfield_detection_items

    def parse(
        self,
        img: Union[str, Path, Image.Image],
        table_as_image: bool = False,
        **kwargs,
    ) -> (List[Dict[str, Any]], Dict[str, Any]):
        """

        Args:
            img ():
            table_as_image ():
            **kwargs ():
              * save_debug_res (str): if `save_debug_res` is set, the directory to save the debug results; default value is `None`, which means not to save
              * expansion_margin (int): expansion margin

        Returns:

        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        img_width, img_height = img0.size
        try:
            # to np.array, RGB -> BGR
            ret = self.detector.run(np.array(img0)[:, :, ::-1])
            layout_detection_info, subfield_detection_info = self.convert_eval_format(
                ret['results'], self.opt
            )
            out = DocXLayoutOutput(
                layout_detection_info, subfield_detection_info, message='success'
            )
        except Exception as e:
            logger.warning("DocXLayoutPredictor Error %s", repr(e))
            out = DocXLayoutOutput([], [], message=repr(e))

        layout_out = out.to_json()
        debug_dir = None
        if kwargs.get('save_debug_res', None):
            debug_dir = Path(kwargs.get('save_debug_res'))
            debug_dir.mkdir(exist_ok=True, parents=True)
        if debug_dir is not None:
            with open(debug_dir / 'layout_out.json', 'w', encoding='utf-8') as f:
                json.dump(
                    layout_out, f, indent=2, ensure_ascii=False,
                )
        if layout_out:
            layout_out = self._preprocess_outputs(img0, layout_out)
            layout_out, column_meta = self._format_outputs(
                img0, layout_out, table_as_image
            )
        else:
            layout_out, column_meta = [], {}

        layout_out = self._merge_overlapped_boxes(layout_out)

        expansion_margin = kwargs.get('expansion_margin', 8)
        layout_out = self._expand_boxes(
            layout_out, expansion_margin, height=img_height, width=img_width
        )

        save_layout_fp = kwargs.get(
            'save_layout_res',
            debug_dir / 'layout_res.jpg' if debug_dir is not None else None,
        )
        if save_layout_fp:
            element_type_list = [t for t in ElementType]
            save_layout_img(
                img0,
                element_type_list,
                layout_out,
                save_path=save_layout_fp,
                key='position',
            )

        return layout_out, column_meta

    def _preprocess_outputs(self, img0, outs):
        width, height = img0.size

        subfields = outs['subfields']
        for column_info in subfields:
            layout_out = column_info['layouts']
            if len(layout_out) < 2:
                continue
            for idx, cur_box_info in enumerate(layout_out[:-1]):
                next_box_info = layout_out[idx + 1]
                cur_box_ymax = cur_box_info['pts'][-1]
                next_box_ymin = next_box_info['pts'][1]
                if (
                    cur_box_info['category'] == 'figure'
                    and next_box_info['category'] == 'figure caption'
                    and -6 < next_box_ymin - cur_box_ymax < 80
                ):
                    new_xmin = min(cur_box_info['pts'][0], next_box_info['pts'][0])
                    # new_xmin = max(new_xmin, 0, col_pts[0])
                    new_xmax = max(cur_box_info['pts'][2], next_box_info['pts'][2])
                    # new_xmax = min(new_xmax, )
                    new_ymin = max(0, cur_box_info['pts'][1])
                    new_ymax = max(cur_box_ymax, next_box_ymin - 16)
                    new_box = [
                        new_xmin,
                        new_ymin,
                        new_xmax,
                        new_ymin,
                        new_xmax,
                        new_ymax,
                        new_xmin,
                        new_ymax,
                    ]
                    layout_out[idx]['pts'] = new_box
                # FIXME: first figure caption, then figure

        return outs

    def _format_outputs(self, img0, out, table_as_image: bool):
        width, height = img0.size

        column_meta = defaultdict(dict)
        final_out = []
        subfields = out['subfields']
        col_number = 0
        for column_info in subfields:
            if column_info['category'] == 'sub column':
                cur_col_number = col_number
                col_number += 1
            elif column_info['category'] == 'full column':  # == 'full column'
                cur_col_number = -1
            else:  # '其他'
                cur_col_number = -2
            box = clipbox(np.array(column_info['pts']).reshape(4, 2), height, width)
            column_meta[cur_col_number]['position'] = box
            column_meta[cur_col_number]['score'] = column_info['confidence']
            layout_out = column_info['layouts']
            for box_info in layout_out:
                image_type = box_info['category']
                isolated = image_type in self.is_isolated
                if image_type in self.ignored_types:
                    image_type = ElementType.IGNORED
                else:
                    image_type = self.type_mappings.get(image_type, ElementType.UNKNOWN)
                if table_as_image and image_type == ElementType.TABLE:
                    image_type = ElementType.FIGURE
                box = clipbox(np.array(box_info['pts']).reshape(4, 2), height, width)
                final_out.append(
                    {
                        'type': image_type,
                        'position': box,
                        'score': box_info['confidence'],
                        'col_number': cur_col_number,
                        'isolated': isolated,
                    }
                )

        if -2 in column_meta and -1 in column_meta:
            filtered_out = []
            full_column_box = column_meta[-1]['position']
            full_column_xmin, full_column_xmax = (
                full_column_box[0, 0],
                full_column_box[1, 0],
            )
            for box_info in final_out:
                if box_info['col_number'] != -2:
                    filtered_out.append(box_info)
                    continue
                cur_box = box_info['position']
                cur_box_xmin, cur_box_xmax = cur_box[0, 0], cur_box[1, 0]
                cur_box_ymin, cur_box_ymax = cur_box[0, 1], cur_box[2, 1]
                if (
                    box_info['type'] == ElementType.TEXT
                    and (
                        cur_box_xmax < full_column_xmin
                        or cur_box_xmin > full_column_xmax
                    )
                    and cur_box_ymax - cur_box_ymin > 5 * (cur_box_xmax - cur_box_xmin)
                ):  # unnecessary block
                    box_info['type'] = ElementType.IGNORED
                filtered_out.append(box_info)

            final_out = filtered_out

        # handle abnormal elements (col_number == -2)
        if -2 in column_meta:
            column_meta.pop(-2)
        # guess which column the box belongs to
        for _box_info in final_out:
            if _box_info['col_number'] != -2:
                continue
            overlap_vals = []
            for col_number, col_info in column_meta.items():
                overlap_val = x_overlap(_box_info, col_info, key='position')
                overlap_vals.append([col_number, overlap_val])
            if overlap_vals:
                overlap_vals.sort(key=lambda x: (x[1], x[0]), reverse=True)
                match_col_number = overlap_vals[0][0]
                _box_info['col_number'] = match_col_number
            else:
                _box_info['col_number'] = 0

        return final_out, column_meta

    def _merge_overlapped_boxes(self, layout_out):
        """
        Detected bounding boxes may overlap; merge these overlapping boxes into a single one.
        """
        if len(layout_out) < 2:
            return layout_out
        layout_out = deepcopy(layout_out)

        def _overlay_vertically(box1, box2):
            if x_overlap(box1, box2, key=None) < 0.8:
                return False
            box1 = box2list(box1)
            box2 = box2list(box2)
            # 判断是否有交集
            if box1[3] <= box2[1] or box2[3] <= box1[1]:
                return False
            # 计算交集的高度
            y_min = max(box1[1], box2[1])
            y_max = min(box1[3], box2[3])
            return y_max - y_min > 10

        for anchor_idx, anchor_box_info in enumerate(layout_out):
            if anchor_box_info['type'] != ElementType.TEXT or anchor_box_info.get(
                'used', False
            ):
                continue
            for cand_idx, cand_box_info in enumerate(layout_out):
                if anchor_idx == cand_idx:
                    continue
                if cand_box_info['type'] != ElementType.TEXT or cand_box_info.get(
                    'used', False
                ):
                    continue
                if not _overlay_vertically(
                    anchor_box_info['position'], cand_box_info['position']
                ):
                    continue
                anchor_box_info['position'] = merge_boxes(
                    anchor_box_info['position'], cand_box_info['position']
                )
                cand_box_info['used'] = True

        return [box_info for box_info in layout_out if not box_info.get('used', False)]

    def _expand_boxes(self, layout_out, expansion_margin, height, width):
        """
        Expand boxes with some margin to get better results
        Args:
            layout_out (): layout_out
            expansion_margin (int): expansion margin
            height (int): height of the image
            width (int): width of the image

        Returns: layout_out with expanded boxes

        """

        def _overlap_with_some_box(idx, anchor_box):
            # anchor_box = layout_out[idx]
            return any(
                [
                    overlap(anchor_box, box_info['position'], key=None) > 0
                    for idx2, box_info in enumerate(layout_out)
                    if idx2 != idx
                ]
            )

        for idx, box_info in enumerate(layout_out):
            if box_info['type'] not in (
                ElementType.TEXT,
                ElementType.TITLE,
                ElementType.FORMULA,
            ):
                continue
            if _overlap_with_some_box(idx, box_info['position']):
                continue

            # expand xmin and xmax
            new_box = box_info['position'].copy()
            xmin, xmax = new_box[0, 0], new_box[1, 0]
            xmin -= expansion_margin
            xmax += expansion_margin
            if xmin <= 8:
                xmin = 0
            if xmax + 8 >= width:
                xmax = width
            new_box[0, 0] = new_box[3, 0] = xmin
            new_box = clipbox(new_box, height, width)
            if not _overlap_with_some_box(idx, new_box):
                layout_out[idx]['position'] = new_box
            new_box = layout_out[idx]['position'].copy()
            new_box[1, 0] = new_box[2, 0] = xmax
            new_box = clipbox(new_box, height, width)
            if not _overlap_with_some_box(idx, new_box):
                layout_out[idx]['position'] = new_box

            # expand ymin and ymax
            new_box = layout_out[idx]['position'].copy()
            ymin, ymax = new_box[0, 1], new_box[2, 1]
            ymin -= expansion_margin
            ymax += expansion_margin
            if ymin <= 8:
                ymin = 0
            if ymax + 8 >= height:
                ymax = height
            new_box[0, 1] = new_box[1, 1] = ymin
            new_box = clipbox(new_box, height, width)
            if not _overlap_with_some_box(idx, new_box):
                layout_out[idx]['position'] = new_box
            new_box = layout_out[idx]['position'].copy()
            new_box[2, 1] = new_box[3, 1] = ymax
            new_box = clipbox(new_box, height, width)
            if not _overlap_with_some_box(idx, new_box):
                layout_out[idx]['position'] = new_box

        return layout_out
