# coding: utf-8
# Adapted from https://github.com/AlibabaResearch/AdvancedLiterateMachinery
import json
import os
import shutil
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
from ..utils import select_device, read_img, data_dir, save_layout_img, clipbox, overlap

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
            'scores_thresh': kwargs.get('structure_thresholds', 0.45),
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
        logger.info("DocXLayoutParser parameters %s", self.opt)

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
        # resized_shape: int = 608,
        table_as_image: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
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
        json.dump(layout_out, open('layout_out.json', 'w'), indent=2, ensure_ascii=False)
        if layout_out:
            layout_out = self._format_outputs(img0, layout_out, table_as_image)
        else:
            layout_out = []

        expansion_margin = kwargs.get('expansion_margin', 10)
        layout_out = self._expand_boxes(
            layout_out, expansion_margin, height=img_height, width=img_width
        )

        if kwargs.get('save_layout_res'):
            element_type_list = [t for t in ElementType]
            save_layout_img(
                img0,
                element_type_list,
                layout_out,
                kwargs.get('save_layout_res'),
                key='position',
            )

        return layout_out

    def _format_outputs(self, img0, out, table_as_image: bool):
        layout_out = out['layouts']
        width, height = img0.size

        final_out = []
        for box_info in layout_out:
            image_type = box_info['category']
            if image_type in self.ignored_types:
                continue
            image_type = self.type_mappings.get(image_type, ElementType.UNKNOWN)
            if table_as_image and image_type == ElementType.TABLE:
                image_type = ElementType.FIGURE
            # if image_type == ElementType.TITLE:
            #     breakpoint()
            box = clipbox(np.array(box_info['pts']).reshape(4, 2), height, width)
            final_out.append(
                {'type': image_type, 'position': box, 'score': box_info['confidence'],}
            )
        return final_out

    def _expand_boxes(self, layout_out, expansion_margin, height, width):
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
            if box_info['type'] not in (ElementType.TEXT, ElementType.TITLE, ElementType.FORMULA):
                continue
            if _overlap_with_some_box(idx, box_info['position']):
                continue

            # expand xmin and xmax
            new_box = box_info['position'].copy()
            new_box[0, 0] -= expansion_margin
            new_box[3, 0] -= expansion_margin
            new_box[1, 0] += expansion_margin
            new_box[2, 0] += expansion_margin
            new_box = clipbox(new_box, height, width)
            if not _overlap_with_some_box(idx, new_box):
                layout_out[idx]['position'] = new_box

            # expand ymin and ymax
            new_box = layout_out[idx]['position'].copy()
            new_box[0, 1] -= expansion_margin
            new_box[1, 1] -= expansion_margin
            new_box[2, 1] += expansion_margin
            new_box[3, 1] += expansion_margin
            new_box = clipbox(new_box, height, width)
            if not _overlap_with_some_box(idx, new_box):
                layout_out[idx]['position'] = new_box

        return layout_out
