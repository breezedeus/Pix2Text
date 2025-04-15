# coding: utf-8
# use DocLayout-YOLO model for layout analysis: https://github.com/opendatalab/DocLayout-YOLO
import json
import os
import logging
import shutil
from collections import defaultdict
from copy import deepcopy, copy
from pathlib import Path
from typing import Union, Optional

from PIL import Image
import numpy as np
import torch
import torchvision

from .consts import MODEL_VERSION
from .layout_parser import ElementType
from .utils import (
    list2box,
    clipbox,
    box2list,
    read_img,
    save_layout_img,
    data_dir,
    select_device,
    y_overlap,
    prepare_model_files2,
)
from . import DocXLayoutParser

from doclayout_yolo import YOLOv10

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)


class DocYoloLayoutParser(object):
    ignored_types = {"abandon", "table_footnote"}
    # names: {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}
    type_mappings = {
        "title": ElementType.TITLE,
        "figure": ElementType.FIGURE,
        "plain text": ElementType.TEXT,
        "table": ElementType.TABLE,
        "table_caption": ElementType.TEXT,
        "figure_caption": ElementType.TEXT,
        "isolate_formula": ElementType.FORMULA,
        "inline formula": ElementType.FORMULA,
        "formula_caption": ElementType.PLAIN_TEXT,
        "ocr text": ElementType.TEXT,
    }
    # types that are isolated and usually don't cross different columns. They should not be merged with other elements
    is_isolated = {"table_caption", "figure_caption", "isolate_formula"}

    def __init__(
        self,
        device: str = None,
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        **kwargs,
    ):
        if model_fp is None:
            model_fp = self._prepare_model_files(root)
        device = select_device(device)
        # device = 'cpu' if device == 'mps' else device
        self.device = device
        self.mapping = {
            0: "title",
            1: "plain text",
            2: "abandon",
            3: "figure",
            4: "figure_caption",
            5: "table",
            6: "table_caption",
            7: "table_footnote",
            8: "isolate_formula",
            9: "formula_caption",
        }
        logger.info("Use DocLayout-YOLO model for Layout Analysis: {}".format(model_fp))
        self.predictor = YOLOv10(model_fp)

    def _prepare_model_files(self, root):
        model_root_dir = Path(root).expanduser() / MODEL_VERSION
        model_dir = model_root_dir / "layout-docyolo"
        model_fp = model_dir / "doclayout_yolo_docstructbench_imgsz1024.pt"
        if model_fp.exists():
            return model_fp
        model_fp = prepare_model_files2(
            model_fp_or_dir=model_fp,
            remote_repo="breezedeus/pix2text-layout-docyolo",
            file_or_dir="file",
        )
        return model_fp

    @classmethod
    def from_config(cls, configs: Optional[dict] = None, device: str = None, **kwargs):
        configs = copy(configs or {})
        device = select_device(device)
        model_fp = configs.pop("model_fp", None)
        root = configs.pop("root", data_dir())
        configs.pop("device", None)

        return cls(device=device, model_fp=model_fp, root=root, **configs)

    def parse(
        self,
        img: Union[str, Path, Image.Image],
        table_as_image: bool = False,
        *,
        imgsz: int = 1024,  # Prediction image size
        conf: float = 0.2,  # Confidence threshold
        iou_threshold: float = 0.45,  # NMS IoU threshold
        **kwargs,
    ):
        """

        Args:
            img ():
            table_as_image ():
            imgsz (int): Prediction image size
            conf (float): Confidence threshold
            iou_threshold (float): NMS IoU threshold
            **kwargs ():
              * save_debug_res (str): if `save_debug_res` is set, the directory to save the debug results; default value is `None`, which means not to save
              * expansion_margin (int): expansion margin

        Returns:

        """
        if isinstance(img, Image.Image):
            img0 = img.convert("RGB")
        else:
            img0 = read_img(img, return_type="Image")
        img_width, img_height = img0.size
        det_res = self.predictor.predict(
            img0,  # Image to predict
            imgsz=imgsz,  # Prediction image size
            conf=conf,  # Confidence threshold
        )[0]
        scores = det_res.__dict__["boxes"].conf
        boxes = det_res.__dict__["boxes"].xyxy
        _classes = det_res.__dict__["boxes"].cls

        indices = torchvision.ops.nms(
            boxes=torch.Tensor(boxes),
            scores=torch.Tensor(scores),
            iou_threshold=iou_threshold,
        )
        boxes, scores, _classes = boxes[indices], scores[indices], _classes[indices]
        # dtype to int
        _classes = _classes.int().tolist()

        page_layout_result = []
        for box, score, _cls in zip(boxes, scores, _classes):
            page_layout_result.append(
                {
                    "type": self.mapping[_cls],
                    "position": list2box(*box.tolist()),
                    "score": float(score),
                }
            )

        ignored_layout_result = [
            item for item in page_layout_result if item["type"] in self.ignored_types
        ]
        for x in ignored_layout_result:
            x["col_number"] = -1
        ignored_layout_out, _ = self._format_outputs(
            img_width, img_height, ignored_layout_result, table_as_image
        )
        if page_layout_result:
            # 目前 MFR 对带 tag 的公式识别效果不太好，所以暂时不合并
            # page_layout_result = self._merge_isolated_formula_and_caption(page_layout_result)

            # 去掉 ignored 类型
            _page_layout_result = [
                item
                for item in page_layout_result
                if item["type"] not in self.ignored_types
            ]
            layout_out = fetch_column_info(_page_layout_result, img_width, img_height)
            layout_out, column_meta = self._format_outputs(
                img_width, img_height, layout_out, table_as_image
            )
        else:
            layout_out, column_meta = [], {}

        debug_dir = None
        if kwargs.get("save_debug_res", None):
            debug_dir = Path(kwargs.get("save_debug_res"))
            debug_dir.mkdir(exist_ok=True, parents=True)
        if debug_dir is not None:
            with open(debug_dir / "layout_out.json", "w", encoding="utf-8") as f:
                json_out = deepcopy(layout_out)
                for item in json_out:
                    item["position"] = item["position"].tolist()
                    item["type"] = item["type"].name
                json.dump(
                    json_out,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        # layout_out = DocXLayoutParser._merge_overlapped_boxes(layout_out)

        expansion_margin = kwargs.get("expansion_margin", 8)
        layout_out = DocXLayoutParser._expand_boxes(
            layout_out, expansion_margin, height=img_height, width=img_width
        )

        save_layout_fp = kwargs.get(
            "save_layout_res",
            debug_dir / "layout_res.jpg" if debug_dir is not None else None,
        )

        layout_out.extend(ignored_layout_out)

        if save_layout_fp:
            element_type_list = [t for t in ElementType]
            save_layout_img(
                img0,
                element_type_list,
                layout_out,
                save_path=save_layout_fp,
                key="position",
            )

        return layout_out, column_meta

    def _merge_isolated_formula_and_caption(self, page_layout_result):
        # 合并孤立的公式和公式标题
        # 对于每个公式标题，找到与它在同一行且在其左侧距离最近的孤立公式，并把它们合并
        isolated_formula = [
            item for item in page_layout_result if item["type"] == "isolate_formula"
        ]
        formula_caption = [
            item for item in page_layout_result if item["type"] == "formula_caption"
        ]
        remaining_elements = [
            item
            for item in page_layout_result
            if item["type"] not in ["isolate_formula", "formula_caption"]
        ]
        for caption in formula_caption:
            caption_xmin, caption_ymin, caption_xmax, caption_ymax = box2list(
                caption["position"]
            )
            min_dist = float("inf")
            nearest_formula = None
            for formula in isolated_formula:
                formula_xmin, formula_ymin, formula_xmax, formula_ymax = box2list(
                    formula["position"]
                )
                if y_overlap(caption, formula, key="position") >= 0.7:
                    dist = caption_xmin - formula_xmax
                    if 0 <= dist < min_dist:
                        min_dist = dist
                        nearest_formula = formula
            if nearest_formula is not None:
                new_formula = deepcopy(nearest_formula)
                formula_xmin, formula_ymin, formula_xmax, formula_ymax = box2list(
                    new_formula["position"]
                )
                new_formula["position"] = list2box(
                    min(caption_xmin, formula_xmin),
                    min(caption_ymin, formula_ymin),
                    max(caption_xmax, formula_xmax),
                    max(caption_ymax, formula_ymax),
                )
                remaining_elements.append(new_formula)
                isolated_formula.remove(nearest_formula)
            else:  # not found
                remaining_elements.append(caption)
        return remaining_elements + isolated_formula

    def _format_outputs(self, width, height, layout_out, table_as_image: bool):
        # 获取每一列的信息
        column_numbers = set([item["col_number"] for item in layout_out])
        column_meta = defaultdict(dict)
        for col_idx in column_numbers:
            cur_col_res = [item for item in layout_out if item["col_number"] == col_idx]
            mean_score = np.mean([item["score"] for item in cur_col_res])
            xmin, ymin, xmax, ymax = box2list(cur_col_res[0]["position"])
            for item in cur_col_res[1:]:
                cur_xmin, cur_ymin, cur_xmax, cur_ymax = box2list(item["position"])
                xmin = min(xmin, cur_xmin)
                ymin = min(ymin, cur_ymin)
                xmax = max(xmax, cur_xmax)
                ymax = max(ymax, cur_ymax)
            column_meta[col_idx]["position"] = clipbox(
                list2box(xmin, ymin, xmax, ymax), height, width
            )
            column_meta[col_idx]["score"] = mean_score

        final_out = []
        for box_info in layout_out:
            image_type = box_info["type"]
            isolated = image_type in self.is_isolated
            if image_type in self.ignored_types:
                image_type = ElementType.IGNORED
            else:
                image_type = self.type_mappings.get(image_type, ElementType.UNKNOWN)
            if table_as_image and image_type == ElementType.TABLE:
                image_type = ElementType.FIGURE
            final_out.append(
                {
                    "type": image_type,
                    "position": clipbox(box_info["position"], height, width),
                    "score": box_info["score"],
                    "col_number": box_info["col_number"],
                    "isolated": isolated,
                }
            )

        return final_out, column_meta


def cal_column_width(layout_res, img_width, img_height):
    widths = [item["position"][1][0] - item["position"][0][0] for item in layout_res]
    if len(widths) <= 2:
        return min(widths + [img_width])

    # 计算所有box的宽度和相对面积
    boxes_info = []
    for item in layout_res:
        x0, y0 = item["position"][0]
        x1, y1 = item["position"][2]
        width = x1 - x0
        height = y1 - y0
        area = width * height
        boxes_info.append({"width": width, "area": area, "y0": y0, "height": height})

    # 按面积排序，获取最大的几个box
    boxes_info.sort(key=lambda x: x["area"], reverse=True)

    # 使用面积权重计算加权平均宽度
    total_weight = 0
    weighted_width_sum = 0

    # 只考虑面积最大的前30%的boxes
    top_boxes = boxes_info[: max(2, int(len(boxes_info) * 0.3))]

    for box in top_boxes:
        # 使用面积作为权重
        weight = box["area"]
        # 给予页面下半部分的box更高权重（因为通常是正文区域）
        if box["y0"] > img_height * 0.5:
            weight *= 1.5
        weighted_width_sum += box["width"] * weight
        total_weight += weight

    estimated_width = (
        weighted_width_sum / total_weight if total_weight > 0 else img_width
    )

    # 设置合理的界限
    min_width = img_width * 0.3  # 列宽不应该太窄
    max_width = img_width * 0.95  # 留一些页边距

    return min(max(estimated_width, min_width), max_width)


def locate_full_column(layout_res, col_width, img_width):
    # 找出跨列的模块
    for item in layout_res:
        cur_width = item["position"][1][0] - item["position"][0][0]
        if cur_width > col_width * 1.5 or cur_width > img_width * 0.7:
            item["category"] = "full column"
            item["col_number"] = 0
        else:
            item["category"] = "sub column"
            item["col_number"] = -1
    return layout_res


def fetch_column_info(layout_res, img_width, img_height):
    # 获取所有模块的横坐标范围
    layout_res.sort(key=lambda x: x["position"][0][0])

    col_width = cal_column_width(layout_res, img_width, img_height)
    layout_res = locate_full_column(layout_res, col_width, img_width)
    col_width = max(
        [
            item["position"][1][0] - item["position"][0][0]
            for item in layout_res
            if item["category"] == "sub column"
        ],
        default=col_width,
    )

    # 分配模块到列中
    col_left = img_width
    cur_col = 1
    for idx, info in enumerate(layout_res):
        if info["category"] == "full column":
            continue
        xmin, xmax = info["position"][0][0], info["position"][1][0]
        if col_left == img_width:
            col_left = xmin
        if xmin < col_left + col_width * 0.99 and xmax <= xmin + col_width * 1.02:
            info["col_number"] = cur_col
            col_left = min(col_left, xmin)
        else:
            cur_col += 1
            col_left = xmin
            info["col_number"] = cur_col
    logger.debug(f"Column number: {cur_col}, with column width: {col_width}")

    if cur_col == 1:
        # 只有一列，直接返回
        for item in layout_res:
            item["col_number"] = 1

    layout_res.sort(
        key=lambda x: (x["col_number"], x["position"][0][1], x["position"][0][0])
    )
    return layout_res
