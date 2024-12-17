# coding: utf-8
# Adapted from https://github.com/microsoft/table-transformer/blob/main/src/inference.py
import os
import shutil
from collections import defaultdict, OrderedDict
from itertools import chain
from pathlib import Path
from typing import Union, Optional, Dict, Any
from copy import deepcopy
import xml.etree.ElementTree as ET

import torch
from torchvision import transforms
from PIL import Image
from fitz import Rect
import numpy as np
import pandas as pd
from transformers import AutoModelForObjectDetection

# from transformers import TableTransformerForObjectDetection

from .consts import MODEL_VERSION
from .ocr_engine import TextOcrEngine
from .utils import (
    select_device,
    data_dir,
    read_img,
    rotated_box_to_horizontal,
    is_valid_box,
    list2box,
    box2list,
    sort_boxes,
    merge_line_texts,
    prepare_model_files2,
)
from . import table_postprocess as postprocess


# detection_class_thresholds = {"table": 0.5, "table rotated": 0.5, "no object": 10}
DEFAULT_STRUCTURE_THRESHOLDS = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10,
}


DEFAULT_CONFIGS = {
    'model_dir': None,
    'root': data_dir(),
    'structure_thresholds': DEFAULT_STRUCTURE_THRESHOLDS,
    'table_expansion_margin': 10,
    'threshold_percentage': 0.10,
}


class TableOCR(object):
    """
    Represents a table extractor for extracting tables from a document.
    """

    def __init__(
        self,
        text_ocr: TextOcrEngine,
        spellchecker=None,
        device: str = None,
        model_dir: Optional[Union[str, Path]] = None,
        root: Union[str, Path] = data_dir(),
        structure_thresholds=None,
        table_expansion_margin=10,
        threshold_percentage=0.10,
        **kwargs,
    ):
        """
        Initialize an TableDataExtractor object.
        """
        self.text_ocr = text_ocr
        self.spellchecker = spellchecker

        self.str_device = select_device(device)
        self.str_class_name2idx = get_class_map('structure')
        self.str_class_idx2name = {v: k for k, v in self.str_class_name2idx.items()}
        self.str_class_thresholds = structure_thresholds or DEFAULT_STRUCTURE_THRESHOLDS

        if model_dir is None:
            model_dir = self._prepare_model_files(root, None)
        # Initialize the model for identifying table structures
        self.str_model = AutoModelForObjectDetection.from_pretrained(model_dir).to(
            self.str_device
        )
        self.str_model.eval()

        # Expand the bounding box slightly for better cropping
        self._table_expansion_margin = table_expansion_margin

        # Use a percentage (e.g., 10%) of the average height as the threshold for a new row
        self._threshold_percentage = threshold_percentage
        self.test = []

    @classmethod
    def from_config(
        cls,
        text_ocr: TextOcrEngine,
        spellchecker=None,
        configs: Optional[dict] = None,
        device: str = None,
        **kwargs,
    ):
        configs = configs or {}
        def_configs = deepcopy(DEFAULT_CONFIGS)
        def_configs.update(configs)
        configs = def_configs
        configs['device'] = select_device(device)

        return cls(
            text_ocr=text_ocr,
            spellchecker=spellchecker,
            device=device,
            model_dir=configs['model_dir'],
            root=configs['root'],
            structure_thresholds=configs['structure_thresholds'],
            table_expansion_margin=configs['table_expansion_margin'],
            threshold_percentage=configs['threshold_percentage'],
            **kwargs,
        )

    def _prepare_model_files(self, root, model_info):
        model_root_dir = Path(root) / MODEL_VERSION
        # model_dir = model_root_dir / model_info['local_model_id']
        model_dir = model_root_dir / 'table-rec'
        if model_dir.is_dir() and list(model_dir.glob('**/[!.]*')):
            return model_dir
        model_dir = prepare_model_files2(
            model_fp_or_dir=model_dir,
            remote_repo="breezedeus/pix2text-table-rec",
            file_or_dir="dir",
        )
        return model_dir

    def recognize(
        self,
        img,
        tokens=None,
        out_objects=False,
        out_cells=True,
        out_html=False,
        out_csv=False,
        out_markdown=True,
        **kwargs,
    ) -> Dict[str, Any]:
        """

        Args:
            img ():
            tokens ():
            out_objects ():
            out_cells ():
            out_html ():
            out_csv ():
            out_markdown ():
            **kwargs ():

                * save_analysis_res (str): Save the parsed result image in this file; default value is `None`, which means not to save

        Returns:

        """
        out_formats = {}
        if self.str_model is None:
            print("No structure model loaded.")
            return out_formats

        if not (out_objects or out_cells or out_html or out_csv):
            print("No output format specified")
            return out_formats

        if isinstance(img, (str, Path)):
            img = read_img(img, return_type='Image')
        # Transform the image how the model expects it
        img_tensor = structure_transform(img)

        # Run input image through the model
        with torch.no_grad():
            outputs = self.str_model(img_tensor.unsqueeze(0).to(self.str_device))

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.str_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not (out_cells or out_html or out_csv):
            return out_formats

        # Further process the detected objects so they correspond to a consistent table
        tokens = tokens or []
        tables_structure = objects_to_structures(
            objects, tokens, self.str_class_thresholds
        )

        # Enumerate all table cells: grid cells and spanning cells
        tables_cells = [
            structure_to_cells(structure, tokens)[0] for structure in tables_structure
        ]
        for cells in tables_cells:
            self._ocr_texts(img, cells)
        if out_cells:
            out_formats['cells'] = tables_cells
            if kwargs.get('save_analysis_res'):
                visualize_cells(img, tables_cells[0], kwargs['save_analysis_res'])

        if not (out_html or out_csv):
            return out_formats

        # Convert cells to HTML
        if out_html:
            tables_htmls = [cells_to_html(cells) for cells in tables_cells]
            out_formats['html'] = tables_htmls

        # Convert cells to CSV, including flattening multi-row column headers to a single row
        if out_csv:
            tables_csvs = [cells_to_csv(cells) for cells in tables_cells]
            out_formats['csv'] = tables_csvs

        if out_markdown:
            tables_mds = [cells_to_markdown(cells) for cells in tables_cells]
            out_formats['markdown'] = tables_mds

        return out_formats

    def _ocr_texts(self, img: Image.Image, cells):
        text_box_infos = self.text_ocr.detect_only(np.array(img))
        box_infos = []
        for line_box_info in text_box_infos['detected_texts']:
            _text_box = rotated_box_to_horizontal(line_box_info['position'])
            if not is_valid_box(_text_box, min_height=8, min_width=2):
                continue
            box_infos.append({'position': _text_box})
        for t_cell in cells:
            table_box = t_cell['bbox']
            inner_text_boxes = []
            for box_info in box_infos:
                _pos = box_info['position']
                text_box = [_pos[0][0], _pos[0][1], _pos[2][0], _pos[2][1]]
                inner_box = list2box(*cut_bbox(table_box, text_box))
                if is_valid_box(inner_box):
                    inner_text_boxes.append({'position': inner_box})
            if inner_text_boxes:
                for _box_info in inner_text_boxes:
                    _box = box2list(_box_info['position'])
                    ocr_out = self.text_ocr.recognize_only(np.array(img.crop(_box)))
                    _box_info['text'] = ocr_out['text']
                    _box_info['type'] = 'text'
                outs = sort_boxes(inner_text_boxes, key='position')
                t_cell['text_bboxes'] = outs
                outs = list(chain(*outs))
                t_cell['cell text'] = merge_line_texts(
                    outs,
                    auto_line_break=True,
                    line_sep=' ',
                    spellchecker=self.spellchecker,
                )


def cut_bbox(anchor_box, box2):
    # x1, y1, x2, y2
    x1 = max(anchor_box[0], box2[0])
    y1 = max(anchor_box[1], box2[1])
    x2 = min(anchor_box[2], box2[2])
    y2 = min(anchor_box[3], box2[3])
    return x1, y1, x2, y2


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


def get_class_map(data_type):
    class_map = {}
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6,
        }
    elif data_type == 'detection':
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map


detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0


def align_headers(headers, rows):
    """
    Adjust the header boundary to be the convex hull of the rows it intersects
    at least 50% of the height of.

    For now, we are not supporting tables with multiple headers, so we need to
    eliminate anything besides the top-most header.
    """

    aligned_headers = []

    for row in rows:
        row['column header'] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row['bbox'][3] - row['bbox'][1]
            min_row_overlap = max(row['bbox'][1], header['bbox'][1])
            max_row_overlap = min(row['bbox'][3], header['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row['column header'] = True
            header_rect = header_rect.include_rect(row['bbox'])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {'bbox': list(header_rect)}
    aligned_headers.append(header)

    return aligned_headers


def refine_table_structure(table_structure, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structure["rows"]
    columns = table_structure['columns']

    # Process the headers
    column_headers = table_structure['column headers']
    column_headers = postprocess.apply_threshold(
        column_headers, class_thresholds["table column header"]
    )
    column_headers = postprocess.nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process spanning cells
    spanning_cells = [
        elem
        for elem in table_structure['spanning cells']
        if not elem['projected row header']
    ]
    projected_row_headers = [
        elem
        for elem in table_structure['spanning cells']
        if elem['projected row header']
    ]
    spanning_cells = postprocess.apply_threshold(
        spanning_cells, class_thresholds["table spanning cell"]
    )
    projected_row_headers = postprocess.apply_threshold(
        projected_row_headers, class_thresholds["table projected row header"]
    )
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = postprocess.align_supercells(spanning_cells, rows, columns)
    spanning_cells = postprocess.nms_supercells(spanning_cells)

    postprocess.header_supercell_tree(spanning_cells)

    table_structure['columns'] = columns
    table_structure['rows'] = rows
    table_structure['spanning cells'] = spanning_cells
    table_structure['column headers'] = column_headers

    return table_structure


def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs['logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append(
                {
                    'label': class_label,
                    'score': float(score),
                    'bbox': [float(elem) for elem in bbox],
                }
            )

    return objects


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [
                token['bbox'][0] - bbox[0],
                token['bbox'][1] - bbox[1],
                token['bbox'][2] - bbox[0],
                token['bbox'][3] - bbox[1],
            ]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [
                    cropped_img.size[0] - bbox[3] - 1,
                    bbox[0],
                    cropped_img.size[0] - bbox[1] - 1,
                    bbox[2],
                ]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).
    """

    tables = [obj for obj in objects if obj['label'] == 'table']
    table_structures = []

    for table in tables:
        table_objects = [
            obj for obj in objects if iob(obj['bbox'], table['bbox']) >= 0.5
        ]
        table_tokens = [
            token for token in tokens if iob(token['bbox'], table['bbox']) >= 0.5
        ]

        structure = {}

        columns = [obj for obj in table_objects if obj['label'] == 'table column']
        rows = [obj for obj in table_objects if obj['label'] == 'table row']
        column_headers = [
            obj for obj in table_objects if obj['label'] == 'table column header'
        ]
        spanning_cells = [
            obj for obj in table_objects if obj['label'] == 'table spanning cell'
        ]
        for obj in spanning_cells:
            obj['projected row header'] = False
        projected_row_headers = [
            obj for obj in table_objects if obj['label'] == 'table projected row header'
        ]
        for obj in projected_row_headers:
            obj['projected row header'] = True
        spanning_cells += projected_row_headers
        for obj in rows:
            obj['column header'] = False
            for header_obj in column_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['column header'] = True

        # Refine table structures
        rows = postprocess.refine_rows(
            rows, table_tokens, class_thresholds['table row']
        )
        columns = postprocess.refine_columns(
            columns, table_tokens, class_thresholds['table column']
        )

        # Shrink table bbox to just the total height of the rows
        # and the total width of the columns
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj['bbox'])
        column_rect = Rect()
        for obj in columns:
            column_rect.include_rect(obj['bbox'])
        table['row_column_bbox'] = [
            column_rect[0],
            row_rect[1],
            column_rect[2],
            row_rect[3],
        ]
        table['bbox'] = table['row_column_bbox']

        # Process the rows and columns into a complete segmented table
        columns = postprocess.align_columns(columns, table['row_column_bbox'])
        rows = postprocess.align_rows(rows, table['row_column_bbox'])

        structure['rows'] = rows
        structure['columns'] = columns
        structure['column headers'] = column_headers
        structure['spanning cells'] = spanning_cells

        if len(rows) > 0 and len(columns) > 1:
            structure = refine_table_structure(structure, class_thresholds)

        table_structures.append(structure)

    return table_structures


def structure_to_cells(table_structure, tokens):
    """
    Assuming the row, column, spanning cell, and header bounding boxes have
    been refined into a set of consistent table structures, process these
    table structures into table cells. This is a universal representation
    format for the table, which can later be exported to Pandas or CSV formats.
    Classify the cells as header/access cells or data cells
    based on if they intersect with the header bounding box.
    """
    columns = table_structure['columns']
    rows = table_structure['rows']
    spanning_cells = table_structure['spanning cells']
    cells = []
    subcells = []

    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            column_rect = Rect(list(column['bbox']))
            row_rect = Rect(list(row['bbox']))
            cell_rect = row_rect.intersect(column_rect)
            header = 'column header' in row and row['column header']
            cell = {
                'bbox': list(cell_rect),
                'column_nums': [column_num],
                'row_nums': [row_num],
                'column header': header,
            }

            cell['subcell'] = False
            for spanning_cell in spanning_cells:
                spanning_cell_rect = Rect(list(spanning_cell['bbox']))
                if (
                    spanning_cell_rect.intersect(cell_rect).get_area()
                    / cell_rect.get_area()
                ) > 0.5:
                    cell['subcell'] = True
                    break

            if cell['subcell']:
                subcells.append(cell)
            else:
                # cell text = extract_text_inside_bbox(table_spans, cell['bbox'])
                # cell['cell text'] = cell text
                cell['projected row header'] = False
                cells.append(cell)

    for spanning_cell in spanning_cells:
        spanning_cell_rect = Rect(list(spanning_cell['bbox']))
        cell_columns = set()
        cell_rows = set()
        cell_rect = None
        header = True
        for subcell in subcells:
            subcell_rect = Rect(list(subcell['bbox']))
            subcell_rect_area = subcell_rect.get_area()
            if (
                subcell_rect.intersect(spanning_cell_rect).get_area()
                / subcell_rect_area
            ) > 0.5:
                if cell_rect is None:
                    cell_rect = Rect(list(subcell['bbox']))
                else:
                    cell_rect.include_rect(Rect(list(subcell['bbox'])))
                cell_rows = cell_rows.union(set(subcell['row_nums']))
                cell_columns = cell_columns.union(set(subcell['column_nums']))
                # By convention here, all subcells must be classified
                # as header cells for a spanning cell to be classified as a header cell;
                # otherwise, this could lead to a non-rectangular header region
                header = (
                    header and 'column header' in subcell and subcell['column header']
                )
        if len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {
                'bbox': list(cell_rect),
                'column_nums': list(cell_columns),
                'row_nums': list(cell_rows),
                'column header': header,
                'projected row header': spanning_cell['projected row header'],
            }
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = postprocess.slot_into_containers(cells, tokens)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    # dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    # dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = Rect()
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(dilated_columns[column_num]['bbox']))
        row_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(dilated_rows[row_num]['bbox']))
        cell_rect = column_rect.intersect(row_rect)
        cell['bbox'] = list(cell_rect)

    span_nums_by_cell, _, _ = postprocess.slot_into_containers(
        cells,
        tokens,
        overlap_threshold=0.001,
        unique_assignment=True,
        forced_assignment=False,
    )

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell['cell text'] = postprocess.extract_text_from_spans(
            cell_spans, remove_integer_superscripts=False
        )
        cell['spans'] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = postprocess.sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = postprocess.sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell['spans']:
            min_x_values_by_column[min_column].append(span['bbox'][0])
            min_y_values_by_row[min_row].append(span['bbox'][1])
            max_x_values_by_column[max_column].append(span['bbox'][2])
            max_y_values_by_row[max_row].append(span['bbox'][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row['bbox'][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row['bbox'][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row['bbox'][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row['bbox'][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column['bbox'][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column['bbox'][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column['bbox'][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column['bbox'][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = Rect()
        column_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(rows[row_num]['bbox']))
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(columns[column_num]['bbox']))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell['bbox'] = list(cell_rect)
            pass

    return cells, confidence_score


def cells_to_csv(cells):
    if len(cells) > 0:
        num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell['column header']]
    if len(header_cells) > 0:
        max_header_row = max([max(cell['row_nums']) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype="object")
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell['row_nums']:
                for column_num in cell['column_nums']:
                    table_array[row_num, column_num] = cell["cell text"]

    header = table_array[: max_header_row + 1, :]
    flattened_header = []
    for col in header.transpose():
        flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
    df = pd.DataFrame(
        table_array[max_header_row + 1 :, :], index=None, columns=flattened_header
    )

    return df.to_csv(index=None)


def cells_to_ET(cells):
    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = min(cell['row_nums'])

        attrib = {}
        colspan = len(cell['column_nums'])
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = len(cell['row_nums'])
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell['column header']:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell['cell text']
    return table


def cells_to_html(cells):
    table = cells_to_ET(cells)
    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))


def cells_to_markdown(cells):
    table = cells_to_ET(cells)
    return etree_to_markdown_table(table)


def etree_to_markdown_table(etree):
    """
    将XML ElementTree对象转换为Markdown格式的表格。

    Args:
        etree (xml.etree.ElementTree.Element): XML表格的根元素。

    Returns:
        str: Markdown格式的表格字符串。
    """
    if etree.tag != 'table':
        return "Invalid XML input: root element is not a table."

    markdown_table = []
    headers = [th.text for th in etree.findall('.//th')]

    if headers:
        markdown_table.append("| " + " | ".join(headers) + " |")
        markdown_table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    rows = etree.findall('.//tr')
    if rows:
        for row in rows:
            cells = [td.text.replace('\n', ' ') for td in row.findall('td')]
            if not cells:
                continue
            markdown_table.append("| " + " | ".join(cells) + " |")
    else:
        return "Invalid XML input: no rows found."

    return "\n".join(markdown_table)


def visualize_detected_tables(img, det_tables, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Patch

    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            continue

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor='none',
            facecolor=facecolor,
            alpha=0.1,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor='none',
            linestyle='-',
            alpha=alpha,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=0,
            edgecolor=edgecolor,
            facecolor='none',
            linestyle='-',
            hatch=hatch,
            alpha=0.2,
        )
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [
        Patch(
            facecolor=(1, 0, 0.45),
            edgecolor=(1, 0, 0.45),
            label='Table',
            hatch='//////',
            alpha=0.3,
        ),
        Patch(
            facecolor=(0.95, 0.6, 0.1),
            edgecolor=(0.95, 0.6, 0.1),
            label='Table (rotated)',
            hatch='//////',
            alpha=0.3,
        ),
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, -0.02),
        loc='upper center',
        borderaxespad=0,
        fontsize=10,
        ncol=2,
    )
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return


def visualize_cells(img, cells, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Patch

    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()

    for cell in cells:
        bbox = cell['bbox']

        if cell['column header']:
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif cell['projected row header']:
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            facecolor = (0.3, 0.74, 0.8)
            edgecolor = (0.3, 0.7, 0.6)
            alpha = 0.3
            linewidth = 2
            hatch = '\\\\\\\\\\\\'

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor='none',
            facecolor=facecolor,
            alpha=0.1,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor='none',
            linestyle='-',
            alpha=alpha,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=0,
            edgecolor=edgecolor,
            facecolor='none',
            linestyle='-',
            hatch=hatch,
            alpha=0.2,
        )
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [
        Patch(
            facecolor=(0.3, 0.74, 0.8),
            edgecolor=(0.3, 0.7, 0.6),
            label='Data cell',
            hatch='\\\\\\\\\\\\',
            alpha=0.3,
        ),
        Patch(
            facecolor=(1, 0, 0.45),
            edgecolor=(1, 0, 0.45),
            label='Column header cell',
            hatch='//////',
            alpha=0.3,
        ),
        Patch(
            facecolor=(0.95, 0.6, 0.1),
            edgecolor=(0.95, 0.6, 0.1),
            label='Projected row header cell',
            hatch='//////',
            alpha=0.3,
        ),
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, -0.02),
        loc='upper center',
        borderaxespad=0,
        fontsize=10,
        ncol=3,
    )
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    return


def output_result(key, val, args, img, img_file):
    import json

    if key == 'objects':
        # if args.verbose:
        #     print(val)
        out_file = img_file.replace(".jpg", "_objects.json")
        with open(os.path.join(args.out_dir, out_file), 'w', encoding='utf-8') as f:
            json.dump(val, f)
        # if args.visualize:
        #     out_file = img_file.replace(".jpg", "_fig_tables.jpg")
        #     out_path = os.path.join(args.out_dir, out_file)
        #     visualize_detected_tables(img, val, out_path)
    elif not key == 'image' and not key == 'tokens':
        for idx, elem in enumerate(val):
            if key == 'crops':
                for idx, cropped_table in enumerate(val):
                    out_img_file = img_file.replace(".jpg", "_table_{}.jpg".format(idx))
                    cropped_table['image'].save(
                        os.path.join(args.out_dir, out_img_file)
                    )
                    out_words_file = out_img_file.replace(".jpg", "_words.json")
                    with open(os.path.join(args.out_dir, out_words_file), 'w', encoding='utf-8') as f:
                        json.dump(cropped_table['tokens'], f)
            elif key == 'cells':
                out_file = img_file.replace(".jpg", "_{}_objects.json".format(idx))
                with open(os.path.join(args.out_dir, out_file), 'w', encoding='utf-8') as f:
                    json.dump(elem, f)
                # if args.verbose:
                #     print(elem)
                if True:
                    out_file = img_file.replace(".jpg", "_fig_cells.jpg")
                    out_path = os.path.join(args.out_dir, out_file)
                    visualize_cells(img, elem, out_path)
            else:
                out_file = img_file.replace(".jpg", "_{}.{}".format(idx, key))
                with open(os.path.join(args.out_dir, out_file), 'w', encoding='utf-8') as f:
                    f.write(elem)
                if args.verbose:
                    print(elem)
