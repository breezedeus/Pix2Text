# coding: utf-8
# [Pix2Text](https://github.com/breezedeus/pix2text): an Open-Source Alternative to Mathpix.
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).

import hashlib
import os
import re
import shutil
from copy import deepcopy
from functools import cmp_to_key
from pathlib import Path
import logging
import platform
import subprocess
from typing import Union, List, Any, Dict
from collections import Counter, defaultdict

from PIL import Image, ImageOps
import numpy as np
from numpy import random
import torch
from torchvision.utils import save_image

from .consts import MODEL_VERSION

fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' 'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()


def set_logger(log_file=None, log_level=logging.INFO, log_file_level=logging.NOTSET):
    """
    Example:
        >>> set_logger(log_file)
        >>> logger.info("abc'")
    """
    log_format = logging.Formatter(fmt)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        if not Path(log_file).parent.exists():
            os.makedirs(Path(log_file).parent)
        if isinstance(log_file, Path):
            log_file = str(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def custom_deepcopy(value):
    if isinstance(value, dict):
        return {key: custom_deepcopy(val) for key, val in value.items()}
    elif isinstance(value, list):
        return [custom_deepcopy(item) for item in value]
    elif isinstance(value, tuple):
        return tuple([custom_deepcopy(item) for item in value])
    elif isinstance(value, set):
        return set([custom_deepcopy(item) for item in value])
    else:
        try:
            return deepcopy(value)
        except TypeError:
            return value  # Return the original value if it cannot be deep copied


def select_device(device) -> str:
    if isinstance(device, str) and device.lower() == "gpu":
        device = "cuda"
    if device is not None:
        return device

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    return device


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'pix2text')
    else:
        return os.path.join(os.path.expanduser("~"), '.pix2text')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('PIX2TEXT_HOME', data_dir_default())


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def read_tsv_file(fp, sep='\t', img_folder=None, mode='eval'):
    img_fp_list, labels_list = [], []
    num_fields = 2 if mode != 'test' else 1
    with open(fp) as f:
        for line in f:
            fields = line.strip('\n').split(sep)
            assert len(fields) == num_fields
            img_fp = (
                os.path.join(img_folder, fields[0])
                if img_folder is not None
                else fields[0]
            )
            img_fp_list.append(img_fp)

            if mode != 'test':
                labels = fields[1].split(' ')
                labels_list.append(labels)

    return (img_fp_list, labels_list) if mode != 'test' else (img_fp_list, None)


def get_average_color(img):
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Check if image is grayscale (2D) or has channels (3D)
    if len(img_array.shape) < 3:
        # Grayscale image (single channel)
        avg_value = img_array.mean()
        return (int(avg_value),) * 3
    
    # Get average color, ignoring fully transparent pixels
    if img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:,:,3]
        rgb = img_array[:,:,:3]
        mask = alpha > 0
        if mask.any():
            avg_color = rgb[mask].mean(axis=0)
        else:
            avg_color = rgb.mean(axis=(0,1))
    else:  # RGB or other format
        channels = img_array.shape[2]
        if channels == 1:  # Single channel (like grayscale with dimension)
            avg_value = img_array.mean()
            return (int(avg_value),) * 3
        elif channels == 3:  # RGB
            avg_color = img_array.mean(axis=(0,1))
        else:  # Other formats, use first 3 channels or pad
            avg_color = img_array[:,:,:min(3, channels)].mean(axis=(0,1))
            # If less than 3 channels, duplicate the last one
            if channels < 3:
                avg_color = list(avg_color)
                while len(avg_color) < 3:
                    avg_color.append(avg_color[-1])
                avg_color = np.array(avg_color)
    
    return tuple(map(int, avg_color))


def get_contrasting_color(color):
    return tuple(255 - c for c in color)


def convert_transparent_to_contrasting(img: Image.Image):
    """
    Convert transparent pixels to a contrasting color.
    """
    # Check if the image has an alpha channel
    if img.mode in ('RGBA', 'LA'):
        # Get average color of non-transparent pixels
        avg_color = get_average_color(img)

        # Get contrasting color for background
        bg_color = get_contrasting_color(avg_color)

        # Create a new background image with the contrasting color
        # Add alpha channel (255) for RGBA format
        rgba_bg_color = bg_color + (255,)
        background = Image.new('RGBA', img.size, rgba_bg_color)

        # Paste the image on the background.
        # The alpha channel will be used as mask
        background.paste(img, (0, 0), img)

        # Convert to RGB (removes alpha channel)
        return background.convert('RGB')
    # Special handling for palette mode with transparency
    elif img.mode == 'P' and 'transparency' in img.info:
        # Convert P to RGBA first, which handles the transparency info properly
        img_rgba = img.convert('RGBA')
        
        # Get average color of non-transparent pixels
        avg_color = get_average_color(img_rgba)

        # Get contrasting color for background
        bg_color = get_contrasting_color(avg_color)

        # Create a new background image with the contrasting color
        rgba_bg_color = bg_color + (255,)
        background = Image.new('RGBA', img.size, rgba_bg_color)

        # Paste the RGBA-converted image on the background
        background.paste(img_rgba, (0, 0), img_rgba)

        # Convert to RGB (removes alpha channel)
        return background.convert('RGB')

    return img.convert('RGB')


def read_img(
    path: Union[str, Path], return_type='Tensor'
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """

    Args:
        path (str): image file path
        return_type (str): 返回类型；
            支持 `Tensor`，返回 torch.Tensor；`ndarray`，返回 np.ndarray；`Image`，返回 `Image.Image`

    Returns: RGB Image.Image, or np.ndarray / torch.Tensor, with shape [Channel, Height, Width]
    """
    assert return_type in ('Tensor', 'ndarray', 'Image')
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # 识别旋转后的图片（pillow不会自动识别）
    img = convert_transparent_to_contrasting(img)
    if return_type == 'Image':
        return img
    img = np.ascontiguousarray(np.array(img))
    if return_type == 'ndarray':
        return img
    return torch.tensor(img.transpose((2, 0, 1)))


def save_img(img: Union[torch.Tensor, np.ndarray], path):
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # img *= 255
    # img = img.to(dtype=torch.uint8)
    save_image(img, path)

    # Image.fromarray(img).save(path)


def get_background_color(image: Image.Image, margin=2):
    width, height = image.size

    # 边缘区域的像素采样
    edge_pixels = []
    for x in range(width):
        for y in range(height):
            if (
                x <= margin
                or y <= margin
                or x >= width - margin
                or y >= height - margin
            ):
                edge_pixels.append(image.getpixel((x, y)))

    # 统计边缘像素颜色频率
    color_counter = Counter(edge_pixels)

    # 获取频率最高的颜色
    background_color = color_counter.most_common(1)[0][0]

    return background_color


def add_img_margin(
    image: Image.Image, left_right_margin, top_bottom_margin, background_color=None
):
    if background_color is None:
        background_color = get_background_color(image)

    # 获取原始图片尺寸
    width, height = image.size

    # 计算新图片的尺寸
    new_width = width + left_right_margin * 2
    new_height = height + top_bottom_margin * 2

    # 创建新图片对象，并填充指定背景色
    new_image = Image.new("RGB", (new_width, new_height), background_color)

    # 将原始图片粘贴到新图片中央
    new_image.paste(image, (left_right_margin, top_bottom_margin))

    return new_image


def prepare_imgs(imgs: List[Union[str, Path, Image.Image]]) -> List[Image.Image]:
    output_imgs = []
    for img in imgs:
        if isinstance(img, (str, Path)):
            img = read_img(img, return_type='Image')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
        else:
            raise ValueError(f'Unsupported image type: {type(img)}')
        output_imgs.append(img)

    return output_imgs


COLOR_LIST = [
    [0, 140, 255],  # 深橙色
    [127, 255, 0],  # 春绿色
    [255, 144, 30],  # 道奇蓝
    [180, 105, 255],  # 粉红色
    [128, 0, 128],  # 紫色
    [0, 255, 255],  # 黄色
    [255, 191, 0],  # 深天蓝色
    [50, 205, 50],  # 石灰绿色
    [60, 20, 220],  # 猩红色
    [130, 0, 75],  # 靛蓝色
    [255, 0, 0],  # 红色
    [0, 255, 0],  # 绿色
    [0, 0, 255],  # 蓝色
]


def save_layout_img(img0, categories, one_out, save_path, key='position'):
    import cv2
    from cnstd.yolov7.plots import plot_one_box

    """可视化版面分析结果。"""
    if isinstance(img0, Image.Image):
        img0 = cv2.cvtColor(np.asarray(img0.convert('RGB')), cv2.COLOR_RGB2BGR)

    if len(categories) > 13:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in categories]
    else:
        colors = COLOR_LIST
    for one_box in one_out:
        _type = one_box.get('type', 'text')
        box = one_box[key]
        xyxy = [box[0, 0], box[0, 1], box[2, 0], box[2, 1]]
        label = str(_type)
        if 'score' in one_box:
            label += f', Score: {one_box["score"]:.2f}'
        if 'col_number' in one_box:
            label += f', Col: {one_box["col_number"]}'
        plot_one_box(
            xyxy,
            img0,
            label=label,
            color=colors[categories.index(_type)],
            line_thickness=1,
        )

    cv2.imwrite(str(save_path), img0)
    logger.info(f" The image with the result is saved in: {save_path}")


def rotated_box_to_horizontal(box):
    """将旋转框转换为水平矩形。

    :param box: [4, 2]，左上角、右上角、右下角、左下角的坐标
    """
    xmin = min(box[:, 0])
    xmax = max(box[:, 0])
    ymin = min(box[:, 1])
    ymax = max(box[:, 1])
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def is_valid_box(box, min_height=8, min_width=2) -> bool:
    """判断box是否有效。
    :param box: [4, 2]，左上角、右上角、右下角、左下角的坐标
    :param min_height: 最小高度
    :param min_width: 最小宽度
    :return: bool, 是否有效
    """
    return (
        box[0, 0] + min_width <= box[1, 0]
        and box[1, 1] + min_height <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_width
        and box[3, 1] >= box[0, 1] + min_height
    )


def list2box(xmin, ymin, xmax, ymax, dtype=float):
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=dtype
    )


def box2list(bbox):
    return [int(bbox[0, 0]), int(bbox[0, 1]), int(bbox[2, 0]), int(bbox[2, 1])]


def clipbox(box, img_height, img_width):
    new_box = np.zeros_like(box)
    new_box[:, 0] = np.clip(box[:, 0], 0, img_width - 1)
    new_box[:, 1] = np.clip(box[:, 1], 0, img_height - 1)
    return new_box


def y_overlap(box1, box2, key='position'):
    # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
    if key:
        box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
        box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    else:
        box1 = [box1[0][0], box1[0][1], box1[2][0], box1[2][1]]
        box2 = [box2[0][0], box2[0][1], box2[2][0], box2[2][1]]
    # 判断是否有交集
    if box1[3] <= box2[1] or box2[3] <= box1[1]:
        return 0
    # 计算交集的高度
    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))


def x_overlap(box1, box2, key='position'):
    # 计算它们在x轴上的IOU: Interaction / min(width1, width2)
    if key:
        box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
        box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    else:
        box1 = [box1[0][0], box1[0][1], box1[2][0], box1[2][1]]
        box2 = [box2[0][0], box2[0][1], box2[2][0], box2[2][1]]
    # 判断是否有交集
    if box1[2] <= box2[0] or box2[2] <= box1[0]:
        return 0
    # 计算交集的宽度
    x_min = max(box1[0], box2[0])
    x_max = min(box1[2], box2[2])
    return (x_max - x_min) / max(1, min(box1[2] - box1[0], box2[2] - box2[0]))


def overlap(box1, box2, key='position'):
    return x_overlap(box1, box2, key) * y_overlap(box1, box2, key)


def get_same_line_boxes(anchor, total_boxes):
    line_boxes = [anchor]
    for box in total_boxes:
        if box['line_number'] >= 0:
            continue
        if max([y_overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes


def _compare_box(box1, box2, anchor, key, left_best: bool = True):
    over1 = y_overlap(box1, anchor, key)
    over2 = y_overlap(box2, anchor, key)
    if box1[key][2, 0] < box2[key][0, 0] - 3:
        return -1
    elif box2[key][2, 0] < box1[key][0, 0] - 3:
        return 1
    else:
        if max(over1, over2) >= 3 * min(over1, over2):
            return over2 - over1 if left_best else over1 - over2
        return box1[key][0, 0] - box2[key][0, 0]


def sort_and_filter_line_boxes(line_boxes, key):
    if len(line_boxes) <= 1:
        return line_boxes

    allowed_max_overlay_x = 20

    def find_right_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        right_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][0, 0] >= anchor[key][2, 0] - allowed_max
        ]
        if not right_boxes:
            return None
        right_boxes = sorted(
            right_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=True)
            ),
        )
        return right_boxes[0]

    def find_left_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        left_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][2, 0] <= anchor[key][0, 0] + allowed_max
        ]
        if not left_boxes:
            return None
        left_boxes = sorted(
            left_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=False)
            ),
        )
        return left_boxes[-1]

    res_boxes = [line_boxes[0]]
    anchor = res_boxes[0]
    line_number = anchor['line_number']

    while True:
        right_box = find_right_box(anchor)
        if right_box is None:
            break
        right_box['line_number'] = line_number
        res_boxes.append(right_box)
        anchor = right_box

    anchor = res_boxes[0]
    while True:
        left_box = find_left_box(anchor)
        if left_box is None:
            break
        left_box['line_number'] = line_number
        res_boxes.insert(0, left_box)
        anchor = left_box

    return res_boxes


def merge_boxes(bbox1, bbox2):
    """
    Merge two bounding boxes to get a bounding box that encompasses both.

    Parameters:
    - bbox1, bbox2: The bounding boxes to merge. Each box is np.ndarray, with shape of [4, 2]

    Returns: new merged box, with shape of [4, 2]
    """
    # 解包两个边界框的坐标
    x_min1, y_min1, x_max1, y_max1 = box2list(bbox1)
    x_min2, y_min2, x_max2, y_max2 = box2list(bbox2)

    # 计算合并后边界框的坐标
    x_min = min(x_min1, x_min2)
    y_min = min(y_min1, y_min2)
    x_max = max(x_max1, x_max2)
    y_max = max(y_max1, y_max2)

    # 返回合并后的边界框
    return list2box(x_min, y_min, x_max, y_max)


def sort_boxes(boxes: List[dict], key='position') -> List[List[dict]]:
    # 按y坐标排序所有的框
    boxes.sort(key=lambda box: box[key][0, 1])
    for box in boxes:
        box['line_number'] = -1  # 所在行号，-1表示未分配

    def get_anchor():
        anchor = None
        for box in boxes:
            if box['line_number'] == -1:
                anchor = box
                break
        return anchor

    lines = []
    while True:
        anchor = get_anchor()
        if anchor is None:
            break
        anchor['line_number'] = len(lines)
        line_boxes = get_same_line_boxes(anchor, boxes)
        line_boxes = sort_and_filter_line_boxes(line_boxes, key)
        lines.append(line_boxes)

    return lines


def merge_adjacent_bboxes(line_bboxes):
    """
    合并同一行中相邻且足够接近的边界框（bboxes）。
    如果两个边界框在水平方向上的距离小于行的高度，则将它们合并为一个边界框。

    :param line_bboxes: 包含边界框信息的列表，每个边界框包含行号、位置（四个角点的坐标）和类型。
    :return: 合并后的边界框列表。
    """
    merged_bboxes = []
    current_bbox = None

    for bbox in line_bboxes:
        # 如果是当前行的第一个边界框，或者与上一个边界框不在同一行
        if current_bbox is None:
            current_bbox = bbox
            continue

        line_number = bbox['line_number']
        position = bbox['position']
        bbox_type = bbox['type']

        # 计算边界框的高度和宽度
        height = position[2, 1] - position[0, 1]

        # 检查当前边界框与上一个边界框的距离
        distance = position[0, 0] - current_bbox['position'][1, 0]
        if (
            current_bbox['type'] == 'text'
            and bbox_type == 'text'
            and distance <= height
        ):
            # 合并边界框：ymin 取两个框对应值的较小值，ymax 取两个框对应值的较大
            # [text]_[text] -> [text_text]
            ymin = min(position[0, 1], current_bbox['position'][0, 1])
            ymax = max(position[2, 1], current_bbox['position'][2, 1])
            xmin = current_bbox['position'][0, 0]
            xmax = position[2, 0]
            current_bbox['position'] = list2box(xmin, ymin, xmax, ymax)
        else:
            if (
                current_bbox['type'] == 'text'
                and bbox_type != 'text'
                and 0 < distance <= height
            ):
                # [text]_[embedding] -> [text_][embedding]
                current_bbox['position'][1, 0] = position[0, 0]
                current_bbox['position'][2, 0] = position[0, 0]
            elif (
                current_bbox['type'] != 'text'
                and bbox_type == 'text'
                and 0 < distance <= height
            ):
                # [embedding]_[text] -> [embedding][_text]
                position[0, 0] = current_bbox['position'][1, 0]
                position[3, 0] = current_bbox['position'][1, 0]
            # 添加当前边界框，并开始新的合并
            merged_bboxes.append(current_bbox)
            current_bbox = bbox

    if current_bbox is not None:
        merged_bboxes.append(current_bbox)

    return merged_bboxes


def adjust_line_height(bboxes, img_height, max_expand_ratio=0.2):
    """
    基于临近行与行之间间隙，把 box 的高度略微调高（检测出来的 box 可以挨着文字很近）。
    Args:
        bboxes (List[List[dict]]): 包含边界框信息的列表，每个边界框包含行号、位置（四个角点的坐标）和类型。
        img_height (int): 原始图像的高度。
        max_expand_ratio (float): 相对于 box 高度来说的上下最大扩展比率

    Returns:

    """

    def get_max_text_ymax(line_bboxes):
        return max([bbox['position'][2, 1] for bbox in line_bboxes])

    def get_min_text_ymin(line_bboxes):
        return min([bbox['position'][0, 1] for bbox in line_bboxes])

    if len(bboxes) < 1:
        return bboxes

    for line_idx, line_bboxes in enumerate(bboxes):
        next_line_ymin = (
            get_min_text_ymin(bboxes[line_idx + 1])
            if line_idx < len(bboxes) - 1
            else img_height
        )
        above_line_ymax = get_max_text_ymax(bboxes[line_idx - 1]) if line_idx > 0 else 0
        for box in line_bboxes:
            if box['type'] != 'text':
                continue
            box_height = box['position'][2, 1] - box['position'][0, 1]
            if box['position'][0, 1] > above_line_ymax:
                expand_size = min(
                    (box['position'][0, 1] - above_line_ymax) // 3,
                    int(max_expand_ratio * box_height),
                )
                box['position'][0, 1] -= expand_size
                box['position'][1, 1] -= expand_size
            if box['position'][2, 1] < next_line_ymin:
                expand_size = min(
                    (next_line_ymin - box['position'][2, 1]) // 3,
                    int(max_expand_ratio * box_height),
                )
                box['position'][2, 1] += expand_size
                box['position'][3, 1] += expand_size
    return bboxes


def adjust_line_width(
    text_box_infos, formula_box_infos, img_width, max_expand_ratio=0.2
):
    """
    如果不与其他 box 重叠，就把 text box 往左右稍微扩展一些（检测出来的 text box 在边界上可能会切掉边界字符的一部分）。
    Args:
        text_box_infos (List[dict]): 文本框信息，其中 'box' 字段包含四个角点的坐标。
        formula_box_infos (List[dict]): 公式框信息，其中 'position' 字段包含四个角点的坐标。
        img_width (int): 原始图像的宽度。
        max_expand_ratio (float): 相对于 box 高度来说的左右最大扩展比率。

    Returns: 扩展后的 text_box_infos。
    """

    def _expand_left_right(box):
        expanded_box = box.copy()
        xmin, xmax = box[0, 0], box[2, 0]
        box_height = box[2, 1] - box[0, 1]
        expand_size = int(max_expand_ratio * box_height)
        expanded_box[3, 0] = expanded_box[0, 0] = max(xmin - expand_size, 0)
        expanded_box[2, 0] = expanded_box[1, 0] = min(xmax + expand_size, img_width - 1)
        return expanded_box

    def _is_adjacent(anchor_box, text_box):
        if overlap(anchor_box, text_box, key=None) < 1e-6:
            return False
        anchor_xmin, anchor_xmax = anchor_box[0, 0], anchor_box[2, 0]
        text_xmin, text_xmax = text_box[0, 0], text_box[2, 0]
        if (
            text_xmin < anchor_xmin < text_xmax < anchor_xmax
            or anchor_xmin < text_xmin < anchor_xmax < text_xmax
        ):
            return True
        return False

    for idx, text_box in enumerate(text_box_infos):
        expanded_box = _expand_left_right(text_box['position'])
        overlapped = False
        cand_boxes = [
            _text_box['position']
            for _idx, _text_box in enumerate(text_box_infos)
            if _idx != idx
        ]
        cand_boxes.extend(
            [_formula_box['position'] for _formula_box in formula_box_infos]
        )
        for cand_box in cand_boxes:
            if _is_adjacent(expanded_box, cand_box):
                overlapped = True
                break
        if not overlapped:
            text_box_infos[idx]['position'] = expanded_box

    return text_box_infos


def crop_box(text_box, formula_box, min_crop_width=2) -> List[np.ndarray]:
    """
    将 text_box 与 formula_box 相交的部分裁剪掉
    Args:
        text_box ():
        formula_box ():
        min_crop_width (int): 裁剪后新的 text box 被保留的最小宽度，低于此宽度的 text box 会被删除。

    Returns:

    """
    text_xmin, text_xmax = text_box[0, 0], text_box[2, 0]
    text_ymin, text_ymax = text_box[0, 1], text_box[2, 1]
    formula_xmin, formula_xmax = formula_box[0, 0], formula_box[2, 0]

    cropped_boxes = []
    if text_xmin < formula_xmin:
        new_text_xmax = min(text_xmax, formula_xmin)
        if new_text_xmax - text_xmin >= min_crop_width:
            cropped_boxes.append((text_xmin, text_ymin, new_text_xmax, text_ymax))

    if text_xmax > formula_xmax:
        new_text_xmin = max(text_xmin, formula_xmax)
        if text_xmax - new_text_xmin >= min_crop_width:
            cropped_boxes.append((new_text_xmin, text_ymin, text_xmax, text_ymax))

    return [list2box(*box, dtype=None) for box in cropped_boxes]


def remove_overlap_text_bbox(text_box_infos, formula_box_infos):
    """
    如果一个 text box 与 formula_box 相交，则裁剪 text box。
    Args:
        text_box_infos ():
        formula_box_infos ():

    Returns:

    """

    new_text_box_infos = []
    for idx, text_box in enumerate(text_box_infos):
        max_overlap_val = 0
        max_overlap_fbox = None

        for formula_box in formula_box_infos:
            cur_val = overlap(text_box['position'], formula_box['position'], key=None)
            if cur_val > max_overlap_val:
                max_overlap_val = cur_val
                max_overlap_fbox = formula_box

        if max_overlap_val < 0.1:  # overlap 太少的情况不做任何处理
            new_text_box_infos.append(text_box)
            continue
        # if max_overlap_val > 0.8:  # overlap 太多的情况，直接扔掉 text box
        #     continue

        cropped_text_boxes = crop_box(
            text_box['position'], max_overlap_fbox['position']
        )
        if cropped_text_boxes:
            for _box in cropped_text_boxes:
                new_box = deepcopy(text_box)
                new_box['position'] = _box
                new_text_box_infos.append(new_box)

    return new_text_box_infos


def is_chinese(ch):
    """
    判断一个字符是否为中文字符
    """
    return '\u4e00' <= ch <= '\u9fff'


def find_first_punctuation_position(text):
    # 匹配常见标点符号的正则表达式
    pattern = re.compile(r'[,.!?;:()\[\]{}\'\"\\/-]')
    match = pattern.search(text)
    if match:
        return match.start()
    else:
        return len(text)


def smart_join(str_list, spellchecker=None):
    """
    对字符串列表进行拼接，如果相邻的两个字符串都是中文或包含空白符号，则不加空格；其他情况则加空格
    """

    def contain_whitespace(s):
        if re.search(r'\s', s):
            return True
        else:
            return False

    str_list = [s for s in str_list if s]
    if not str_list:
        return ''
    res = str_list[0]
    for i in range(1, len(str_list)):
        if (is_chinese(res[-1]) and is_chinese(str_list[i][0])) or contain_whitespace(
            res[-1] + str_list[i][0]
        ):
            res += str_list[i]
        elif spellchecker is not None and res.endswith('-'):
            fields = res.rsplit(' ', maxsplit=1)
            if len(fields) > 1:
                new_res, prev_word = fields[0], fields[1]
            else:
                new_res, prev_word = '', res

            fields = str_list[i].split(' ', maxsplit=1)
            if len(fields) > 1:
                next_word, new_next = fields[0], fields[1]
            else:
                next_word, new_next = str_list[i], ''

            punct_idx = find_first_punctuation_position(next_word)
            next_word = next_word[:punct_idx]
            new_next = str_list[i][len(next_word) :]
            new_word = prev_word[:-1] + next_word
            if (
                next_word
                and spellchecker.unknown([prev_word + next_word])
                and spellchecker.known([new_word])
            ):
                res = new_res + ' ' + new_word + new_next
            else:
                new_word = prev_word + next_word
                res = new_res + ' ' + new_word + new_next
        else:
            res += ' ' + str_list[i]
    return res


def cal_block_xmin_xmax(lines, indentation_thrsh):
    total_min_x, total_max_x = min(lines[:, 0]), max(lines[:, 1])
    if lines.shape[0] < 2:
        return total_min_x, total_max_x

    min_x, max_x = min(lines[1:, 0]), max(lines[1:, 1])
    first_line_is_full = total_max_x > max_x - indentation_thrsh
    if first_line_is_full:
        return min_x, total_max_x

    return total_min_x, total_max_x


def merge_line_texts(
    outs: List[Dict[str, Any]],
    auto_line_break: bool = True,
    line_sep='\n',
    embed_sep=(' $', '$ '),
    isolated_sep=('$$\n', '\n$$'),
    spellchecker=None,
) -> str:
    """
    把 Pix2Text.recognize_by_mfd() 的返回结果，合并成单个字符串
    Args:
        outs (List[Dict[str, Any]]):
        auto_line_break: 基于box位置自动判断是否该换行
        line_sep: 行与行之间的分隔符
        embed_sep (tuple): Prefix and suffix for embedding latex; default value is `(' $', '$ ')`
        isolated_sep (tuple): Prefix and suffix for isolated latex; default value is `('$$\n', '\n$$')`
        spellchecker: Spell Checker

    Returns: 合并后的字符串

    """
    if not outs:
        return ''
    out_texts = []
    line_margin_list = []  # 每行的最左边和最右边的x坐标
    isolated_included = []  # 每行是否包含了 `isolated` 类型的数学公式
    line_height_dict = defaultdict(list)  # 每行中每个块对应的高度
    line_ymin_ymax_list = []  # 每行的最上边和最下边的y坐标
    for _out in outs:
        line_number = _out.get('line_number', 0)
        while len(out_texts) <= line_number:
            out_texts.append([])
            line_margin_list.append([100000, 0])
            isolated_included.append(False)
            line_ymin_ymax_list.append([100000, 0])
        cur_text = _out['text']
        cur_type = _out.get('type', 'text')
        box = _out['position']
        if cur_type in ('embedding', 'isolated'):
            sep = isolated_sep if _out['type'] == 'isolated' else embed_sep
            cur_text = sep[0] + cur_text + sep[1]
        if cur_type == 'isolated':
            isolated_included[line_number] = True
            cur_text = line_sep + cur_text + line_sep
        out_texts[line_number].append(cur_text)
        line_margin_list[line_number][1] = max(
            line_margin_list[line_number][1], float(box[2, 0])
        )
        line_margin_list[line_number][0] = min(
            line_margin_list[line_number][0], float(box[0, 0])
        )
        if cur_type == 'text':
            line_height_dict[line_number].append(box[2, 1] - box[1, 1])
            line_ymin_ymax_list[line_number][0] = min(
                line_ymin_ymax_list[line_number][0], float(box[0, 1])
            )
            line_ymin_ymax_list[line_number][1] = max(
                line_ymin_ymax_list[line_number][1], float(box[2, 1])
            )

    line_text_list = [smart_join(o) for o in out_texts]

    for _line_number in line_height_dict.keys():
        if line_height_dict[_line_number]:
            line_height_dict[_line_number] = np.mean(line_height_dict[_line_number])
    _line_heights = list(line_height_dict.values())
    mean_height = np.mean(_line_heights) if _line_heights else None

    default_res = re.sub(rf'{line_sep}+', line_sep, line_sep.join(line_text_list))
    if not auto_line_break:
        return default_res

    line_lengths = [rx - lx for lx, rx in line_margin_list]
    line_length_thrsh = max(line_lengths) * 0.3
    if line_length_thrsh < 1:
        return default_res

    lines = np.array(
        [
            margin
            for idx, margin in enumerate(line_margin_list)
            if isolated_included[idx] or line_lengths[idx] >= line_length_thrsh
        ]
    )
    if lines.shape[0] < 1:
        return default_res
    min_x, max_x = min(lines[:, 0]), max(lines[:, 1])

    indentation_thrsh = (max_x - min_x) * 0.1
    if mean_height is not None:
        indentation_thrsh = 1.5 * mean_height

    min_x, max_x = cal_block_xmin_xmax(lines, indentation_thrsh)

    res_line_texts = [''] * len(line_text_list)
    line_text_list = [(idx, txt) for idx, txt in enumerate(line_text_list) if txt]
    for idx, (line_number, txt) in enumerate(line_text_list):
        if isolated_included[line_number]:
            res_line_texts[line_number] = line_sep + txt + line_sep
            continue

        tmp = txt
        if line_margin_list[line_number][0] > min_x + indentation_thrsh:
            tmp = line_sep + txt
        if line_margin_list[line_number][1] < max_x - indentation_thrsh:
            tmp = tmp + line_sep
        if idx < len(line_text_list) - 1:
            cur_height = line_ymin_ymax_list[line_number][1] - line_ymin_ymax_list[line_number][0]
            next_line_number = line_text_list[idx + 1][0]
            if (
                cur_height > 0
                and line_ymin_ymax_list[next_line_number][0] < line_ymin_ymax_list[next_line_number][1]
                and line_ymin_ymax_list[next_line_number][0] - line_ymin_ymax_list[line_number][1]
                > cur_height
            ):  # 当前行与下一行的间距超过了一行的行高，则认为它们之间应该是不同的段落
                tmp = tmp + line_sep
        res_line_texts[idx] = tmp

    outs = smart_join([c for c in res_line_texts if c], spellchecker)
    return re.sub(rf'{line_sep}+', line_sep, outs)  # 把多个 '\n' 替换为 '\n'


def prepare_model_files(root, model_info, mirror_url='https://hf-mirror.com') -> Path:
    model_root_dir = Path(root) / MODEL_VERSION
    model_dir = model_root_dir / model_info['local_model_id']
    if model_dir.is_dir() and list(model_dir.glob('**/[!.]*')):
        return model_dir
    assert 'hf_model_id' in model_info
    model_dir.mkdir(parents=True)
    download_cmd = f'huggingface-cli download --repo-type model --resume-download --local-dir-use-symlinks False {model_info["hf_model_id"]} --local-dir {model_dir}'
    subprocess.run(download_cmd, shell=True)
    # 如果当前目录下无文件，就从huggingface上下载
    if not list(model_dir.glob('**/[!.]*')):
        if model_dir.exists():
            shutil.rmtree(str(model_dir))
        # os.system('HF_ENDPOINT=https://hf-mirror.com ' + download_cmd)
        env = os.environ.copy()
        env['HF_ENDPOINT'] = mirror_url
        subprocess.run(download_cmd, env=env, shell=True)
    return model_dir


def prepare_model_files2(model_fp_or_dir, remote_repo, file_or_dir='file', mirror_url='https://hf-mirror.com'):
    """
    从远程指定的仓库下载模型文件。
    Args:
        model_fp_or_dir: 下载的模型文件会保存到此路径
        remote_repo: 指定的远程仓库
        file_or_dir: model_fp_or_dir 是文件路径还是目录路径。注：下载的都是目录
        mirror_url: 指定的 HuggingFace 国内镜像网址；如果无法从 HuggingFace 官方仓库下载，会自动从此国内镜像下载。默认值为 'https://hf-mirror.com'
    """
    model_fp_or_dir = Path(model_fp_or_dir)
    if file_or_dir == 'file':
        if model_fp_or_dir.exists():
            return model_fp_or_dir
        model_dir = model_fp_or_dir.parent
    else:
        model_dir = model_fp_or_dir
    if model_dir.exists():
        shutil.rmtree(str(model_dir))
    model_dir.mkdir(parents=True)
    download_cmd = f'huggingface-cli download --repo-type model --resume-download --local-dir-use-symlinks False {remote_repo} --local-dir {model_dir}'
    subprocess.run(download_cmd, shell=True)
    download_status = False
    if file_or_dir == 'file':
        if model_fp_or_dir.exists():  # download failed above
            download_status = True
    else:  # model_dir 存在且非空，则下载成功
        if model_dir.exists() and list(model_dir.glob('**/[!.]*')):
            download_status = True
    if not download_status:  # download failed above
        if model_dir.exists():
            shutil.rmtree(str(model_dir))
        env = os.environ.copy()
        env['HF_ENDPOINT'] = mirror_url
        subprocess.run(download_cmd, env=env, shell=True)
    return model_fp_or_dir
