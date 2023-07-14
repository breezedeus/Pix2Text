# coding: utf-8
# Copyright (C) 2022-2023, [Breezedeus](https://www.breezedeus.com).

import hashlib
import os
import re
from functools import cmp_to_key
from pathlib import Path
import logging
import platform
from typing import Union, List, Any, Dict

from PIL import Image, ImageOps
import numpy as np
from numpy import random
import torch
from torch import Tensor
from torchvision.utils import save_image


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


def check_context(context):
    if isinstance(context, str):
        return any([ctx in context.lower() for ctx in ('gpu', 'cpu', 'cuda')])
    if isinstance(context, list):
        if len(context) < 1:
            return False
        return all(isinstance(ctx, torch.device) for ctx in context)
    return isinstance(context, torch.device)


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
    img = ImageOps.exif_transpose(img).convert('RGB')  # 识别旋转后的图片（pillow不会自动识别）
    if return_type == 'Image':
        return img
    img = np.array(img)
    if return_type == 'ndarray':
        return img
    return torch.tensor(img.transpose((2, 0, 1)))


def save_img(img: Union[Tensor, np.ndarray], path):
    if not isinstance(img, Tensor):
        img = torch.from_numpy(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # img *= 255
    # img = img.to(dtype=torch.uint8)
    save_image(img, path)

    # Image.fromarray(img).save(path)


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
    [130, 0, 75]  # 靛蓝色
]


def save_layout_img(img0, categories, one_out, save_path, key='position'):
    import cv2
    from cnstd.yolov7.plots import plot_one_box

    """可视化版面分析结果。"""
    if isinstance(img0, Image.Image):
        img0 = cv2.cvtColor(np.asarray(img0.convert('RGB')), cv2.COLOR_RGB2BGR)

    if len(categories) > 10:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in categories]
    else:
        colors = COLOR_LIST
    for one_box in one_out:
        _type = one_box['type']
        box = one_box[key]
        xyxy = [box[0, 0], box[0, 1], box[2, 0], box[2, 1]]
        label = f'{_type}'
        plot_one_box(
            xyxy,
            img0,
            label=label,
            color=colors[categories.index(_type)],
            line_thickness=1,
        )

    cv2.imwrite(save_path, img0)
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


def list2box(xmin, ymin, xmax, ymax):
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
    )


def overlap(box1, box2, key='position'):
    # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
    # 判断是否有交集
    box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
    box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    if box1[3] <= box2[1] or box2[3] <= box1[1]:
        return 0
    # 计算交集的高度
    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))


def get_same_line_boxes(anchor, total_boxes):
    line_boxes = [anchor]
    for box in total_boxes:
        if box['line_number'] >= 0:
            continue
        if max([overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes


def _compare_box(box1, box2, anchor, key, left_best: bool = True):
    over1 = overlap(box1, anchor, key)
    over2 = overlap(box2, anchor, key)
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


def is_chinese(ch):
    """
    判断一个字符是否为中文字符
    """
    return '\u4e00' <= ch <= '\u9fff'


def smart_join(str_list):
    """
    对字符串列表进行拼接，如果相邻的两个字符串都是中文或包含空白符号，则不加空格；其他情况则加空格
    """

    def contain_whitespace(s):
        if re.search(r'\s', s):
            return True
        else:
            return False

    res = str_list[0]
    for i in range(1, len(str_list)):
        if (is_chinese(res[-1]) and is_chinese(str_list[i][0])) or contain_whitespace(
            res[-1] + str_list[i][0]
        ):
            res += str_list[i]
        else:
            res += ' ' + str_list[i]
    return res


def merge_line_texts(
    out: List[Dict[str, Any]], auto_line_break: bool = True, line_sep='\n'
) -> str:
    """
    把 Pix2Text.recognize_by_mfd() 的返回结果，合并成单个字符串
    Args:
        out (List[Dict[str, Any]]):
        auto_line_break: 基于box位置自动判断是否该换行

    Returns: 合并后的字符串

    """
    out_texts = []
    line_margin_list = []  # 每行的最左边和左右边的x坐标
    isolated_included = []  # 每行是否包含了 `isolated` 类型的数学公式
    for o in out:
        line_number = o.get('line_number', 0)
        if len(out_texts) <= line_number:
            out_texts.append([])
            line_margin_list.append([0, 0])
            isolated_included.append(False)
        out_texts[line_number].append(o['text'])
        line_margin_list[line_number][1] = max(
            line_margin_list[line_number][1], float(o['position'][2, 0])
        )
        line_margin_list[line_number][0] = min(
            line_margin_list[line_number][0], float(o['position'][0, 0])
        )
        if o['type'] == 'isolated':
            isolated_included[line_number] = True

    line_text_list = [smart_join(o) for o in out_texts]

    if not auto_line_break:
        return line_sep.join(line_text_list)

    line_lengths = [rx - lx for lx, rx in line_margin_list]
    line_length_thrsh = max(line_lengths) * 0.3

    lines = np.array(
        [
            margin
            for idx, margin in enumerate(line_margin_list)
            if isolated_included[idx] or line_lengths[idx] >= line_length_thrsh
        ]
    )
    min_x, max_x = lines.max(axis=0)

    indentation_thrsh = (max_x - min_x) * 0.1
    res_line_texts = [''] * len(line_text_list)
    for idx, txt in enumerate(line_text_list):
        if isolated_included[idx]:
            res_line_texts[idx] = line_sep + txt + line_sep
            continue

        tmp = txt
        if line_margin_list[idx][0] > min_x + indentation_thrsh:
            tmp = line_sep + txt
        if line_margin_list[idx][1] < max_x - indentation_thrsh:
            tmp = tmp + line_sep
        res_line_texts[idx] = tmp

    out = smart_join(res_line_texts)
    return out.replace(line_sep + line_sep, line_sep)  # 把 '\n\n' 替换为 '\n'
