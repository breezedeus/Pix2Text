# coding: utf-8
import random
from pprint import pprint
import numpy as np

from pix2text.utils import sort_and_filter_line_boxes, sort_boxes


def list2box(xmin, ymin, xmax, ymax):
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def test_sort_line_boxes():
    boxes = [
        {'position': list2box(20, 0, 30, 25), 'id': 12},  # anchor，不能动
        {'position': list2box(0, 5, 20, 20), 'id': 11},
        {'position': list2box(30, 5, 40, 18), 'id': 13},
        {'position': list2box(38, 5, 60, 20), 'id': 14},
        {'position': list2box(21, 20, 40, 30), 'id': 22},
    ]
    for box in boxes:
        box['__line__'] = -1  # 所在行号，-1表示未分配
    boxes[0]['__line__'] = 1
    # random.shuffle(boxes)
    outs = sort_and_filter_line_boxes(boxes, 'position')
    pprint(outs)


def test_sort_boxes():
    boxes = [
        {'position': list2box(0, 5, 20, 20), 'id': 11},
        {'position': list2box(30, 5, 40, 18), 'id': 13},
        {'position': list2box(38, 5, 60, 20), 'id': 14},
        {'position': list2box(20, 0, 30, 25), 'id': 12},
        {'position': list2box(0, 25, 20, 45), 'id': 21},
        {'position': list2box(40, 25, 60, 44), 'id': 23},
        {'position': list2box(21, 20, 40, 30), 'id': 22},
        {'position': list2box(2, 85, 30, 105), 'id': 41},
        {'position': list2box(10, 55, 50, 80), 'id': 31},
        {'position': list2box(35, 83, 58, 103), 'id': 42},
    ]

    line_boxes = sort_boxes(boxes, 'position')
    pprint(line_boxes)

    line_ids = []
    for boxes in line_boxes:
        line_ids.append([box['id'] for box in boxes])

    pprint(line_ids)
    assert line_ids == [
        [11, 12, 13, 14],
        [21, 22, 23],
        [31],
        [41, 42],
    ]
