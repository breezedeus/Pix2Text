import math
from shapely.geometry import Polygon
from functools import cmp_to_key


def calc_main_angle(pts_list):
    if len(pts_list) == 0:
        return 0
    good_angles, other_angles = [], []
    for pts in pts_list:
        d_x_1, d_y_1 = pts[2] - pts[0], pts[3] - pts[1]
        d_x_2, d_y_2 = pts[4] - pts[2], pts[5] - pts[3]

        width = math.sqrt(d_x_1 ** 2 + d_y_1 ** 2)
        height = math.sqrt(d_x_2 ** 2 + d_y_2 ** 2)
        angle = math.atan2(d_y_1, d_x_1)

        if width > height * 3:
            good_angles.append(angle)
        else:
            other_angles.append(angle)

    if len(good_angles) > 0:
        good_angles.sort()
        return good_angles[len(good_angles) // 2]
    else:
        other_angles.sort()
        return other_angles[len(other_angles) // 2]


def calc_x_type(a, b):
    x_type = 0
    minx_a, maxx_a = a[0], a[0] + a[2]
    minx_b, maxx_b = b[0], b[0] + b[2]

    start_left = 0
    if minx_a < minx_b:
        start_left = 1
    elif minx_a > minx_b:
        start_left = -1
    end_right = 0
    if maxx_a > maxx_b:
        end_right = 1
    elif maxx_a < maxx_b:
        end_right = -1

    if maxx_a < minx_b + 1e-4 and maxx_a < maxx_b - 1e-4:
        x_type = 1  # left
    elif minx_a > maxx_b - 1e-4 and minx_a > minx_b + 1e-4:
        x_type = 2  # right
    elif start_left == 1 and end_right == -1:
        x_type = 3  # near left
    elif start_left == -1 and end_right == 1:
        x_type = 4  # near right
    elif start_left >= 0 and end_right >= 0:
        x_type = 5  # contain
    elif start_left <= 0 and end_right <= 0:
        x_type = 6  # inside
    else:
        x_type = 0

    return x_type


def calc_y_type(a, b):
    y_type = 0
    miny_a, maxy_a = a[1], a[1] + a[3]
    miny_b, maxy_b = b[1], b[1] + b[3]

    start_up = 0
    if miny_a < miny_b:
        start_up = 1
    elif miny_a > miny_b:
        start_up = -1
    end_down = 0
    if maxy_a > maxy_b:
        end_down = 1
    elif maxy_a < maxy_b:
        end_down = -1

    if maxy_a < miny_b + 1e-4 and maxy_a < maxy_b - 1e-4:
        y_type = 1  # up
    elif miny_a > maxy_b - 1e-4 and miny_a > miny_b + 1e-4:
        y_type = 2  # down
    elif start_up == 1 and end_down == -1:
        y_type = 3  # near up
    elif start_up == -1 and end_down == 1:
        y_type = 4  # near down
    elif start_up >= 0 and end_down >= 0:
        y_type = 5  # contain
    elif start_up <= 0 and end_down <= 0:
        y_type = 6  # inside
    else:
        y_type = 0

    return y_type


def sort_pts(blocks):
    main_angle = calc_main_angle([blk['pts'] for blk in blocks])
    main_sin, main_cos = math.sin(main_angle), math.cos(main_angle)

    def pts2rect(pts):
        xs, ys = [], []
        for k in range(0, len(pts), 2):
            x0 = pts[k] * main_cos + pts[k + 1] * main_sin
            y0 = pts[k + 1] * main_cos - pts[k] * main_sin
            xs.append(x0)
            ys.append(y0)
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        rect = [minx, miny, maxx - minx, maxy - miny]
        # print('===', pts, '->', rect)
        return rect

    def cmp_pts_udlr(a, b, thres=0.5):
        rect_a, rect_b = pts2rect(a['pts']), pts2rect(b['pts'])
        minx_a, miny_a, maxx_a, maxy_a = (
            rect_a[0],
            rect_a[1],
            rect_a[0] + rect_a[2],
            rect_a[1] + rect_a[3],
        )
        minx_b, miny_b, maxx_b, maxy_b = (
            rect_b[0],
            rect_b[1],
            rect_b[0] + rect_b[2],
            rect_b[1] + rect_b[3],
        )

        x_type, y_type = calc_x_type(rect_a, rect_b), calc_y_type(rect_a, rect_b)

        y_near_rate = 0.0
        if y_type == 3:
            y_near_rate = (maxy_a - miny_b) / min(maxy_a - miny_a, maxy_b - miny_b)
        elif y_type == 4:
            y_near_rate = (maxy_b - miny_a) / min(maxy_a - miny_a, maxy_b - miny_b)

        # print(rect_a, rect_b, x_type, y_type, y_near_rate)
        # exit(0)

        if y_type == 1:
            return -1
        elif y_type == 2:
            return 1
        elif y_type == 3:
            if x_type in [2, 4]:
                if y_near_rate < thres:
                    return -1
                else:
                    return 1
            else:
                return -1
        elif y_type == 4:
            if x_type in [1, 3]:
                if y_near_rate < thres:
                    return 1
                else:
                    return -1
            else:
                return 1
        else:
            if x_type == 1 or x_type == 3:
                return -1
            elif x_type == 2 or x_type == 4:
                return 1
            else:
                center_y_diff = abs(0.5 * (miny_a + maxy_a) - 0.5 * (miny_b + maxy_b))
                max_h = max(maxy_a - miny_a, maxy_b - miny_b)
                if center_y_diff / max_h < 0.1:
                    if (minx_a + maxx_a) < (minx_b + maxx_b):
                        return -1
                    elif (minx_a + maxx_a) > (minx_b + maxx_b):
                        return 1
                    else:
                        return 0
                else:
                    if (miny_a + maxy_a) < (miny_b + maxy_b):
                        return -1
                    elif (miny_a + maxy_a) > (miny_b + maxy_b):
                        return 1
                    else:
                        return 0

    # print(blocks)
    # print(cmp_pts_udlr(blocks[0], blocks[1]))
    blocks.sort(key=cmp_to_key(cmp_pts_udlr))
    # print(blocks)
    # exit(0)


def pts2poly(pts):
    new_pts = [(pts[k], pts[k + 1]) for k in range(0, len(pts), 2)]
    return Polygon(new_pts)


def pts_intersection_rate(src, tgt):
    src_poly, tgt_poly = pts2poly(src), pts2poly(tgt)
    src_area = src_poly.area
    inter_area = src_poly.intersection(tgt_poly).area
    return inter_area / src_area


def wrap_result(layout_detection_info, subfield_detection_info, category_map):
    if layout_detection_info is None or subfield_detection_info is None:
        return {}
    # layout_detection_info = result["layout_dets"]
    # subfield_detection_info = result["subfield_dets"]

    info = {'subfields': []}
    for itm in subfield_detection_info:
        subfield = {
            'category': category_map[itm['category_id']],
            'pts': itm['poly'],
            'confidence': itm['score'],
            'layouts': [],
        }
        info['subfields'].append(subfield)
    sort_pts(info['subfields'])

    if len(info['subfields']) > 0:
        other_subfield = {
            'category': '其他',
            'pts': [0, 0, 0, 0, 0, 0, 0, 0],
            'confidence': 0,
            'layouts': [],
        }
        for itm in layout_detection_info:
            layout = {
                'category': category_map[itm['category_id']],
                'pts': itm['poly'],
                'confidence': itm['score'],
            }
            best_rate, best_idx = 0.0, -1
            for k in range(len(info['subfields'])):
                inter_rate = pts_intersection_rate(
                    layout['pts'], info['subfields'][k]['pts']
                )
                if inter_rate > best_rate:
                    best_rate = inter_rate
                    best_idx = k
            if best_idx >= 0 and best_rate > 0.1:
                info['subfields'][best_idx]['layouts'].append(layout)
            else:
                other_subfield['layouts'].append(layout)
        if len(other_subfield['layouts']) > 0:
            info['subfields'].append(other_subfield)
    else:
        subfield = {
            'category': '其他',
            'pts': [0, 0, 0, 0, 0, 0, 0, 0],
            'confidence': 0,
            'layouts': [],
        }
        info['subfields'].append(subfield)
        for itm in layout_detection_info:
            layout = {
                'category': category_map[itm['category_id']],
                'pts': itm['poly'],
                'confidence': itm['score'],
            }
            info['subfields'][0]['layouts'].append(layout)

    for subfield in info['subfields']:
        sort_pts(subfield['layouts'])

    new_subfields = []
    for subfield in info['subfields']:
        if subfield['category'] != '其他':
            new_subfields.append(subfield)
        else:
            for layout in subfield['layouts']:
                layout_subfield = {
                    'category': layout['category'],
                    'pts': layout['pts'],
                    'confidence': layout['confidence'],
                    'layouts': [layout],
                }
                new_subfields.append(layout_subfield)
    sort_pts(new_subfields)
    info['layouts'] = []
    for subfield in new_subfields:
        for layout in subfield['layouts']:
            info['layouts'].append(layout)

    return info
