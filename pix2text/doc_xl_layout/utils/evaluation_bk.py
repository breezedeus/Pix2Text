import json
import os
import sys

import cv2
import numpy as np
from shapely.geometry import Polygon
from tabulate import tabulate
import time

def visual_badcase(image_name, pred_list, label_list, output_dir="visual_badcase", info=None, prefix=''):
    """
    """
    image_name = image_name + '.jpg'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.abspath('../../data/huntie/test_images/')
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    if img is None:
        print("--> Warning: skip, given image dir NOT exists: {}".format(image_path))
        return None

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(image_path)
    for label in label_list:
        points, class_id = label[:8], label[8]
        pts = np.array(points).reshape((1, -1, 2)).astype(np.int32)
        cv2.polylines(img, pts, isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.putText(img, "gt:" + str(class_id), tuple(pts[0][0].tolist()), font, 1, (0, 255, 0), 2)

    for label in pred_list:
        points, class_id = label[:8], label[8]
        pts = np.array(points).reshape((1, -1, 2)).astype(np.int32)
        cv2.polylines(img, pts, isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.putText(img, "pred:" + str(class_id), tuple(pts[0][-1].tolist()), font, 1, (255, 0, 0), 2)

    if info is not None:
        cv2.putText(img, str(info), (40, 40), font, 1, (0, 0, 255), 2)
    output_path = os.path.join(output_dir, prefix + os.path.basename(image_path))
    print("--> info: visualizing badcase: {}".format(output_path))
    cv2.imwrite(output_path, img)


def load_gt_from_json(json_path):
    """
    """
    with open(json_path) as f:
        gt_info = json.load(f)
    gt_image_list = gt_info["images"]
    gt_anno_list = gt_info["annotations"]

    id_to_image_info = {}
    for image_item in gt_image_list:
        id_to_image_info[image_item['id']] = {
            "file_name": image_item['file_name'],
            "group_name": image_item.get("group_name", "huntie")
        }

    group_info = {}
    for annotation_item in gt_anno_list:
        image_info = id_to_image_info[annotation_item['image_id']]
        image_name, group_name = image_info["file_name"], image_info["group_name"]

        if group_name not in group_info:
            group_info[group_name] = {}
        if image_name not in group_info[group_name]:
            group_info[group_name][image_name] = []
        anno_info = {
            "category_id": annotation_item["category_id"],
            "poly": annotation_item["poly"],
            "secondary_id": annotation_item.get("secondary_id", -1),
            "direction_id": annotation_item.get("direction_id", -1)
        }
        group_info[group_name][image_name].append(anno_info)

    group_info_str = ", ".join(["{}[{}]".format(k, len(v)) for k, v in group_info.items()])
    print("--> load {} groups: {}".format(len(group_info.keys()), group_info_str))
    return group_info

def save_res_to_file(table_head, table_body_sorted):
    with open('val_out.txt', 'a') as fout:
        fout.write(time.strftime('%Y-%m-%d-%H-%M') + '\n')
        fout.write('\t'.join(table_head) + '\n')
        for line in table_body_sorted:
            new_line = []
            for ele in line:
                if isinstance(ele, int):
                    new_line.append('{:d}'.format(ele))
                elif isinstance(ele, float):
                    new_line.append('{:.6f}'.format(ele))
                elif isinstance(ele, str):
                    new_line.append(ele)
            fout.write('\t'.join(new_line) + '\n')

def calc_iou(label, detect):
    label_box = []
    detect_box = []

    d_area = []
    for i in range(0, len(detect)):
        pred_poly = detect[i]["poly"]
        box_det = []
        for k in range(0, 4):
            box_det.append([pred_poly[2 * k], pred_poly[2 * k + 1]])
        detect_box.append(box_det)
        try:
            poly = Polygon(box_det)
            d_area.append(poly.area)
        except:
            print('invalid detects', pred_poly)
            exit(-1)

    l_area = []
    for i in range(0, len(label)):
        gt_poly = label[i]["poly"]
        box_gt = []
        for k in range(4):
            box_gt.append([gt_poly[2 * k], gt_poly[2 * k + 1]])
        label_box.append(box_gt)
        try:
            poly = Polygon(box_gt)
            l_area.append(poly.area)
        except:
            print('invalid detects', gt_poly)
            exit(-1)

    ol_areas = []
    for i in range(0, len(detect_box)):
        ol_areas.append([])
        poly1 = Polygon(detect_box[i])
        for j in range(0, len(label_box)):
            poly2 = Polygon(label_box[j])
            try:
                ol_area = poly2.intersection(poly1).area
            except:
                print('invaild pair', detect_box[i], label_box[j])
                ol_areas[i].append(0.0)
            else:
                ol_areas[i].append(ol_area)

    d_ious = [0.0] * len(detect_box)
    l_ious = [0.0] * len(label_box)
    det2label_idx = [-1] * len(detect_box) # 每个检测框iou最大标注框的index
    for i in range(0, len(detect_box)):
        for j in range(0, len(label_box)):
            if int(label[j]["category_id"]) == int(detect[i]["category_id"]):
                # iou = min(ol_areas[i][j] / (d_area[i] + 1e-10), ol_areas[i][j] / (l_area[j] + 1e-10))
                iou = ol_areas[i][j] / (d_area[i] + l_area[j] - ol_areas[i][j] + 1e-10)
            else:
                iou = 0
            det2label_idx[i] = j if iou > d_ious[i] else det2label_idx[i]
            d_ious[i] = max(d_ious[i], iou)
            l_ious[j] = max(l_ious[j], iou)
    return l_ious, d_ious, det2label_idx


def eval(instance_info):
    img_name, label_info = instance_info
    label = label_info['gt']
    detect = label_info['det']
    l_ious, d_ious, det2label_idx = calc_iou(label, detect)
    return [img_name, d_ious, l_ious, detect, label, det2label_idx]


def static_with_class(rets, iou_thresh=0.7, is_verbose=True, map_info=None):
    if is_verbose:
        table_head = ['Class_id', 'Class_name', 'Pre_hit', 'Pre_num', 'GT_hit', 'GT_num', 'Precision', 'Recall', 'F-score', 'All_recalled', 'Img_num', 'Acc.']
    else:
        table_head = ['Class_id', 'Class_name', 'Precision', 'Recall', 'F-score']
    table_body = []
    class_dict = {}
    all_dict = {} # 用以统计合计结果
    all_dict['dm'] = 0
    all_dict['dv'] = 0
    all_dict['lm'] = 0
    all_dict['lv'] = 0
    all_dict['Img_num'] = 0
    all_dict['All_recalled'] = 0

    no_need_keys = ['group_name', 'poly', 'score' , 'category_id']
    # import pdb; pdb.set_trace()
    extra_keys = [_ for _ in rets[0][4][0].keys() if _ not in no_need_keys]
    # extra_table_heads = [[_] for _ in rets[0][4][0].keys() if _ not in no_need_keys]
    extra_table_heads = {}
    extra_dict = {}
    extra_table_body = {}
    for key in extra_keys:
        extra_table_heads[key] = [key, 'Name', 'Pre_hit', 'Pre_num', 'GT_hit', 'GT_num', 'Precision', 'Recall', 'F-score']
        extra_dict[key] = {}
        extra_table_body[key] = []
        # _ += ['Pre_hit', 'Pre_num', 'GT_hit', 'GT_num', 'Precision', 'Recall', 'F-score']

    # pdb.set_trace()
    for i in range(len(rets)):
        img_name, d_ious, l_ious, detects, labels, det2label_idx = rets[i]
        item_lv, item_dv, item_dm, item_lm = 0, 0, 0, 0            
        current_dict = {}

        for label in labels:
            item_lv += 1
            category_id = label["category_id"]
            if category_id not in class_dict:
                class_dict[category_id] = {}
                class_dict[category_id]['dm'] = 0
                class_dict[category_id]['dv'] = 0
                class_dict[category_id]['lm'] = 0
                class_dict[category_id]['lv'] = 0
                class_dict[category_id]['Img_num'] = 0
                class_dict[category_id]['All_recalled'] = 0
            class_dict[category_id]['lv'] += 1

        category_container = []
        for label in labels:
            if label['category_id'] not in category_container:
                category_container.append(label['category_id'])
        for category_id in category_container:
            class_dict[category_id]['Img_num'] += 1
            current_dict[category_id] = {'dm':0, 'dv':0, 'lm':0, 'lv':0, 'Img_num':0, 'All_recalled':0}
        # 统计各额外key的id list和label、detect中检出的量
        for key in extra_keys:
            for label in labels:
                if label[key] not in extra_dict[key] and label[key] != -1:
                    extra_dict[key][label[key]] = {'dm':0, 'dv':0, 'lm':0, 'lv':0}
            for det in detects:
                if det[key] not in extra_dict[key] and det[key] != -1:
                    extra_dict[key][det[key]] = {'dm':0, 'dv':0, 'lm':0, 'lv':0}
            for label in labels:
                if label[key] != -1:
                    extra_dict[key][label[key]]['lv'] += 1
            for det in detects:
                if det[key] != -1:
                    try:
                        extra_dict[key][det[key]]['dv'] += 1
                    except:
                        import pdb; pdb.set_trace()

        for label in labels:
            current_dict[label['category_id']]['lv'] += 1
        for det in detects:
            current_dict[label['category_id']]['dv'] += 1

        for det in detects:
            item_dv += 1
            category_id = det["category_id"]
            if category_id not in class_dict:
                print("--> category_id not exists in gt: {}".format(category_id))
                continue
            class_dict[category_id]['dv'] += 1

        for idx, iou in enumerate(d_ious):
            if iou >= iou_thresh:
                item_dm += 1
                class_dict[detects[idx]["category_id"]]['dm'] += 1
                current_dict[detects[idx]["category_id"]]['dm'] += 1

                for key in extra_keys:
                    if labels[det2label_idx[idx]][key] != -1 and detects[idx][key] == labels[det2label_idx[idx]][key]:
                        extra_dict[key][detects[idx][key]]['dm'] += 1
                        extra_dict[key][detects[idx][key]]['lm'] += 1
                        
        for idx, iou in enumerate(l_ious):
            if iou >= iou_thresh:
                item_lm += 1
                class_dict[labels[idx]["category_id"]]['lm'] += 1
                current_dict[labels[idx]["category_id"]]['lm'] += 1


        
        # 将recall append到结果list当中
        item_r = item_lm / (item_lv + 1e-6)
        # item_p = item_dm / (item_dv + 1e-6)
        # item_f = 2 * item_p * item_r / (item_p + item_r + 1e-6)
        # if (1 - item_r) < 1e-5:
        #     class_dict[category_id]['All_recalled'] += 1
        rets[i].append(item_r)

        # 计算各个box类别全召回率
        for category_id in category_container:
            id_recall = current_dict[category_id]['lm'] / (current_dict[category_id]['lv'] + 1e-6)
            if (1 - id_recall) < 1e-5:
                class_dict[category_id]['All_recalled'] += 1

        # 计算所有类别总计的全召回率
        # import pdb; pdb.set_trace()
        all_dict['dv'] += item_dv
        all_dict['lv'] += item_lv
        all_dict['dm'] += item_dm
        all_dict['lm'] += item_lm
        all_dict['Img_num'] += 1
        if (1 - item_r) < 1e-5:
            all_dict['All_recalled'] += 1
        
        # if img_name == 'train10w_val2w_69008cd9828a455fb1bf751a95ad8921.jpg':
        #     import pdb; pdb.set_trace()
        # if item_r
        # if item_f < 0.97 and is_save_badcase:
        #     prefix = '_'.join(map(str, sorted(list(badcase_class_set)))) + '_'
        #     item_info = "IOU{}, {}, {}, {}".format(iou_thresh, item_r, item_p, item_f)
        #     visual_badcase(img_name, detects, labels, output_dir="visual_badcase", info=item_info, prefix=prefix)

    dm, dv, lm, lv, total, recalled = 0, 0, 0, 0, 0, 0
    map_info = {} if map_info is None else map_info
    for key in class_dict.keys():
        dm += class_dict[key]['dm']
        dv += class_dict[key]['dv']
        lm += class_dict[key]['lm']
        lv += class_dict[key]['lv']
        recalled += class_dict[category_id]['All_recalled']
        total += class_dict[key]['Img_num']
        p = class_dict[key]['dm'] / (class_dict[key]['dv'] + 1e-6)
        r = class_dict[key]['lm'] / (class_dict[key]['lv'] + 1e-6)
        fscore = 2 * p * r / (p + r + 1e-6)
        acc = class_dict[key]['All_recalled'] / (class_dict[key]['Img_num'] + 1e-6)
        if is_verbose:
            table_body.append((key, map_info.get("primary_map", {}).get(str(key), str(key)), class_dict[key]['dm'],
                               class_dict[key]['dv'], class_dict[key]['lm'], class_dict[key]['lv'], p, r, fscore,
                               class_dict[category_id]['All_recalled'], class_dict[key]['Img_num'], acc))
        else:
            table_body.append((key,  map_info.get(str(key), str(key)), p, r, fscore))

    p = dm / (dv + 1e-6)
    r = lm / (lv + 1e-6)
    f = 2 * p * r / (p + r + 1e-6)
    acc = recalled / (total + 1e-6)
    table_body_sorted = sorted(table_body, key=lambda x: int((x[0])))
    if is_verbose:
        table_body_sorted.append(('IOU_{}'.format(iou_thresh), 'average', dm, dv, lm, lv, p, r, f,
                                  all_dict['All_recalled'], all_dict['Img_num'], (all_dict['All_recalled']/all_dict['Img_num']+1e-6)))
    else:
        table_body_sorted.append(('IOU_{}'.format(iou_thresh), 'average', p, r, f))
    # import pdb; pdb.set_trace()
    save_res_to_file(table_head, table_body_sorted)
    print(tabulate(table_body_sorted, headers=table_head, tablefmt='pipe'))
    # ---------------print(extra_keys)
    for _key in extra_dict.keys():
        dm, dv, lm, lv = 0, 0, 0, 0
        for key in extra_dict[_key].keys():
            dm += extra_dict[_key][key]['dm']
            dv += extra_dict[_key][key]['dv']
            lm += extra_dict[_key][key]['lm']
            lv += extra_dict[_key][key]['lv']
            # 找当前key对应的map_info key的name
            map_name = ''
            for candidate_name in map_info.keys():
                if candidate_name.split('_')[0] == _key.split('_')[0]:
                    map_name = candidate_name

            precision = extra_dict[_key][key]['dm'] / (extra_dict[_key][key]['dv'] + 1e-6)
            recall = extra_dict[_key][key]['lm'] / (extra_dict[_key][key]['lv'] + 1e-6)
            fscore = 2 * precision * recall / (precision + recall + 1e-6)
            if map_name == '': # 没有在map_info中找到对应类表
                extra_table_body[_key].append((key, '', extra_dict[_key][key]['dm'], extra_dict[_key][key]['dv'],
                                                extra_dict[_key][key]['lm'], extra_dict[_key][key]['lv'],
                                            precision, recall, fscore))
            else:
                extra_table_body[_key].append((key, map_info.get(map_name, {}).get(str(key), str(key)), extra_dict[_key][key]['dm'], extra_dict[_key][key]['dv'],
                                                extra_dict[_key][key]['lm'], extra_dict[_key][key]['lv'],
                                            precision, recall, fscore))                
        extra_table_body[_key] = sorted(extra_table_body[_key], key=lambda x: int((x[0])))
        p = dm / (dv + 1e-6)
        r = lm / (lv + 1e-6)
        f = 2 * p * r / (p + r + 1e-6)
        extra_table_body[_key].append((key, 'average', dm, dv, lm, lv, p, r, f))
    # import pdb; pdb.set_trace()
    for _key in extra_keys:
        save_res_to_file(extra_table_heads[_key], extra_table_body[_key])
        print(tabulate(extra_table_body[_key], headers=extra_table_heads[_key], tablefmt='pipe'))

    return [table_head] + table_body_sorted


def multiproc(func, task_list, proc_num=30, retv=True, progress_bar=False):
    from multiprocessing import Pool
    pool = Pool(proc_num)

    rets = []
    if progress_bar:
        import tqdm
        with tqdm.tqdm(total=len(task_list)) as t:
            for ret in pool.imap(func, task_list):
                rets.append(ret)
                t.update(1)
    else:
        for ret in pool.imap(func, task_list):
            rets.append(ret)

    pool.close()
    pool.join()

    if retv:
        return rets


def eval_and_show(label_dict, detect_dict, output_dir, iou_thresh=0.7, map_info=None):
    """
    """
    evaluation_group_info = {}
    for group_name, gt_info in label_dict.items():
        group_pair_list = []
        for file_name, value_list in gt_info.items():
            if file_name not in detect_dict:
                # print("--> missing pred:", file_name)
                continue
            group_pair_list.append([file_name, {'gt': gt_info[file_name], 'det': detect_dict[file_name]}])
        evaluation_group_info[group_name] = group_pair_list

    res_info_all = {}
    for group_name, group_pair_list in evaluation_group_info.items():
        print(" ------- group name: {} -----------".format(group_name))
        rets = multiproc(eval, group_pair_list, proc_num=16)
        # import pdb; pdb.set_trace()
        group_name_map_info = map_info.get(group_name, None) if map_info is not None else None
        res_info = static_with_class(rets, iou_thresh=iou_thresh, map_info=group_name_map_info)
        res_info_all[group_name] = res_info

    evaluation_res_info_path = os.path.join(output_dir, "results_val.json")
    with open(evaluation_res_info_path, "w") as f:
        json.dump(res_info_all, f, ensure_ascii=False, indent=4)
    print("--> info: evaluation result is saved at {}".format(evaluation_res_info_path))
    return rets

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python {} gt_json_path pred_json_path output_dir  iou_thresh".format(__file__))
        exit(-1)
    else:
        print('--> info: {}'.format(sys.argv))
        gt_json_path, pred_json_path, output_dir, iou_thresh = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])

    label_dict = load_gt_from_json(gt_json_path)
    with open(pred_json_path, "r") as f:
        detect_dict = json.load(f)
    res_info = eval_and_show(label_dict, detect_dict, output_dir, iou_thresh=iou_thresh, map_info=None)

