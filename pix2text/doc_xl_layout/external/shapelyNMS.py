import numpy as np


def pnms(dets, thresh):
    if len(dets) < 2:
        return dets
    scores = dets[:, 8]
    index_keep = []
    keep = []
    for i in range(len(dets)):
        box = dets[i]
        if box[8] < thresh:
            continue
        max_score_index = -1
        ctx = (dets[i][0] + dets[i][2] + dets[i][4] + dets[i][6]) / 4
        cty = (dets[i][1] + dets[i][3] + dets[i][5] + dets[i][7]) / 4
        for j in range(len(dets)):
            if i == j or dets[j][8] < thresh:
                continue
            x1, y1 = dets[j][0], dets[j][1]
            x2, y2 = dets[j][2], dets[j][3]
            x3, y3 = dets[j][4], dets[j][5]
            x4, y4 = dets[j][6], dets[j][7]
            a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
            b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
            c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
            d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
            if ((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                if dets[i][8] > dets[j][8] and max_score_index < 0:
                    max_score_index = i
                elif dets[i][8] < dets[j][8]:
                    max_score_index = -2
                    break
        if max_score_index > -1:
            index_keep.append(max_score_index)
        elif max_score_index == -1:
            index_keep.append(i)
    for i in range(0, len(index_keep)):
        keep.append(dets[index_keep[i]])

    return np.array(keep)

    '''
    pts = []
    for i in range(dets.shape[0]):
        pts.append([dets[i][0:2],dets[i][2:4],dets[i][4:6],dets[i][6:8]])

    areas = np.zeros(scores.shape)
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0],scores.shape[0]))
    
    for i in range(0,len(pts)):
        poly = Polygon(pts[i])
        areas[i] = poly.area
    
        for j in range(i, len(pts)):
            polyj = Polygon(pts[j])
            try:
                inS = poly.intersection(polyj)
            except Exception as e:
                print(pts[i],'\n',pts[j])
                return dets
            inter_areas[i][j] = inS.area
            inter_areas[j][i] = inS.area

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(dets[i])
        ovr = inter_areas[i][order[1:]] / (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
 
    return keep
    '''
