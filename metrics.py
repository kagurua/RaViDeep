import numpy as np
import cv2
from scipy.spatial import ConvexHull


def cal_NOarea(original_hull):
    original_hull_a = np.concatenate([np.array([0., 0.]).reshape(1, 2), original_hull], axis=0)
    out_hull_o = list(ConvexHull(original_hull_a).vertices)
    out_hull_c = out_hull_o.index(0)
    out_hull_cl = out_hull_o[out_hull_c - 1]
    out_hull_cr = out_hull_o[(out_hull_c + 1) % len(out_hull_o)]
    NOarea = [original_hull_a[i, :] for i in range(len(original_hull_a)) if i not in out_hull_o]
    NOarea.append(original_hull_a[out_hull_cr])
    NOarea.append([0., 0.])
    NOarea.append(original_hull_a[out_hull_cl])
    NOarea = np.concatenate(NOarea, axis=0).reshape(-1, 2)
    return NOarea


def cal_contour_similarity(contour1: np.ndarray, contour2: np.ndarray):
    if contour1.ndim == 2:
        contour1 = contour1[:, np.newaxis, :]
    if contour2.ndim == 2:
        contour2 = contour2[:, np.newaxis, :]
    contour1 = (contour1 * 1000).astype(np.int32)
    contour2 = (contour2 * 1000).astype(np.int32)
    return cv2.matchShapes(contour1, contour2, 1, 0.0)


def cal_metric(targets, targets_gt):
    IoU_matrix = []
    points_num = 0
    for target in targets:
        NOarea_pred = target['noarea']
        points_num += target['points'].shape[0]
        for target_gt in targets_gt:
            NOarea_gt = target_gt['noarea']
            if target['class'] != 3 and target['class'] != target_gt['class']:
                IoU = 0.
            else:
                IoU = cal_iou(NOarea_pred, NOarea_gt)
            IoU_matrix.append(IoU)
    IoU_matrix = np.array(IoU_matrix).reshape([len(targets), len(targets_gt)])

    TP_NOO = []
    con_sim_list = []
    target_mark = [1 for _ in targets]
    for i in range(len(targets_gt)):
        poly_gt = targets_gt[i]['noarea']
        line = IoU_matrix[:, i]
        ti = np.argmax(line)
        if line[ti] > 0.1:
            target_mark[ti] = 0
            NOO = cal_iou(targets[ti]['noarea'], poly_gt)
            TP_NOO.append(NOO)
            con_sim = cal_contour_similarity(targets[ti]['hull'], targets_gt[i]['hull'])
            con_sim_list.append(con_sim)
        else:
            TP_NOO.append(0.)

    P_labels = [item['label'] for item in targets_gt]
    FP = sum([t['points'].shape[0] * flag for t, flag in zip(targets, target_mark)]) / points_num
    return TP_NOO, P_labels, FP, sum(target_mark), con_sim_list


def cal_iou(poly1, poly2):
    color = (1, 0, 0)
    color2 = (0, 1, 0)
    img = np.zeros([1000, 2000, 2])
    triangle1 = (poly1 + np.array([20, 0])) * 50
    triangle1 = triangle1.astype(int)
    cv2.fillConvexPoly(img, triangle1, color)
    area1 = img.sum()
    img = np.zeros([1000, 2000, 3])
    triangle2 = (poly2 + np.array([20, 0])) * 50
    triangle2 = triangle2.astype(int)
    cv2.fillConvexPoly(img, triangle2, color2)
    area2 = img.sum()
    cv2.fillConvexPoly(img, triangle1, color)
    union_area = img.sum()
    inter_area = area1 + area2 - union_area
    IOU = inter_area / union_area
    return IOU


def cal_overlap(poly1, poly2):
    color = (1, 0, 0)
    color2 = (0, 1, 0)
    img = np.zeros([1000, 2000, 2])
    triangle1 = (poly1 + np.array([20, 0])) * 50
    triangle1 = triangle1.astype(int)
    cv2.fillConvexPoly(img, triangle1, color)
    area1 = img.sum()
    img = np.zeros([1000, 2000, 3])
    triangle2 = (poly2 + np.array([20, 0])) * 50
    triangle2 = triangle2.astype(int)
    cv2.fillConvexPoly(img, triangle2, color2)
    area2 = img.sum()
    cv2.fillConvexPoly(img, triangle1, color)
    union_area = img.sum()
    inter_area = area1 + area2 - union_area
    overlap = inter_area / min(area1, area2)
    return overlap

