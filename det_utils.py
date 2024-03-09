import numpy as np
import sklearn.cluster as skc
from scipy.spatial import ConvexHull
from metrics import cal_metric, cal_NOarea
from utils import restore_radar_points


label_color_map = {0: 'brown', 1: 'magenta', 2: 'yellow', 3: 'black'}


def points_filter(points, frame_ids=None, compensate=None, soft_labels=None):
    mask_rm_noise = np.linalg.norm(points[:, 0:3], axis=1) > rm_range
    mask_time = frame_ids > (np.max(frame_ids) - time_lenth)
    mask = np.bitwise_and(mask_rm_noise, mask_time)
    raw_points = restore_radar_points(points[:, :3], compensate)
    points = points[mask, :]
    raw_points = raw_points[mask, :]
    raw_points = np.concatenate([raw_points, points[:, 3:]], axis=1)
    if soft_labels is not None:
        soft_labels = soft_labels[mask]
    return points, raw_points, soft_labels


def estimate(data, params):
    eps, min_samples = params
    estimator = skc.DBSCAN(eps=eps, min_samples=min_samples)
    points = data[, :2]
    estimator.fit(points)
    labels = estimator.labels_
    return labels


def preprocess_predictions(points, labels, ax, show_level=1):
    if show_level >= 1:
        noises = points[labels == -1]
        ax.plot(noises[:, 0], noises[:, 1], 'o', c='#000000', markersize=0.5)
    targets = []
    for i in np.unique(labels):
        if i == -1:
            continue
        one_cluster = points[labels == i]
        try:
            hull = ConvexHull(one_cluster[:, :2])
        except:
            continue
        target_dimension = one_cluster[hull.vertices, :2]
        target_class = np.mean(one_cluster[:, 3:], axis=0)
        target_center = np.mean(one_cluster[:, :2], axis=0)
        target = {'hull': target_dimension, 'location': target_center, 'points': one_cluster[:, :2],
                  'class': np.argmax(target_class)}
        if np.max(target_class) < 0.1:
            target['class'] = 3
        if show_level >= 1:
            ax.plot(target['points'][:, 0], target['points'][:, 1], 'o', color=f'{label_color_map[target["class"]]}',
                    markersize=0.5)
            ax.plot(target['location'][0], target['location'][1], '.r', markersize=0.5)
            for simplex in hull.simplices:
                ax.plot(one_cluster[simplex, 0], one_cluster[simplex, 1], 'b-')
        NOarea = cal_NOarea(target['hull'])
        target['noarea'] = NOarea
        targets.append(target)
    return targets


def preprocess_targets(truth_points, ax, show_level=1):
    targets_gt = []
    labels_gt = truth_points[:, -1]
    for label_gt in set(labels_gt):
        one_cluster_gt = truth_points[labels_gt == label_gt, :]
        if one_cluster_gt.shape[0] < 3:
            continue
        hull_gt = ConvexHull(one_cluster_gt[:, :2])
        target_gt_dimension = one_cluster_gt[hull_gt.vertices, :2]
        target_gt_class = int(one_cluster_gt[0, 2])
        target_gt_center = np.mean(one_cluster_gt[:, :2], axis=0)
        NOarea_gt = cal_NOarea(target_gt_dimension)
        target_gt = {'hull': target_gt_dimension, 'location': target_gt_center, 'points': one_cluster_gt[:, :2],
                     'class': target_gt_class, 'noarea': NOarea_gt, 'label': label_gt}
        targets_gt.append(target_gt)
        if show_level >= 1:
            for simplex in hull_gt.simplices:
                ax.plot(one_cluster_gt[simplex, 0], one_cluster_gt[simplex, 1], 'g-', markersize=0.5)
    return targets_gt


def current_detection(targets, targets_gt):
    noo_score = -1 * np.ones(len(targets_gt['label']))
    noos, p_label, fp, false_positive_num, con_sims = cal_metric(targets, targets_gt)
    true_positive_num = sum(1 for noo in noos if noo > 0)
    targets_num = len(noos)
    preds_num = true_positive_num + false_positive_num
    for target_id, noo in zip(p_label, noos):
        noo_score[int(target_id)] = noo
    false_alarm_rate = fp
    return noo_score, false_alarm_rate, true_positive_num, targets_num, preds_num, con_sims
