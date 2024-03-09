import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.cluster as skc  # 密度聚类
from utils import get_points_semantic_ids, get_uv, comp_xy0


class_names = ['sea', 'sky', 'shore', 'ship', 'pillar', 'bank', 'background']
class_names.insert(0, 'unseen')


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def collect_point2img_feature(radar_points, H, img_seg_maps, height, width):
    uv = get_uv(radar_points, H)  # [2, n]
    points_semantic_ids, points_soft_semantics = get_points_semantic_ids(uv, img_seg_maps, height, width)
    mask = np.logical_and(np.ones(radar_points.shape[1]), [1, 1, 1, 1, 0, 0, 0, 0, 0])  # select out xyzv
    xyzv = radar_points[:, mask]
    xyzvs = np.concatenate([xyzv, points_semantic_ids.reshape([-1, 1])], axis=1)
    return xyzvs, points_soft_semantics


def convert_point_feature(xyzvs, USE_COMP=False, soft_labels=None):
    mask1 = np.linalg.norm(xyzvs[:, 0:3], axis=1) > rm_range
    compensate = np.ones_like(xyzvs[:, :3])
    if USE_COMP:
        xy0, compensate = comp_xy0(xyzvs[:, :3])
    else:
        xy = xyzvs[:, 0:2]
        xy0 = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
    xyz = xyzvs[:, 0:3]

    s = xyzvs[:, -7:]
    xyzs = np.concatenate([xyz, s], axis=1)
    xy0s = np.concatenate([xy0, s], axis=1)
    # cut out points
    cut_xyzs = xyzs[mask1, :]
    cut_xy0s = xy0s[mask1, :]
    cut_compensate = compensate[mask1, :]
    if soft_labels is not None:
        soft_labels = soft_labels[mask1, :]
    return cut_xy0s, cut_xyzs, cut_compensate, soft_labels


def to_cluster_feature(xyzs, labels_frame, eps=0.8, min_samples=3, USE_WEIGHT=False, show_level=0):
    frame2weight = []
    frame_set = np.unique(labels_frame)
    cur_weight = 0.1
    for _ in frame_set:
        frame2weight.append(cur_weight)
        cur_weight += 0.1
    frame2weight.reverse()
    db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(xyzs[:, :3])
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    cluster_feature = []
    cluster_xyz = []
    for i in range(n_clusters):
        if USE_WEIGHT:
            # using weight
            one_cluster = xyzs[labels == i, :]
            one_cluster_xyz = one_cluster[:, :3]
            one_cluster_frame = labels_frame[labels == i]
            one_cluster_weight = [frame2weight[int(id)] for id in one_cluster_frame]
            one_cluster_s = np.average(one_cluster[:, 3:], axis=0, weights=one_cluster_weight).reshape(1, 3)
            one_cluster_s = np.dot(np.ones(one_cluster.shape[0]).reshape(-1, 1), one_cluster_s)
            one_cluster_c = np.average(one_cluster_xyz, axis=0, weights=one_cluster_weight).reshape(1, 3)
            one_cluster_c = np.dot(np.ones(one_cluster.shape[0]).reshape(-1, 1), one_cluster_c)
            one_cluster_dxyz = one_cluster_xyz - one_cluster_c
        else:
            # not using weight
            one_cluster = xyzs[labels == i, :]
            one_cluster_s = np.dot(np.ones(one_cluster.shape[0]).reshape(-1, 1),
                                   np.mean(one_cluster[:, 3:], axis=0).reshape(1, 3))
            one_cluster_xyz = one_cluster[:, :3]
            one_cluster_c = np.dot(np.ones(one_cluster.shape[0]).reshape(-1, 1),
                                   np.mean(one_cluster_xyz, axis=0).reshape(1, 3))
            one_cluster_dxyz = one_cluster_xyz - one_cluster_c

        one_cluster_feature = np.concatenate([one_cluster_xyz, one_cluster_dxyz, one_cluster_c, one_cluster_s], axis=1)
        cluster_feature.append(one_cluster_feature)
        cluster_xyz.append(one_cluster_c)
    cluster_feature = np.concatenate(cluster_feature, axis=0)
    cluster_xyz = np.concatenate(cluster_xyz, axis=0)

    # SHOW draw points with semantic info
    if show_level >= 2:
        fig, axes = plt.subplots(1, 1)
        plt.cla()
        plt.plot(xyzs[:, 0], xyzs[:, 1], 'o', c='#000000')
        for i in range(n_clusters):
            one_cluster = xyzs[labels == i]
            plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
            plt.plot(np.mean(one_cluster[:, :3], axis=0)[0], np.mean(one_cluster[:, :3], axis=0)[1], '.r')
        axes.set_aspect('equal')
        plt.xlim([-10, 10])
        plt.ylim([0, 20])
        plt.title(f'filter-before dbscan result')
        plt.show()

    return cluster_xyz, cluster_feature
