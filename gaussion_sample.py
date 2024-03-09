import numpy as np
import math


def cal_weight_gauss(x, y, sigma_x, sigma_y):
    hx = np.exp(-(x * x) / (2 * sigma_x * sigma_x)) / (math.sqrt(2 * math.pi) * sigma_x)
    hy = np.exp(-(y * y) / (2 * sigma_y * sigma_y)) / (math.sqrt(2 * math.pi) * sigma_y)
    h = hx @ hy
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def sample_img_depth(uvd, depth_map, sigma_uv, radius=1, seg_mask=None, seg_map=None):
    height, width = depth_map.shape
    weight_list = []
    depth_list = []
    radar_depth_list = []
    point_seg_list = []
    for loc, loc_sigma in zip(uvd, sigma_uv):  # uv/sigma_uv: (n, 2)
        u, v, d = int(loc[0]), int(loc[1]), loc[2]
        sigma_u, sigma_v = int(loc_sigma[0]), int(loc_sigma[1])
        radius_u, radius_v = sigma_u * radius, sigma_v * radius

        left, right = min(u, radius_u), min(width - u, radius_u + 1)
        top, bottom = min(v, radius_v), min(height - v, radius_v + 1)

        sample_gap = 2
        v_u, u_u = np.ogrid[0:bottom:sample_gap, 0:right:sample_gap]
        v_d, u_d = np.ogrid[0:-top - 1:-sample_gap, 0:-left - 1:-sample_gap]
        v_d = v_d[::-1, :][:-1, :]
        u_d = u_d[:, ::-1][:, :-1]
        v_s0 = np.concatenate([v_d, v_u], axis=0)  # m, 1
        u_s0 = np.concatenate([u_d, u_u], axis=1)  # 1, n
        sampled_gauss_weight = cal_weight_gauss(v_s0, u_s0, sigma_v, sigma_u)

        # sample depth_map
        v_s = v_s0.reshape(-1) + v
        u_s = u_s0.reshape(-1) + u
        sampled_depth_map = depth_map[v_s[0]:v_s[-1] + 1:sample_gap, u_s[0]:u_s[-1] + 1:sample_gap]

        depth = sampled_depth_map.flatten()
        weight = sampled_gauss_weight.flatten()
        radar_dpeth = np.ones_like(depth) * d

        if seg_mask is not None:
            ui, vi = np.meshgrid(u_s, v_s)
            point_seg_mask = seg_mask[vi.flatten(), ui.flatten()]
            depth = depth[point_seg_mask == 1]
            weight = weight[point_seg_mask == 1]
            radar_dpeth = radar_dpeth[point_seg_mask == 1]
            if seg_map is not None:
                point_seg = seg_map[vi.flatten(), ui.flatten()]
                point_seg = point_seg[point_seg_mask == 1]
                point_seg_list.append(point_seg)

        depth_list.append(depth)
        weight_list.append(weight)
        radar_depth_list.append(radar_dpeth)
    if seg_map is None:
        return np.concatenate(radar_depth_list), np.concatenate(depth_list), np.concatenate(weight_list)
    else:
        return np.concatenate(radar_depth_list), np.concatenate(depth_list), np.concatenate(weight_list), \
               np.concatenate(point_seg_list)
