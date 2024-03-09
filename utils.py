import numpy as np
import math
import json
from shapely.geometry import Polygon, MultiPoint
from scipy.spatial import ConvexHull


with open('./data/calib.txt', 'r') as f:
    calib = json.load(f)


def estimate_ego_v(data):
    return - np.matmul(np.linalg.pinv(data[:, :3]), data[:, 3:4])


def cal_v_comp_all(radar_frame):
    xyz = radar_frame[:, :3]
    v = radar_frame[:, 3:4]
    xyz_normal = xyz / np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    xyzv = np.concatenate([xyz_normal, v], axis=1)

    v_boat = estimate_ego_v(xyzv)
    v_rela = - np.matmul(xyz_normal, v_boat)
    return v_boat


def get_H():
    return np.dot(np.array(calib['Intrinsic']), np.array(calib['Extrinsic']))


def get_IRT():
    return np.array(calib['Intrinsic']), np.array(calib['Rotation']), np.array(calib['Translation']),


def gen_R(thetas):
    Rx = np.array([1, 0, 0, 0, np.cos(thetas[0]), np.sin(thetas[0]), 0, -np.sin(thetas[0]), np.cos(thetas[0])]).reshape((3, 3))
    Ry = np.array([np.cos(thetas[1]), 0, -np.sin(thetas[1]), 0, 1, 0, np.sin(thetas[1]), 0, np.cos(thetas[1])]).reshape((3, 3))
    Rz = np.array([np.cos(thetas[2]), -np.sin(thetas[2]), 0, np.sin(thetas[2]), np.cos(thetas[2]), 0, 0, 0, 1]).reshape((3, 3))
    R = np.dot(Rx, Ry, Rz)
    return R


def comp_xy0(radar_points):
    thetas = calib['thetas']
    R = gen_R(thetas)

    p = radar_points[:, :3]
    r = np.array(
        [math.sin(thetas[1]) * math.cos(thetas[0]), - math.sin(thetas[0]), math.cos(thetas[1]) * math.cos(thetas[0])])
    compensate = np.dot(p, r).reshape(-1, 1) @ r.reshape(1, -1)
    p = p - compensate
    p = p @ R.T
    return p, compensate


def restore_radar_points(p, C):
    thetas = calib['thetas']
    R = gen_R(thetas)
    p_prime = np.dot(p, R)
    radar_points_restored = p_prime + C
    return radar_points_restored


def get_uv(radar_points, H):
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    uv = np.floor(uv1[0:2, :]).astype(int)
    return uv


def get_uvd(radar_points, H, video_frame=None):
    # project xyz to uv
    xyz = radar_points[:, 0:3].T
    xyz1 = np.concatenate([xyz, np.ones([1, xyz.shape[1]])])
    uv1 = np.dot(H, np.dot(xyz1, np.diag(1. / (1e-5 + np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))))))
    uv = uv1[0:2, :]
    d = np.dot(np.array([0, 0, 1]), np.dot(H, xyz1))[None, :]  # (1, n)
    uvd = np.concatenate([uv, d])
    return uvd  # (3, n)


def get_points_semantic_ids(uv, img_seg_maps, height, width):
    point_num = int(uv.shape[1])
    point_ids = np.arange(point_num)
    points_semantic_ids = np.zeros(point_num)
    points_soft_semantics = np.zeros((point_num, 7))
    points_semantic_ids[uv[0, :] >= width] = -1
    points_semantic_ids[uv[0, :] <= 0] = -1
    points_semantic_ids[uv[1, :] >= height] = -1
    points_semantic_ids[uv[1, :] <= 0] = -1
    point_ids = point_ids[points_semantic_ids == 0]
    uv = uv[:, points_semantic_ids == 0]
    semantic_id_probs = img_seg_maps[:, uv[1, :], uv[0, :]].reshape([7, -1]).T
    semantic_ids = np.argmax(semantic_id_probs, axis=1)
    points_semantic_ids[point_ids] = semantic_ids
    e_x = np.exp(semantic_id_probs - np.max(semantic_id_probs, axis=1, keepdims=True))
    points_soft_semantics[point_ids] = e_x / e_x.sum(axis=1, keepdims=True)
    return points_semantic_ids, points_soft_semantics


def get_xyz(uvd_points, I, R, t):
    # project uvd to xyz1
    udvdd = uvd_points[:, :3].T.copy()
    udvdd[:2, :] = udvdd[:2, :] * udvdd[2:, :]
    xyz = np.linalg.pinv(R) @ (np.linalg.pinv(I) @ udvdd - t)
    return xyz  # (3, n)


def get_fov_mask(points_xyz, cam_W=1280):
    if points_xyz.shape[1] == 2:
        points_xyz = np.concatenate([points_xyz, np.zeros([points_xyz.shape[0], 1])], axis=1)
    points_uv = get_uv(points_xyz, get_H())
    maskW0 = points_uv[0, :] > 0
    maskW1 = points_uv[0, :] < cam_W
    mask_fov = np.bitwise_and(maskW0, maskW1)
    return mask_fov


def create_fov_fan(cx, cy, r, theta1, theta2, num_points=100, require_polygon=True):
    angles = np.linspace(theta1, theta2, num_points)
    points = [(cx, cy)] + [(r * math.cos(math.radians(angle)) + cx, r * math.sin(math.radians(angle)) + cy) for angle in angles]
    return Polygon(points) if require_polygon else np.array(points)


def find_boundary_intersections(hull_polygon, fov_fan):
    boundary_intersections = hull_polygon.boundary.intersection(fov_fan.boundary)
    if boundary_intersections.is_empty:
        return []
    if isinstance(boundary_intersections, MultiPoint):
        return [(point.x, point.y) for point in boundary_intersections.geoms]
    else:
        return [(boundary_intersections.x, boundary_intersections.y)]


def split_convex_hull(hull_points, fov_fan):
    hull_polygon = Polygon(hull_points)
    if not hull_polygon.is_valid:
        hull_polygon = hull_polygon.buffer(0)
    intersection = hull_polygon.intersection(fov_fan)
    if intersection.is_empty:
        return None
    if isinstance(intersection, Polygon):
        coords = np.array(intersection.exterior.coords[:-1])
    else:
        largest = max(intersection, key=lambda x: x.area)
        coords = np.array(largest.exterior.coords[:-1])
    boundary_intersections = find_boundary_intersections(hull_polygon, fov_fan)
    if boundary_intersections:
        coords = np.vstack([coords, np.array(boundary_intersections)])
    return coords[ConvexHull(coords).vertices, :]


def upsample_by_fov(data, cx, cy, r, theta1, theta2):
    convex_hull_indices = np.unique(data[:, -1])
    final_hulls = []
    fov_fan = create_fov_fan(cx, cy, r, theta1, theta2)
    for idx in convex_hull_indices:
        hull_data = data[data[:, -1] == idx][:, :]
        in_fov = get_fov_mask(hull_data[:, :2])
        if all(in_fov):
            final_hulls.append(hull_data)
        elif not any(in_fov):
            continue
        elif hull_data.shape[0] >= 4:
            new_hull = split_convex_hull(hull_data[:, :2], fov_fan)
            if new_hull is not None:
                other_attributes = hull_data[0, 2:]
                new_hull = np.concatenate([new_hull, np.tile(other_attributes, (new_hull.shape[0], 1))], axis=1)
                final_hulls.append(new_hull)
    return np.concatenate(final_hulls, axis=0) if final_hulls else None
