from probreg import cpd, filterreg
import transforms3d as t3d
import cv2
import os
import dill
import yaml
import datetime

from data_io import read_radar, read_video, load_gt
from model.process import gain_img_info
from my_particle import N_PARTICLE, STATE_SIZE, Particle, calc_final_state, motion_model, particle_filtering
from my_registration import collect_point2img_feature, convert_point_feature, to_cluster_feature
from utils import get_H, cal_v_comp_all, get_uvd, get_IRT, get_xyz
from vis_utils import PointCloudVisualizer
from det_utils import points_filter, estimate, preprocess_predictions, preprocess_targets, current_detection
from ransac_fit_line import *
from gaussion_sample import sample_img_depth

config_file = 'config.yaml'
with open(config_file, 'r') as f:
    configs = yaml.safe_load(f)
show_level = configs['show_level']
saved_function_file = './data/error_function_saved.bin'
f_sigma_u, f_sigma_v, f_sigma_d = dill.load(open(saved_function_file, 'rb'))
save_dir = f'./results/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
frame_target = []
reg_acu = []
mae_lrs = []
var_lrs = []
mae_mlrs = []
var_mlrs = []
mae_glrs = []
var_glrs = []


def LOG(*args, **kwargs):
    output_str = " ".join(map(str, args))
    print(output_str, **kwargs)
    print(output_str, file=log_f, **kwargs)


def cluster_semantic_registration(xyzs, win_xyzs, labels_frame):
    win_xyz, win_cluster_f = to_cluster_feature(win_xyzs, labels_frame, USE_WEIGHT=False, show_level=show_level)
    xyz, cluster_f = to_cluster_feature(xyzs, np.zeros(xyzs.shape[0]), show_level=show_level)
    if configs['use_point_seg']:
        tf_param, _, _ = filterreg.registration_filterreg(cluster_f, win_cluster_f)
    else:
        tf_param, _, _ = cpd.registration_cpd(xyz, win_xyz, update_scale=False)
    return tf_param


def radar_particle_filtering(particles, radar_points_far, reg_result, Q_est, Q_reg, DT):
    vx_est, vy_ext, _ = cal_v_comp_all(radar_points_far)
    u = np.array([vx_est, vy_ext]).reshape([2, 1])
    dx, dy, dtheta = reg_result
    z = np.array([dx, dy, dtheta]).reshape([3, 1])
    particles = particle_filtering(particles, u, z, Q_est, Q_reg, DT)
    xEst, T = calc_final_state(particles)
    dxDR = motion_model(None, u, DT)
    return xEst, T, dxDR


def main():
    video_file = './data/2021-01-06_15-18-18.mp4'
    radar_file = "./data/test.h5"
    H = get_H()

    window_lenth = 5
    concate_num_radar = 5
    video_g, total_frames = read_video(video_file)
    radar_g = read_radar(radar_file)
    truth_g = load_gt()
    average_win_lenth = 5
    results = [np.array([0, 0, 0])]
    tolerance = np.array([0.5, 0.5, 0.1])
    base_var = np.array([0.1, 0.1, 0.01])

    last_radar_points = radar_g.__next__().T
    last_video_frame = video_g.__next__()

    seg_probs, grid_map, depth_map = gain_img_info(last_video_frame)
    last_xyzvs, points_soft_semantics = collect_point2img_feature(last_radar_points, H, seg_probs,
                                                                  last_video_frame.shape[0],
                                                                  last_video_frame.shape[1],)
    last_xyzs, last_xyzs_all, last_compensate, last_points_soft_semantics = convert_point_feature(last_xyzvs,
                                                                                                  USE_COMP=True,
                                                                                                  soft_labels=points_soft_semantics)
    R0 = np.identity(3)
    T0 = np.zeros(3)
    xyzs_C0 = last_xyzs_all.copy()
    labelsC = np.zeros(last_xyzs.shape[0])
    Q_est = np.diag([0.5, 0.5]) ** 2  # vx, vy
    rm_range = 2.5
    DT = 0.1 * concate_num_radar  # time interval between frame
    particles = [Particle() for _ in range(N_PARTICLE)]
    xEst = np.zeros((STATE_SIZE, 1))  # weight sum of all particle's location
    xDR = np.zeros((STATE_SIZE, 1))  # weight sum of all particle's dx/dy/dz
    hxEst = xEst
    hxDR = xDR
    radar_points_far = last_radar_points[np.linalg.norm(last_radar_points[:, 0:3], axis=1) > rm_range, :]
    noo_scores = []
    frame_noos = []
    frame_false_alarm_rates = []
    frame_recalls = []
    con_sim_list = []
    all_true_positive_num = 0
    all_targets_num = 0
    all_preds_num = 0

    for i in range(total_frames):
        radar_points = radar_g.__next__().T
        video_frame = video_g.__next__()
        truth_points, frame_gt, pred_transform_list = truth_g.__next__()

        seg_probs, grid_map, depth_map = gain_img_info(video_frame)
        xyzvs, points_soft_semantics = collect_point2img_feature(radar_points, H, seg_probs,
                                                                 video_frame.shape[0],
                                                                 video_frame.shape[1],
                                                                 ori_seg=True)
        xyzs, xyzs_all, cur_compensate, soft_semantics = convert_point_feature(xyzvs,
                                                                               USE_COMP=True,
                                                                               soft_labels=points_soft_semantics)

        win_xyzs = last_xyzs[labelsC > (i - window_lenth), :]
        labels_frame = labelsC[labelsC > (i - window_lenth)]
        labels_frame = np.max(labels_frame) - labels_frame
        labelsC = np.concatenate([labelsC, (i + 1) * np.ones(xyzs.shape[0])])
        tf_param = cluster_semantic_registration(xyzs, win_xyzs, labels_frame)
        new_xyz = tf_param.transform(xyzs[:, :3].copy())
        R, T = tf_param.rot.T, tf_param.t
        Rc = R
        Tc = T @ R.T
        win_results = np.concatenate(results[max(0, len(results) - average_win_lenth): len(results)], axis=0).reshape(
            (-1, 3))  # all filter reg results in window (for var calculation)
        reg_result = np.array([Tc[0], Tc[1], list(t3d.euler.mat2euler(Rc))[2]])
        var_factor = np.maximum(abs(np.average(win_results, axis=0) - reg_result) / tolerance, np.array([1, 1, 1]))
        reg_var = base_var * (var_factor ** 2)
        LOG("reg_result:", reg_result)
        LOG("reg_var:", reg_var)

        Q_reg = np.diag(reg_var[:-1]) ** 2
        xEst, filter_result, dxDR = radar_particle_filtering(particles, radar_points_far, reg_result, Q_est, Q_reg, DT)
        results.append(filter_result)
        Rf = t3d.euler.euler2mat(0., 0., filter_result[2])
        Tf = np.array([filter_result[0], filter_result[1], 0]) @ Rf  # points' transform
        R0 = np.dot(Rf, R0)
        T0 = T0 + np.dot(Tf, R0)

        last_xyzs = np.concatenate([np.concatenate([np.dot((last_xyzs[:, :3] - Tf), Rf.T), last_xyzs[:, 3:]], axis=1),
                                    xyzs])
        last_compensate = np.concatenate([last_compensate, cur_compensate])
        last_points_soft_semantics = np.concatenate([last_points_soft_semantics, soft_semantics])
        xyzs_C0 = np.concatenate([xyzs_C0, np.concatenate([(np.dot(xyzs_all[:, :3], R0) + T0), xyzs_all[:, 3:]],
                                                          axis=1)])  # convert points to the initial coordinate
        radar_points_xyzs, raw_points, cur_soft_labels = points_filter(last_xyzs, type='radar', frame_ids=labelsC,
                                                                           compensate=last_compensate,
                                                                           soft_labels=last_points_soft_semantics)

        if show_level >= 1:
            fig, axs = plt.subplots(1, 5, figsize=(20, 5))
            axs[0].scatter(xyzs[:, 0], xyzs[:, 1], c='r', label='a')
            axs[0].set_aspect('equal')
            axs[0].set_title('current points')
            axs[1].scatter(win_xyzs[:, 0], win_xyzs[:, 1], c='b', label='b')
            axs[1].set_aspect('equal')
            axs[1].set_title('history points')
            axs[2].scatter(xyzs[:, 0], xyzs[:, 1], c='r', label='a')
            axs[2].scatter(win_xyzs[:, 0], win_xyzs[:, 1], c='b', label='b')
            axs[2].set_aspect('equal')
            axs[2].set_title('simple concat')
            axs[3].scatter(new_xyz[:, 0], new_xyz[:, 1], c='r', label='a')
            axs[3].scatter(win_xyzs[:, 0], win_xyzs[:, 1], c='b', label='b shifted')
            axs[3].set_aspect('equal')
            axs[3].set_title('filter reg results')
            new_xyz = np.dot(xyzs[:, :3], R0) + T0
            axs[4].scatter(new_xyz[:, 0], new_xyz[:, 1], c='r', label='a')
            axs[4].scatter(win_xyzs[:, 0], win_xyzs[:, 1], c='b', label='b shifted')
            axs[4].set_aspect('equal')
            axs[4].set_title('particle filter results')
            plt.show()

        if configs['detect_level'] in ['fusion', 'cam']:
            seg_map = np.argmax(seg_probs, axis=0)  # 0-6, keep 2, 3, 4
            foreground_seg_mask = np.zeros_like(seg_map)
            foreground_seg_mask[seg_map == 2] = 1
            foreground_seg_mask[seg_map == 3] = 1
            foreground_seg_mask[seg_map == 4] = 1
            guided_points_dis = np.linalg.norm(raw_points[:, 0:3], axis=1)
            guided_radar_points = raw_points[np.bitwise_and(guided_points_dis > 8, guided_points_dis < 20), :3]
            uvd = get_uvd(guided_radar_points, H).T  # (n, 3)
            mask = np.ones(guided_radar_points.shape[0])
            mask[uvd[:, 0] < 0] = 0
            mask[uvd[:, 0] > 1280] = 0
            mask[uvd[:, 1] < 0] = 0
            mask[uvd[:, 1] > 720] = 0
            mask[uvd[:, 2] < 0] = 0
            uvd = uvd[mask == 1]
            guided_radar_points = guided_radar_points[mask == 1, :]
            radar_points_r = np.linalg.norm(guided_radar_points, axis=1)  # shape of n
            radar_points_t = np.arctan(
                guided_radar_points[:, 0] / guided_radar_points[:, 1])  # arctan() will return (-pi/2 - pi/2)
            radar_points_p = np.arctan(guided_radar_points[:, 2] / np.linalg.norm(guided_radar_points[:, :2], axis=1))
            sigma_u_array = f_sigma_u(radar_points_r, radar_points_t, radar_points_p)
            sigma_v_array = f_sigma_v(radar_points_r, radar_points_t, radar_points_p)
            sigma_uv = np.hstack((sigma_u_array.reshape(-1, 1), sigma_v_array.reshape(-1, 1)))
            sampled_radar_depth, sampled_img_depth, sampled_weight, sampled_seg = sample_img_depth(uvd, depth_map,
                                                                                                   sigma_uv,
                                                                                                   seg_mask=foreground_seg_mask,
                                                                                                   seg_map=seg_map)
            sampled_img_depth_weight = np.concatenate([sampled_img_depth.reshape(-1, 1), sampled_weight.reshape(-1, 1)],
                                                      axis=1)
            regressor_s = RANSAC(model=Weighted_LinearRegressor_0(), loss=square_error_loss, metric=mean_square_error,
                                 n=10, k=100, t=1.5, d=100)
            regressor_s.fit(sampled_img_depth_weight, sampled_radar_depth.reshape(-1, 1))
            if regressor_s.best_fit is not None:
                # get regress params
                params_s = regressor_s.best_fit.params
                inner_rate_s = regressor_s.best_fit_inner_num / sampled_radar_depth.shape[0]
                square_error = square_error_loss(sampled_radar_depth.reshape(-1, 1),
                                                 regressor_s.best_fit.predict(sampled_img_depth_weight))
            else:  # use vanilla linear regression(without RANSAC)
                lr_s_regressor = Weighted_LinearRegressor_0()
                lr_s_regressor.fit(sampled_img_depth_weight, sampled_radar_depth.reshape(-1, 1))
                params_s = lr_s_regressor.params
                square_error = square_error_loss(sampled_radar_depth.reshape(-1, 1),
                                                 lr_s_regressor.predict(sampled_img_depth_weight))
                threshold = (square_error < 1.5)
                inner_rate_s = np.flatnonzero(threshold).flatten().shape[0] / sampled_radar_depth.shape[0]
            LOG("inner rate for weighted sample regressor:", inner_rate_s)
            ratio_lr_s = params_s[0][0]

            d_rate = 5
            x_idx, y_idx = np.mgrid[d_rate // 2:720:d_rate, d_rate // 2:1280:d_rate]
            depth_map_d = depth_map[x_idx, y_idx].flatten()
            seg_probs_label = np.argmax(seg_probs, axis=0)
            foreground_seg_mask_d = np.zeros_like(seg_probs_label)
            foreground_seg_mask_d[seg_probs_label == 2] = 1
            foreground_seg_mask_d[seg_probs_label == 3] = 1
            foreground_seg_mask_d[seg_probs_label == 4] = 1
            foreground_seg_mask_d[np.sum(seg_probs, axis=0) < 0.1] = 0
            foreground_seg_mask_d = foreground_seg_mask_d[x_idx, y_idx].flatten()
            seg_map_d = seg_probs_label[x_idx, y_idx].flatten()
            point_seg = seg_map_d[foreground_seg_mask_d == 1].reshape(-1, 1)
            point_depth = ratio_lr_s * depth_map_d[foreground_seg_mask_d == 1].reshape(-1, 1)
            point_u = y_idx.flatten()[foreground_seg_mask_d == 1].reshape(-1, 1)
            point_v = x_idx.flatten()[foreground_seg_mask_d == 1].reshape(-1, 1)
            point_uvds = np.concatenate([point_u, point_v, point_depth, point_seg], axis=1)
            I, R, t = get_IRT()
            point_xyz = get_xyz(point_uvds[:, :3], I, R, t).T
            foreground_probs = seg_probs[:, x_idx, y_idx].reshape(7, -1)[:, foreground_seg_mask_d == 1].transpose()
            e_x = np.exp(foreground_probs - np.max(foreground_probs, axis=1, keepdims=True))
            foreground_probs = e_x / e_x.sum(axis=1, keepdims=True)
            presudo_point_xyzs = np.concatenate([point_xyz, foreground_probs], axis=1)
            _, presudo_point_xyzs, _ = convert_point_feature(presudo_point_xyzs)
            presudo_point_xyzs = presudo_point_xyzs[np.sum(presudo_point_xyzs[:, 3:], axis=1) > 0.1]
        else:
            presudo_point_xyzs = None

        final_radar_points = raw_points[np.sum(raw_points[:, 3:], axis=1) > 0.1]
        final_radar_points = np.concatenate([final_radar_points[:, :3], cur_soft_labels], axis=1)

        if configs['detect_level'] == 'radar':
            final_points = final_radar_points
        elif configs['detect_level'] == 'cam':
            final_points = presudo_point_xyzs
        elif configs['detect_level'] == 'fusion':
            final_points = np.concatenate([presudo_point_xyzs, final_radar_points], axis=0)
        else:
            raise ValueError(f'Not Supporr for type {configs["detect_level"]}')

        labels = estimate(final_points, best_params)
        ax = None
        if show_level >= 1:
            fig, ax = plt.subplots(1, 1, figsize=(18, 18))
        pred_targets = preprocess_predictions(final_points, labels, ax, configs['detect_level'])
        label_targets = preprocess_targets(truth_points, ax)
        if show_level >= 1:
            ax.axis('off')
            plt.margins(0, 0)
            plt.savefig(f'{bev_det_save_dir}/{i}.png')
            plt.close()

        if configs['save_points']:
            if configs['detect_level'] == 'fusion':
                presudo_points_num = presudo_point_xyzs.shape[0]
                radar_points_num = final_radar_points.shape[0]
                assert final_points.shape[0] == presudo_points_num + radar_points_num
                np.save(os.path.join(save_dir, 'presudo_points', f'{i}.npy'), final_points[:presudo_points_num])
                np.save(os.path.join(save_dir, 'radar_points', f'{i}.npy'), final_points[presudo_points_num:])
            elif configs['detect_level'] == 'radar':
                np.save(os.path.join(save_dir, 'radar_points', f'{i}.npy'), final_points)
            elif configs['detect_level'] == 'cam':
                np.save(os.path.join(save_dir, 'presudo_points', f'{i}.npy'), final_points)
            else:
                raise ValueError(f'Wrong value {configs["detect_level"]}')

        if show_level >= 1:
            if configs['save_img']:
                fig = plt.figure()
                plt.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                raw_uvd = get_uvd(radar_points, H).T
                raw_uvd_mask = np.ones(radar_points.shape[0])
                raw_uvd_mask[raw_uvd[:, 0] < 0] = 0
                raw_uvd_mask[raw_uvd[:, 0] > 1278] = 0
                raw_uvd_mask[raw_uvd[:, 1] < 0] = 0
                raw_uvd_mask[raw_uvd[:, 1] > 717] = 0
                raw_uvd_mask[raw_uvd[:, 2] < 0] = 0
                raw_uvd = raw_uvd[raw_uvd_mask == 1]
                plt.scatter(raw_uvd[:, 0], raw_uvd[:, 1], c='blue', s=3)
                plt.savefig(f'{img_save_dir}/{i}.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig)
            visualizer1 = PointCloudVisualizer()
            visualizer1.add_point_cloud(final_points[:, :3], np.argmax(final_points[:, 3:], axis=1))
            visualizer1.save_screenshot(os.path.join(points_save_dir, f"{i}.png"))
            visualizer2 = PointCloudVisualizer()
            visualizer2.add_point_cloud(final_points[:, :3], np.argmax(final_points[:, 3:], axis=1))
            visualizer2.save_screenshot_with_3d_labels(final_points, labels, os.path.join(det_save_dir, f"{i}.png"))

        noo_score, frame_false_alarm_rate, true_positive_num, targets_num, preds_num, con_sims = current_detection(
            pred_targets, label_targets)
        noo_scores.append(noo_score)
        frame_noo = sum(noo_score[noo_score > 0]) / sum(noo_score > 0)
        frame_recall = sum(noo_score > 0) / sum(noo_score > -1)
        LOG(frame_noo, frame_recall, frame_false_alarm_rate)
        frame_noos.append(frame_noo)
        frame_false_alarm_rates.append(frame_false_alarm_rate)
        frame_recalls.append(frame_recall)
        con_sim_list.append(con_sims)
        all_true_positive_num += true_positive_num
        all_targets_num += targets_num
        all_preds_num += preds_num
        frame_target.append(preds_num)

        # store data history
        xDR[:2, [0]] = xDR[:2, [0]] + dxDR
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        radar_points_far = radar_points[np.linalg.norm(radar_points[:, 0:3], axis=1) > rm_range, :]

    noo_scores = np.concatenate(noo_scores)
    LOG(noo_scores)


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if configs['save_points']:
        if not os.path.exists(os.path.join(save_dir, 'presudo_points')):
            os.makedirs(os.path.join(save_dir, 'presudo_points'))
        if not os.path.exists(os.path.join(save_dir, 'radar_points')):
            os.makedirs(os.path.join(save_dir, 'radar_points'))
    det_save_dir = os.path.join(save_dir, 'det')
    if not os.path.exists(det_save_dir):
        os.makedirs(det_save_dir)
    bev_det_save_dir = os.path.join(save_dir, 'bev_det')
    if not os.path.exists(bev_det_save_dir):
        os.makedirs(bev_det_save_dir)
    points_save_dir = os.path.join(save_dir, 'points')
    if not os.path.exists(points_save_dir):
        os.makedirs(points_save_dir)
    img_save_dir = os.path.join(save_dir, 'imgs')
    if configs['save_img']:
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
    log_f = open(os.path.join(save_dir, 'log.txt'), 'w')
    main()
    log_f.close()
