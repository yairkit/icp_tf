import os
import re
import cv2
import json
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils as utls


def atoi(text):
    return int(text) if text.isdigit() else text


def int_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_results(results_dir_path):
    results = []
    frames_filenames = os.listdir(results_dir_path)
    frames_filenames_sorted = sorted(frames_filenames, key=int_keys)
    for frame_filename in frames_filenames_sorted:
        frame_file_path = os.path.join(results_dir_path, frame_filename)
        frame_pc = utls.load_pc(frame_file_path)
        results.append(frame_pc)
    return results


def avg_l2(all_frames_list, target_pc, plots_dir, save_avg_l2_flag):
    all_avg_l2_dist = []
    avg_l2_fig_filename = os.path.join(plots_dir, r'l2_dist_avg_graph.png')
    for source_pc in all_frames_list:
        l2_dist = np.linalg.norm(source_pc - target_pc, axis=1)
        avg_l2_dist = np.average(l2_dist, axis=0)
        all_avg_l2_dist.append(avg_l2_dist)
    if save_avg_l2_flag:
        plt.figure()
        plt.plot(all_avg_l2_dist)
        plt.savefig(avg_l2_fig_filename)
        plt.close()

    return all_avg_l2_dist


def save_vid(frames_dir, plots_dir):
    size = None
    img_array = []
    frames_filenames = os.listdir(frames_dir)
    video_filename = os.path.join(plots_dir, r'video.avi')
    frames_filenames_sorted = sorted(frames_filenames, key=int_keys)
    for filename in frames_filenames_sorted:
        img_file_path = os.path.join(frames_dir, filename)
        img = cv2.imread(img_file_path)
        height, width, layers = img.shape
        img_array.append(img)
        if not size:
            size = (width, height)

    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'DIVX'), fps=3, frameSize=size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def save_plot(cur_frame_file, source_pc, target_pc, all_avg_l2_dist, config_params):
    mlab.figure( bgcolor=(0, 0, 0))
    mlab.points3d(source_pc[:, 0], source_pc[:, 1], source_pc[:, 2], color=(0.3, 0.45, 1), mode='sphere',
                  scale_factor=0.007, opacity=0.6)
    mlab.points3d(target_pc[:, 0], target_pc[:, 1], target_pc[:, 2], color=(0.8, 0.85, 0.45), mode='sphere',
                  scale_factor=0.007, opacity=0.6)
    if config_params['fid_indices'] is not None:
        mlab.points3d(source_pc[config_params['fid_indices']['source'], 0],
                      source_pc[config_params['fid_indices']['source'], 1],
                      source_pc[config_params['fid_indices']['source'], 2],
                      color=(0.8, 0.2, 0.2), mode='sphere', scale_factor=0.02)
        mlab.points3d(target_pc[config_params['fid_indices']['target'], 0],
                      target_pc[config_params['fid_indices']['target'], 1],
                      target_pc[config_params['fid_indices']['target'], 2],
                      color=(0.2, 0.8, 0.2), mode='sphere', scale_factor=0.02)
    mlab.view(azimuth=135, elevation=55)
    mlab.title(f'avg l2 dist - {all_avg_l2_dist:9.5f}', size=0.8, height=0.7)
    mlab.savefig(cur_frame_file, magnification=2)
    mlab.close()


def save_frames_imgs(all_frames_list, target_pc, all_avg_l2_dist, config_params,
                    plots_dir, save_frames_img_flag, save_vid_flag):
    frames_dir = os.path.join(plots_dir, r'frames_plots')
    os.makedirs(frames_dir, exist_ok=True)
    for i, source_pc in enumerate(all_frames_list):
        cur_frame_file = os.path.join(frames_dir, f'{i}.png')
        save_plot(cur_frame_file, source_pc, target_pc, all_avg_l2_dist[i], config_params)
    if save_vid_flag:
        save_vid(frames_dir, plots_dir)


def plot_results(results_dir_path, save_avg_l2_flag=True, save_frames_img_flag=True, save_vid_flag=True):
    plots_dir = os.path.join(results_dir_path, r'plots')
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(results_dir_path, r'config_params.json'), 'r') as f:
        config_params = json.load(f)

    target_pc = utls.load_pc(os.path.join(results_dir_path, r'target.ply'))

    for results_type in ['rigid', 'non_rigid']:
        cur_results_dir = f'{results_type}_results'
        if (cur_results_dir in os.listdir(results_dir_path)) and (config_params[results_type]['debug_mode']):
            cur_results_dir_path = os.path.join(results_dir_path, cur_results_dir)
            all_frames_list = load_results(cur_results_dir_path)
            cur_plots_dir = os.path.join(plots_dir, results_type)
            os.makedirs(cur_plots_dir, exist_ok=True)
            all_avg_l2_dist = avg_l2(all_frames_list, target_pc, cur_plots_dir, save_avg_l2_flag)
            save_frames_imgs(all_frames_list, target_pc, all_avg_l2_dist, config_params[results_type],
                             cur_plots_dir, save_frames_img_flag, save_vid_flag)
