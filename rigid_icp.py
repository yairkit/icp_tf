import copy
import numpy as np
import tensorflow as tf
import utils as utls


def transform_points(points, transform):
    points_new = tf.matmul(points, transform['rotation_mat'], transpose_b=True) + tf.reshape(transform['translation_vec'], [1, 3])
    return points_new


def best_fit_transform(pc_source, pc_target):
    assert pc_source.shape == pc_target.shape
    transform = dict()
    # translate points to their centroids
    centroid_source = tf.math.reduce_mean(pc_source, axis=0, keepdims=True)
    centroid_target = tf.math.reduce_mean(pc_target, axis=0, keepdims=True)
    pc_source_centered = pc_source - centroid_source
    pc_target_centered = pc_target - centroid_target

    # rotation matrix
    H = tf.linalg.matmul(pc_source_centered, pc_target_centered, transpose_a=True)
    S, U, V = tf.linalg.svd(H, full_matrices=True)
    R = tf.matmul(V, U, transpose_b=True)
    transform['rotation_mat'] = R

    # special reflection case
    if tf.linalg.det(R) < 0:
        mask = tf.constant([[1, -1, 1], [1, -1, 1], [1, -1, 1]])
        V = tf.math.multiply(V, mask)
        R = tf.matmul(V, U, transpose_b=True)

    # translation
    transform['translation_vec'] = tf.transpose(centroid_target) - tf.matmul(R, centroid_source, transpose_b=True)

    return transform


def align(pc_source, pc_target, config_params):
    all_iterations_results = []
    pc_source_aligned_rigid = copy.deepcopy(pc_source)
    mean_error_prev = 0

    for i in range(config_params['max_iterations']):
        print(f'rigid icp iteration: {i}')
        distances, indices = utls.get_knn(pc_source_aligned_rigid, pc_target)
        target_closest = utls.get_closest(pc_target, indices)
        transform = best_fit_transform(pc_source_aligned_rigid, target_closest)
        pc_source_aligned_rigid = transform_points(pc_source_aligned_rigid, transform)

        if config_params['debug_mode']:
            all_iterations_results.append(pc_source_aligned_rigid)
        else:
            all_iterations_results = [pc_source_aligned_rigid]

        mean_error = np.mean(distances)
        print('mean error: {}'.format(mean_error))
        if np.abs(mean_error - mean_error_prev) < config_params['tolerance']:
            print('error less than tolerance')
            break
        mean_error_prev = mean_error

    return all_iterations_results
