import os
import numpy as np
import tensorflow as tf
from plyfile import PlyData, PlyElement


def read_pc_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)
    points = ([np.array(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    points = np.stack(points, axis=-1)
    return points


def load_pc(pc_file_path):
    pc_array = None
    filename = os.path.basename(pc_file_path)
    if filename.endswith('ply'):
        pc_array = read_pc_ply(pc_file_path)

    return pc_array


def save_pc(pc_array, file_path):
    vertex = np.array(
        [tuple(i) for i in pc_array],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex")
    plydata = PlyData([el])
    plydata.write(file_path)


def pairwise_distance(p1, p2):
    """compute pairwise-distance between 2 point cloud

    """
    p2_transpose = tf.transpose(a=p2)
    p_inner = tf.matmul(p1, p2_transpose)
    p_inner = -2*p_inner
    p1_square = tf.reduce_sum(input_tensor=tf.square(p1), axis=-1, keepdims=True)
    p2_square_transpose = tf.transpose(tf.reduce_sum(input_tensor=tf.square(p2),axis=-1, keepdims=True))
    return p1_square + p_inner + p2_square_transpose


def knn(adj_matrix, k=1):
    """
    Get KNN based on the pairwise distance.
    """
    neg_adj = -adj_matrix
    distance, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return -distance, nn_idx


def get_knn(pc_source, pc_target):
    adj_mat = pairwise_distance(pc_source, pc_target)
    return knn(adj_mat, k=1)


def get_closest(pc_target, indices):
    target_closest = tf.squeeze(tf.gather(pc_target, indices=indices, axis=0))
    return target_closest


def mask_outliers(nn_dist_thresh, source_pc, target_closest):
    target_closest_in = tf.identity(target_closest)
    diff_norms = tf.norm(source_pc - target_closest, axis=1)
    inliers_mask = tf.math.less(diff_norms, nn_dist_thresh)
    mask_not = tf.cast(inliers_mask, dtype=tf.float32)
    target_closest_in = target_closest_in * tf.reshape(mask_not, [-1,1])

    return target_closest_in

