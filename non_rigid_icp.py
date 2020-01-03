import copy
import tensorflow as tf
import utils as utls
import numpy as np


def update_config_params(config_params, iteration):
    updated_config_params = config_params

    diff_thresh_decay = (((config_params['diff_thresh_init'] - config_params['diff_thresh_end']) * iteration) /
                         (config_params['max_iterations'] - 1))
    updated_config_params['diff_thresh'] = config_params['diff_thresh_init'] - diff_thresh_decay

    stiff_decay = (((config_params['stiff_w_init'] - config_params['stiff_w_end']) * iteration) /
                   (config_params['max_iterations'] - 1))
    updated_config_params['stiff_w'] = config_params['stiff_w_init'] - stiff_decay

    p_to_point_decay = (((config_params['p_to_point_w_init'] - config_params['p_to_point_w_end']) * iteration) /
                        (config_params['max_iterations'] - 1))
    updated_config_params['p_to_point_w'] = config_params['p_to_point_w_init'] - p_to_point_decay

    fid_points_w_decay = (((config_params['fid_points_w_init'] - config_params['fid_points_w_end']) * iteration) /
                          (config_params['max_iterations'] - 1))
    updated_config_params['fid_points_w'] = config_params['fid_points_w_init'] - fid_points_w_decay

    return updated_config_params


def calc_p_to_point_mat(points1_closest_vec):
    """
    E_Fit_Point2Point:
    sum | | (v_i - c_i) | | ^ 2  for all i.
    c_i are the corresponding closest target vertices
    :param points1_closest_vec:
    :return: logical sparse matrix with 1 on it's diag ith if  verts_target_closest_vec[i] != 0
    """
    index = tf.reshape(tf.where(tf.not_equal(points1_closest_vec, 0)), [-1, 1])
    indices = tf.cast(tf.concat([index, index], axis=1), dtype=tf.int64)
    data = tf.ones(indices.shape[0], dtype=tf.float64)
    p_to_point_fit_mat = tf.SparseTensor(indices=indices, values=data, dense_shape=(points1_closest_vec.shape[0], points1_closest_vec.shape[0]))
    return p_to_point_fit_mat


def kronecker_product_identity(mat1, mat2):
    # Computes the Kronecker product of two matrices.
    m1, n1 = tf.shape(mat1)
    m2, n2 = tf.shape(mat2)
    out_shape = [m1 * m2, n1 * n2]
    a_row = tf.gather(mat1.indices, indices=[0], axis=-1)
    a_col = tf.gather(mat1.indices, indices=[1], axis=-1)
    a_data = mat1.values
    b_data = mat2.values
    b_row = tf.reshape(tf.gather(mat2.indices, indices=[0], axis=-1), [-1])
    b_col = tf.reshape(tf.gather(mat2.indices, indices=[1], axis=-1), [-1])

    row = tf.squeeze(tf.keras.backend.repeat(a_row, n=m2))
    col = tf.squeeze(tf.keras.backend.repeat(a_col, n=m2))
    data = tf.reshape(tf.squeeze(tf.keras.backend.repeat(tf.reshape(a_data, [-1, 1]), n=m2)), [-1])

    row = row * tf.cast(m2, tf.int64)
    col = col * tf.cast(n2, tf.int64)
    row += b_row
    col += b_col
    row, col = tf.reshape(row, [-1]), tf.reshape(col, [-1])
    data = tf.reshape(data, [-1, m2]) * tf.cast(b_data, dtype=tf.float64)
    data = tf.reshape(data, [-1])

    index = tf.concat([tf.reshape(row, [-1, 1]), tf.reshape(col, [-1, 1])], -1)
    index = tf.dtypes.cast(index, tf.int64)

    data = data / tf.reduce_max(data)

    return tf.sparse.reorder(tf.SparseTensor(indices=index, values=data, dense_shape=out_shape))


def calc_laplacian_mat(points, k):
    num_of_points = points.shape[0]
    adj_mat = utls.pairwise_distance(points, points)
    distance, indices = utls.knn(adj_mat, k=k)
    dst_1_k = 1/tf.cast(distance, dtype=tf.float64)[:, 1:]
    dst_0 = tf.reduce_sum(dst_1_k, axis=1, keepdims=True)
    distance = tf.concat((-dst_0, dst_1_k), axis=1)
    data = tf.reshape(distance, [-1])
    columns = tf.reshape(indices, [-1, 1])
    rows = tf.reshape(tf.range(num_of_points), [-1, 1])
    rows = tf.keras.backend.repeat(rows, k)
    rows = tf.reshape(rows, [-1, 1])
    index = tf.cast(tf.concat((rows, columns), axis=1), dtype=tf.int64)

    return tf.sparse.reorder(tf.SparseTensor(indices=index, values=data, dense_shape=(num_of_points, num_of_points)))


def calc_stiff_mat(laplacian_mat):
    a_t = tf.sparse.reorder(tf.sparse.transpose(laplacian_mat))
    laplace = tf.sparse.to_dense(laplacian_mat)
    aat = tf.sparse.sparse_dense_matmul(a_t, laplace)
    ## convert dense to sparse
    idx = tf.where(tf.not_equal(aat, 0))
    aat = tf.SparseTensor(idx, tf.gather_nd(aat, idx), tf.cast(tf.shape(aat),dtype=tf.int64))

    eye = tf.sparse.eye(3)

    return kronecker_product_identity(aat, eye)


def calc_fid_vals_vec(pc_target, pc_source, fid_indices):
    #  E_Ref -> sum ||v_j-r_j||^2 for all j in reference indices
    shape1_fid_vals = tf.squeeze(tf.gather(pc_target, fid_indices['target']))
    shape1_fid_vals_full = tf.zeros([pc_target.shape[0], 3])
    shape1_fid_vals_full_ta = tf.TensorArray(dtype=shape1_fid_vals_full.dtype, size=shape1_fid_vals_full.shape[0])
    shape1_fid_vals_full_ta = shape1_fid_vals_full_ta.unstack(shape1_fid_vals_full)
    fid2_id = tf.squeeze(fid_indices['source'])

    k = tf.constant(0)
    while_condition = lambda j, shape1_fid_vals_full_ta: tf.less(j, len(fid_indices['target']))
    def body(k, shape1_fid_vals_full_ta):
        shape1_fid_vals_full_ta = shape1_fid_vals_full_ta.write(index=fid2_id[k], value=shape1_fid_vals[k, :])
        k = k + 1
        return k, shape1_fid_vals_full_ta
    m, shape1_fid_vals_full_ta = tf.while_loop(while_condition, body, [k, shape1_fid_vals_full_ta])

    shape1_fid_vals_full_ta = shape1_fid_vals_full_ta.stack()

    return tf.reshape(shape1_fid_vals_full_ta, [-1])


def calc_fid_points_mat(fid_vals_vec):
    indice_row = tf.where(tf.not_equal(fid_vals_vec, 0))
    indices = tf.concat((indice_row, indice_row), axis=1)
    data = tf.ones(indices.shape[0], dtype=tf.float64)
    fid_points_mat = tf.SparseTensor(indices=indices, values=data,
                                     dense_shape=(fid_vals_vec.shape[0], fid_vals_vec.shape[0]))
    return fid_points_mat


def get_b(a_mat, weighted_fit_mat, weighted_stiffness_mat, weighted_fid_points_mat,
          target_closest_vec, pc_source_vec, fid_vals_vec, pc_source_vec_current):
    b1 = tf.sparse.sparse_dense_matmul(weighted_fit_mat,
                                       tf.reshape(tf.cast(target_closest_vec, dtype=tf.float64), [-1, 1]))
    b2 = tf.sparse.sparse_dense_matmul(weighted_stiffness_mat,
                                       tf.reshape(tf.cast(pc_source_vec, dtype=tf.float64), [-1, 1]))
    b4 = tf.sparse.sparse_dense_matmul(a_mat, tf.reshape(tf.cast(pc_source_vec_current, dtype=tf.float64), [-1, 1]))
    b = b1 + b2 - b4
    if weighted_fid_points_mat is not None:
        b3 = tf.sparse.sparse_dense_matmul(weighted_fid_points_mat,
                                           tf.reshape(tf.cast(fid_vals_vec, dtype=tf.float64), [-1, 1]))
        b = b + b3

    return b


def solve_optimization(a_mat, b):
    a_mat = tf.sparse.reorder(a_mat)
    a = tf.sparse.to_dense(a_mat)
    per_point_transform = tf.linalg.solve(a, b)
    return per_point_transform


def transform_points(points, per_point_transform):
    points_vec = tf.reshape(points, [-1])
    points_new = points_vec + tf.reshape(tf.cast(per_point_transform, dtype=tf.float32), [-1])
    points_new = tf.reshape(points_new, [-1, 3])
    return points_new


def align(pc_source, pc_target, orig_config_params):
    all_iterations_results = []
    pc_source_aligned = copy.deepcopy(pc_source)
    pc_source_vec = tf.reshape(pc_source, [-1])
    source_laplacian_mat = calc_laplacian_mat(pc_source, orig_config_params['laplacian_k'])
    for i in range(orig_config_params['max_iterations']):
        print(f'Non rigid icp iteration #{i}')
        config_params = update_config_params(orig_config_params, i)
        distances, indices = utls.get_knn(pc_source_aligned, pc_target)

        # fit term:
        target_closest = utls.get_closest(pc_target, indices)
        target_closest_in = utls.mask_outliers(config_params['diff_thresh'], pc_source_aligned, target_closest)
        target_closest_vec = tf.reshape(target_closest_in, [-1])
        point_to_point_term = calc_p_to_point_mat(target_closest_vec)
        weighted_fit_mat = (tf.cast(config_params['p_to_point_w'], dtype=tf.float64) * point_to_point_term)

        # stiffness term:
        pc_source_aligned_vec = tf.reshape(pc_source_aligned, [-1])
        stiffness_term = calc_stiff_mat(source_laplacian_mat)
        weighted_stiffness_mat = (tf.cast(config_params['stiff_w'], dtype=tf.float64) * stiffness_term)

        a_mat = tf.sparse.add(weighted_fit_mat, weighted_stiffness_mat)

        # fiducial points term:
        if config_params['fid_indices'] is not None:
            fid_vals_vec = calc_fid_vals_vec(pc_target, pc_source, config_params['fid_indices'])  # TODO
            fid_points_term = calc_fid_points_mat(fid_vals_vec)
            weighted_fid_points_mat = (tf.cast(config_params['fid_points_w'], dtype=tf.float64) * fid_points_term)
            a_mat = tf.sparse.add(a_mat, weighted_fid_points_mat)
        else:
            fid_vals_vec = None
            weighted_fid_points_mat = None

        b = get_b(a_mat, weighted_fit_mat, weighted_stiffness_mat, weighted_fid_points_mat,
                  target_closest_vec, pc_source_vec, fid_vals_vec, pc_source_vec_current=pc_source_aligned_vec)
        # ------------- solve optimization problem: Ax = b -------------
        per_point_transform = solve_optimization(a_mat, b)
        pc_source_aligned = transform_points(pc_source_aligned, per_point_transform)
        all_iterations_results.append(pc_source_aligned)

    return all_iterations_results
