import os
import time
import json
from icp_main import main
from visual_utils import plot_results


# --------------------------- inputs paths: ------------------------------------------
pc_source_path = r'inputs/duck_orig.ply'
pc_target_path = r'inputs/duck_affine.ply'
output_dir_path = '../outputs'

# --------------------------- inputs config params: ----------------------------------
fid_indices = dict()
fid_indices['source'] = [[0], [2000], [3000]]
fid_indices['target'] = [[0], [2000], [3000]]

config_params = dict()
config_params['do_normalize'] = True

config_params['rigid'] = dict()
config_params['rigid']['debug_mode'] = True
config_params['rigid']['max_iterations'] = 20
config_params['rigid']['tolerance'] = 0.00001
config_params['rigid']['fid_indices'] = fid_indices

config_params['non_rigid'] = dict()
config_params['non_rigid']['laplacian_k'] = 6
config_params['non_rigid']['debug_mode'] = True
config_params['non_rigid']['max_iterations'] = 15
config_params['non_rigid']['diff_thresh_init'] = 0.3
config_params['non_rigid']['diff_thresh_end'] = 0.1
config_params['non_rigid']['p_to_point_w_init'] = 1e2
config_params['non_rigid']['p_to_point_w_end'] = 1e3
config_params['non_rigid']['stiff_w_init'] = 1e7
config_params['non_rigid']['stiff_w_end'] = 14

config_params['non_rigid']['fid_points_w_init'] = 1e6
config_params['non_rigid']['fid_points_w_end'] = 1e6
config_params['non_rigid']['fid_indices'] = fid_indices


os.makedirs(output_dir_path, exist_ok=True)
time_str = time.strftime("%Y-%m-%d_%H-%M")
cur_output_dir_path = os.path.join(output_dir_path, time_str)
os.makedirs(cur_output_dir_path, exist_ok=True)

config_params_file_path = os.path.join(cur_output_dir_path, 'config_params.json')
with open(config_params_file_path, 'w') as fp:
    json.dump(config_params, fp)

# --------------------------- run icp: ---------------------------------
main(pc_source_path, pc_target_path, cur_output_dir_path, config_params_file_path)


# --------------------------- plot results: ----------------------------
plot_results(cur_output_dir_path)
