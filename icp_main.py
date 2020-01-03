import os
import json
import argparse
import numpy as np
import utils as utls
import rigid_icp
import non_rigid_icp


def load_config_params(config_params_filename):
    print('load opt_params from file:')
    print(config_params_filename)
    print('')
    with open(config_params_filename, 'r') as f:
        config_params = json.load(f)
    return config_params


def save_results(output_dir_path, rigid_results, non_rigid_results, pc_target):
    os.makedirs(output_dir_path, exist_ok=True)
    rigid_results_dir = os.path.join(output_dir_path, r'rigid_results')
    os.makedirs(rigid_results_dir, exist_ok=True)
    for i, iteration_result in enumerate(rigid_results):
        pc_file_path = os.path.join(rigid_results_dir, f'{i}.ply')
        print(f'save {pc_file_path}')
        utls.save_pc(iteration_result, pc_file_path)

    non_rigid_results_dir = os.path.join(output_dir_path, r'non_rigid_results')
    os.makedirs(non_rigid_results_dir, exist_ok=True)
    for i, iteration_result in enumerate(non_rigid_results):
        pc_file_path = os.path.join(non_rigid_results_dir, f'{i}.ply')
        print(f'save {pc_file_path}')
        utls.save_pc(iteration_result, pc_file_path)

    pc_target_file_path = os.path.join(output_dir_path, r'target.ply')
    utls.save_pc(pc_target, pc_target_file_path)
    print(f'save {pc_file_path}')


def do_normalization(pc_source, pc_target):
    source_max_val = np.max(np.abs(pc_source))
    target_max_val = np.max(np.abs(pc_target))
    normalization_factor = np.max([source_max_val, target_max_val])
    pc_source_new = pc_source / normalization_factor
    pc_target_new = pc_target / normalization_factor
    return pc_source_new, pc_target_new


def main(pc_source_path, pc_target_path, output_dir_path, config_params_file_path):
    pc_source = utls.load_pc(pc_source_path)
    pc_target = utls.load_pc(pc_target_path)

    config_params = load_config_params(config_params_file_path)

    if config_params['do_normalize']:
        pc_source, pc_target = do_normalization(pc_source, pc_target)

    print(f'run rigid ICP..')
    rigid_results = rigid_icp.align(pc_source, pc_target, config_params['rigid'])
    pc_source_aligned_rigid = rigid_results[-1]
    print(f'rigid ICP done')

    print(f'run non-rigid ICP')
    non_rigid_results = non_rigid_icp.align(pc_source_aligned_rigid, pc_target, config_params['non_rigid'])
    # non_rigid_results = []
    print(f'non-rigid ICP done')

    print(f'==========================')
    print('save results..')
    save_results(output_dir_path,
                 rigid_results, non_rigid_results, pc_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc1', '--pc1_filename', type=str, required=True, help='path to pc .mat or .ply file')
    parser.add_argument('-pc2', '--pc2_filename', type=str, required=True, help='path to pc .mat or .ply file')
    parser.add_argument('-odp', '--output_dir_path', type=str, required=True, help='path to directory in which result will be saved')
    parser.add_argument('-cpf', '--config_params_file_path', type=str, required=True, help='path to json file with configuration params')
    args = vars(parser.parse_args())
    main(**args)
