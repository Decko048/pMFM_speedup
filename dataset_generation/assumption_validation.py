import os
from re import sub
import sys
from matplotlib import pyplot as plt
from seaborn import histplot
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ftian/storage/pMFM_speedup/')

from CBIG_pMFM_basic_functions_HCP import CBIG_combined_cost_train

from src.utils.SC_utils import corr_between_SCs
from src.basic.constants import PATH_TO_PROJECT
from src.basic.subject_group import SubjectGroup


def forward_simulation(param_vectors, path_to_group: str, n_dup: int = 5):
    """
    Args:
        param_vectors:  (N*3+1)*M matrix. 
                        N is the number of ROI
                        M is the number of candidate parameter vectors. 
    """
    FCD = os.path.join(path_to_group, 'group_level_FCD.mat')
    SC = os.path.join(path_to_group, 'group_level_SC.csv')
    FC = os.path.join(path_to_group, 'group_level_FC.csv')
    total_cost, fc_corr_cost, fc_L1_cost, fcd_cost, bold_d, r_E, emp_fc_mean = CBIG_combined_cost_train(
        param_vectors, n_dup, FCD, SC, FC, 0)
    return fc_corr_cost, fc_L1_cost, fcd_cost, total_cost


def test_swap_param(swapping_subject_group,
                    swapped_subject_group,
                    path_to_swapped_group,
                    num_param_to_swap=10,
                    file_name: str = 'simulation_with_swapped_params.csv'):
    SC_corr = corr_between_SCs([swapping_subject_group.SC, swapped_subject_group.SC])[0][1]
    print(f'SC corr between {swapping_subject_group.split_name} {swapping_subject_group.group_index} \
            and {swapped_subject_group.split_name} {swapped_subject_group.group_index} is {SC_corr}')
    params, original_performances = swapping_subject_group.sample_k_params(num_param_to_swap)
    params = params.cpu().numpy()
    original_performances = original_performances.cpu().numpy()
    # print('PARAM VECTORS:', params.shape)  # shape: (205, k)
    # print('ORIGINAL PERFORMANCE:', original_performances.shape)  # shape: (4, k)

    fc_corr_cost, fc_L1_cost, fcd_cost, total_cost = forward_simulation(params, path_to_swapped_group)
    new_performances = np.stack((fc_corr_cost, fc_L1_cost, fcd_cost, total_cost), axis=0)  # shape: (4, k)
    performance_diff = new_performances - original_performances
    SC_corr_row = np.full((1, num_param_to_swap), SC_corr)  # shape: (1, k)

    result_after_swapping = np.concatenate(
        (SC_corr_row, performance_diff, original_performances, new_performances, params), axis=0)
    # print('Result after swapping:', result_after_swapping.shape)  # shape: (218, k)
    np.savetxt(file_name, result_after_swapping, delimiter=',')
    check_swap_param_effect(file_name)

    return result_after_swapping


def check_swap_param_effect(path_to_file):
    df = pd.read_csv(path_to_file, header=None)
    SC_corr = df.iloc[0, 0]
    print("correlation between two SCs:", SC_corr)
    n = df.shape[1]
    print(f'{n} parameters have been swapped')

    total_cost_diff = df.iloc[4, :]
    much_worse_count = total_cost_diff.where(lambda x: x > 2).count()
    print(f'{much_worse_count} parameters with new SC does not have meaningful time course while the original one has')
    much_better_count = total_cost_diff.where(lambda x: x < -2).count()
    print(
        f'{much_better_count} parameters with the original SC does not have meaningful time course while the new one has'
    )
    both_not_meaningful_count = total_cost_diff.where(lambda x: x == 0).count()
    print(
        f'{both_not_meaningful_count} parameters with identical performance after swapping (most likely due to no meaningful time course being generated in either case)'
    )
    both_meaningful_count = len(total_cost_diff) - much_worse_count - much_better_count - both_not_meaningful_count
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    fig.suptitle(
        f'Cost differences, where SC\'s correlation is {SC_corr:.6f} ({both_meaningful_count} params involved)',
        y=1.08,
        fontsize=18)

    # plot total cost difference
    total_cost_diff = total_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(total_cost_diff, ax=ax[0]).set_title(
        f'Total cost difference\n(mean={total_cost_diff.mean():.4f}, std={total_cost_diff.std():.4f})')
    ax[0].set_xlabel('total cost difference')

    # plot FC_CORR cost difference
    fc_corr_cost_diff = df.iloc[1, :]
    fc_corr_cost_diff = fc_corr_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(fc_corr_cost_diff, ax=ax[1]).set_title(
        f'FC_CORR cost difference\n(mean={fc_corr_cost_diff.mean():.4f}, std={fc_corr_cost_diff.std():.4f})')
    ax[1].set_xlabel('FC_CORR cost difference')

    # plot FC_L1 cost difference
    fc_L1_cost_diff = df.iloc[2, :]
    fc_L1_cost_diff = fc_L1_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(fc_L1_cost_diff, ax=ax[2]).set_title(
        f'FC_L1 cost difference\n(mean={fc_L1_cost_diff.mean():.4f}, std={fc_L1_cost_diff.std():.4f})')
    ax[2].set_xlabel('FC_L1 cost difference')

    # plot FCD_KS cost difference
    fcd_cost_diff = df.iloc[3, :]
    fcd_cost_diff = fcd_cost_diff.where(lambda x: (abs(x) <= 2) & (x != 0)).dropna()
    histplot(
        fcd_cost_diff,
        ax=ax[3]).set_title(f'FCD cost difference\n(mean={fcd_cost_diff.mean():.4f}, std={fcd_cost_diff.std():.4f})')
    ax[3].set_xlabel('FCD cost difference')

    img_file_path = os.path.splitext(path_to_file)[0] + '.png'
    fig.savefig(img_file_path, bbox_inches='tight')


if __name__ == '__main__':
    path_to_swapped_group = os.path.join(PATH_TO_PROJECT, 'dataset_generation/input_to_pMFM/train/1')
    # swapped_subject_group = SubjectGroup('train', 1)
    swapping_subject_group = SubjectGroup('train', 1)

    split_name = sys.argv[1]
    group_index = sys.argv[2]
    swapped_subject_group = SubjectGroup(split_name, group_index)

    test_swap_param(
        swapping_subject_group,
        swapped_subject_group,
        path_to_swapped_group,
        num_param_to_swap=5000,
        file_name=os.path.join(
            PATH_TO_PROJECT,
            # f'dataset_generation/swap_params/use_{split_name}_{group_index}_param_on_train_1_SC.csv'))
            f'dataset_generation/swap_params/use_train_1_param_on_{split_name}_{group_index}_SC.csv'))
