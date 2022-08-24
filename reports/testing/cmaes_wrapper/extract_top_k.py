import os
from pathlib import Path
import numpy as np


def replace_high_threshold(file_path):
    all_params = np.loadtxt(file_path, delimiter=',')
    fc_corr_cost = all_params[0, :]
    fc_L1_cost = all_params[1, :]
    fcd_cost = all_params[2, :]
    total_cost = all_params[3, :]

    fc_corr_cost[fc_corr_cost >= 1] = 1
    fc_L1_cost[fc_L1_cost >= 1] = 1
    fcd_cost[fcd_cost >= 1] = 1
    total_cost[total_cost >= 3] = 3
    new_performances = np.stack((fc_corr_cost, fc_L1_cost, fcd_cost, total_cost), axis=0)  # shape: (4, k)

    all_params[:4, :] = new_performances

    np.savetxt(file_path, all_params, delimiter=',')


def extract_top_k(file_path, k=20):
    all_params = np.loadtxt(file_path, delimiter=',')
    actual_total_costs = all_params[3, :]
    sorted_indices = np.argsort(actual_total_costs)
    top_k_params = all_params[:, sorted_indices[:k]]
    save_file_path = file_path.replace('.csv', f'_top_{k}.csv')
    np.savetxt(save_file_path, top_k_params, delimiter=',')


def remove_actual_costs(file_path, save_dir='./'):
    all_params = np.loadtxt(file_path, delimiter=',')
    actual_costs = all_params[:4, :]
    params = all_params[4:209, :]
    predicted_costs = all_params[-4:, :]
    top_params_from_val = np.concatenate((predicted_costs, params), axis=0)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(save_dir, 'top_params_from_val.csv'), top_params_from_val, delimiter=',')


if __name__ == '__main__':
    for i in range(1, 15):
        file_path = f'./validation/{i}/actual_costs_and_prediction.csv'
        replace_high_threshold(file_path)
        extract_top_k(file_path, k=20)

    # remove_actual_costs('./validation/10/actual_costs_and_prediction_top_20.csv', save_dir='./validation/')