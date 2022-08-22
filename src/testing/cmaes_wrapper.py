from pathlib import Path
import numpy as np
import time
import torch
import os
import sys
import csv
import pandas as pd

sys.path.insert(0, '/home/ftian/storage/pMFM_speedup/')
from src.testing.testing_lib import *


def csv_matrix_read(filename):
    '''
    Convert a .csv file to numpy array
    Args:
        filename:   input path of a .csv file
    Returns:
        out_array:  a numpy array
    '''
    csv_file = open(filename, "r")
    read_handle = csv.reader(csv_file)
    out_list = []
    R = 0
    for row in read_handle:
        out_list.append([])
        for col in row:
            out_list[R].append(float(col))
        R = R + 1
    out_array = np.array(out_list)
    csv_file.close()
    return out_array


def get_init(myelin_data, gradient_data, init_para):
    '''
    This function is implemented to calculate the initial parametrized coefficients
    '''

    n_node = myelin_data.shape[0]
    concat_matrix = np.vstack((np.ones(n_node), myelin_data.T, gradient_data.T)).T  # bias, myelin PC, RSFC gradient PC
    para = np.linalg.inv(concat_matrix.T @ concat_matrix) @ concat_matrix.T @ init_para
    return para, concat_matrix


def predict_costs(input_para, SC, model=load_naive_net_no_SC()):
    '''
    This function is implemented to predict the costs of input parameters using the deep learning model given
    '''
    n_param = input_para.shape[1]

    SC = SC.to(model.device)
    batched_SC = SC.type(torch.FloatTensor).unsqueeze(0).repeat((n_param, 1, 1))

    batched_param = torch.from_numpy(input_para).type(torch.FloatTensor).transpose(0, 1).to(model.device)

    with torch.no_grad():
        model.eval()
        y_pred = model((batched_SC, batched_param))

    y_pred = y_pred.cpu().numpy()
    return y_pred.sum(1), y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    # prediction = torch.concat((batched_param, y_pred), 1).transpose(0, 1).cpu().numpy()


def cmaes_wrapper(SC_path,
                  myelin_path,
                  gradient_path,
                  random_seed=MANUAL_SEED,
                  dl_model=load_naive_net_no_SC(),
                  output_path=os.path.join(PATH_TO_TESTING_REPORT, 'cmaes_wrapper'),
                  output_file='prediction.csv'):

    SC = pd.read_csv(SC_path, header=None)
    SC = df_to_tensor(SC)

    # torch.cuda.set_device(gpu_number)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Setting random seed and GPU
    random_seed_cuda = random_seed
    random_seed_np = random_seed
    torch.manual_seed(random_seed_cuda)
    rng = np.random.Generator(np.random.PCG64(random_seed_np))

    # Initializing input parameters
    myelin_data = csv_matrix_read(myelin_path)
    num_myelin_component = myelin_data.shape[1]

    gradient_data = csv_matrix_read(gradient_path)
    num_gradient_component = gradient_data.shape[1]
    N = 3 * (num_myelin_component + num_gradient_component + 1) + 1  # number of parameterized parameters 10
    N_p = num_myelin_component + num_gradient_component + 1  # nunber of parameterized parameter associated to each parameter 3
    n_node = SC.shape[0]  # 68
    dim = n_node * 3 + 1

    wEE_min, wEE_max, wEI_min, wEI_max = 1, 10, 1, 5
    search_range = np.zeros((dim, 2))
    search_range[0:n_node, :] = [wEE_min, wEE_max]  # search range for w_EE
    search_range[n_node:n_node * 2, :] = [wEI_min, wEI_max]  # search range for w_EI
    search_range[n_node * 2, :] = [0, 3]  # search range for G
    search_range[n_node * 2 + 1:dim, :] = [0.0005, 0.01]  # search range for sigma
    init_para = rng.uniform(0, 1, dim) * (search_range[:, 1] - search_range[:, 0]) + search_range[:, 0]
    start_point_w_EE, template_mat = get_init(myelin_data, gradient_data, init_para[0:n_node])
    start_point_w_EI, template_mat = get_init(myelin_data, gradient_data, init_para[n_node:n_node * 2])
    start_point_sigma, template_mat = get_init(myelin_data, gradient_data, init_para[n_node * 2 + 1:dim])

    # Initializing childrens
    xmean = np.zeros(N)  # size 1 x N
    xmean[0:N_p] = start_point_w_EE.squeeze()
    xmean[1] = start_point_w_EE[1] / 2
    xmean[N_p:2 * N_p] = start_point_w_EI.squeeze()
    xmean[4] = start_point_w_EI[1] / 2
    xmean[2 * N_p] = init_para[2 * n_node]  # G
    xmean[2 * N_p + 1:N] = start_point_sigma.squeeze()

    # Initializing optimization hyper-parameters
    sigma = 0.25  # 'spread' of children
    maxloop = 1000
    n_dup = 5  # duplication for pMFM simulation

    # lambda * maxloop * num_of_initialization = number of iterations
    # 36 hours for lambda = 100, max_loop = 100, num_of_initialization = 5
    # we use: lambda = 100, max_loop = 100, num_of_initialization = 1
    # CMA-ES parameters setting
    Lambda = 1000  # number of child processes
    mu = 10  # number of top child processes
    all_parameters = np.zeros((209, maxloop * Lambda))  # numpy array to store all parameters and their associated costs

    weights = np.log(mu + 1 / 2) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    # Strategy parameter setting: adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)**2 + mueff)
    cmu = np.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
    damps = 1 + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    # Initializing dynamic strategy parameters and constants'''
    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)
    D = np.ones(N)
    D[0:N_p] = start_point_w_EE[0] / 2
    D[N_p:2 * N_p] = start_point_w_EI[0] / 2
    D[2 * N_p] = 0.4
    D[2 * N_p + 1:N] = 0.001 / 2

    C = np.dot(np.dot(B, np.diag(np.power(D, 2))), B.T)
    invsqrtC = np.dot(np.dot(B, np.diag(np.power(D, -1))), B.T)
    chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2))

    # Evolution loop
    countloop = 0
    arx = np.zeros([N, Lambda])
    input_para = np.zeros((dim, Lambda))
    xmin = np.zeros(N + 4)

    while countloop < maxloop:
        iteration_log = open(os.path.join(output_path, 'training_iteration.txt'), 'w')
        iteration_log.write(str(countloop))

        start_time = time.time()

        # Generating lambda offspring
        arx[:, 0] = xmean
        j = 0
        infinite_loop_count = 0

        while j < Lambda:
            arx[:, j] = xmean + sigma * np.dot(B, (D * rng.standard_normal(N)))
            input_para[0:n_node, j] = template_mat @ arx[0:N_p, j]
            input_para[n_node:2 * n_node, j] = template_mat @ arx[N_p:2 * N_p, j]
            input_para[2 * n_node:2 * n_node + 1, j] = arx[2 * N_p, j]
            input_para[2 * n_node + 1:dim, j] = template_mat @ arx[2 * N_p + 1:N, j]

            if (input_para[:, j] < search_range[:, 0]).any() or (input_para[:, j] > search_range[:, 1]).any():
                j = j - 1
                infinite_loop_count += 1
                if infinite_loop_count > 20000:
                    iteration_log.write(str(countloop) + ' Infinite Loop')
                    iteration_log.close()
                    return
            j = j + 1

        # Calculating costs of offspring
        print("Predicting using deep learning model ...")
        total_cost, fc_corr_cost, fc_L1_cost, fcd_cost = predict_costs(input_para, SC, model=dl_model)
        # print(input_para.shape, total_cost.shape, fc_corr_cost.shape,
        #       fc_L1_cost.shape, fcd_cost.shape, bold_d.shape, r_E.shape)
        ## (205, 10) (10,) (10,) (10,) (10,) torch.Size([68, 50, 1200]) torch.Size([68, 50]) when Lambda = 10

        # Storing all parameters and their associated costs
        all_parameters[4:, countloop * Lambda:(countloop + 1) * Lambda] = input_para
        all_parameters[0, countloop * Lambda:(countloop + 1) * Lambda] = fc_corr_cost
        all_parameters[1, countloop * Lambda:(countloop + 1) * Lambda] = fc_L1_cost
        all_parameters[2, countloop * Lambda:(countloop + 1) * Lambda] = fcd_cost
        all_parameters[3, countloop * Lambda:(countloop + 1) * Lambda] = total_cost

        countloop = countloop + 1

        # Sort by total cost and compute weighted mean
        arfitsort = np.sort(total_cost)
        arindex = np.argsort(total_cost)
        xold = xmean
        xmean = np.dot(arx[:, arindex[0:mu]], weights)
        xshow = xmean - xold

        # Cumulation
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invsqrtC, xshow) / sigma
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * countloop)) / chiN < (1.4 + 2 / (N + 1))) * 1
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * xshow / sigma

        # Adapting covariance matrix C
        artmp = (1 / sigma) * (arx[:, arindex[0:mu]] - np.tile(xold, [mu, 1]).T)
        C = (1-c1-cmu)*C+c1*(np.outer(pc, pc)+(1-hsig)*cc*(2-cc)*C) + \
            cmu*np.dot(artmp, np.dot(np.diag(weights), artmp.T))

        # Adapting step size
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Decomposition
        if 1 > 1 / (c1 + cmu) / N / 10:
            C = np.triu(C, k=1) + np.triu(C).T
            D, B = np.linalg.eigh(C)
            D = D.real
            B = B.real
            D = np.sqrt(D)
            invsqrtC = np.dot(B, np.dot(np.diag(D**(-1)), B.T))

        # Monitoring the evolution status
        cost_log = open(os.path.join(output_path, f'cost_{str(random_seed)}.txt'), 'w')
        print('******** Generation: ' + str(countloop) + ' ********')
        cost_log.write('******** Generation: ' + str(countloop) + ' ********' + '\n')
        print('The mean of total cost: ', np.mean(arfitsort[0:mu]))

        xmin[0:N] = arx[:, arindex[0]]
        xmin[N] = fc_corr_cost[arindex[0]]
        xmin[N + 1] = fc_L1_cost[arindex[0]]
        xmin[N + 2] = fcd_cost[arindex[0]]
        xmin[N + 3] = np.min(total_cost)
        xmin_save = np.reshape(xmin, (-1, N + 4))
        cost_log.write('sigma: ' + str(sigma) + '\n')
        print('Best parameter set: ', arindex[0])
        cost_log.write('Best parameter set: ' + str(arindex[0]) + '\n')
        print('Best total cost: ', np.min(total_cost))
        cost_log.write('Best total cost: ' + str(np.min(total_cost)) + '\n')
        print('FC correlation cost: ', fc_corr_cost[arindex[0]])
        cost_log.write('FC correlation cost: ' + str(fc_corr_cost[arindex[0]]) + '\n')
        print('FC L1 cost: ', fc_L1_cost[arindex[0]])
        cost_log.write('FC L1 cost: ' + str(fc_L1_cost[arindex[0]]) + '\n')
        print('FCD KS statistics cost: ', fcd_cost[arindex[0]])
        cost_log.write('FCD KS statistics cost: ' + str(fcd_cost[arindex[0]]) + '\n')
        print('wEI search range: ' + str(wEI_min) + ', ' + str(wEI_max))
        print('wEE search range: ' + str(wEE_min) + ', ' + str(wEE_max))
        cost_log.write('wEE search range: ' + str(wEE_min) + ', ' + str(wEE_max) + '\n')
        cost_log.write('wEI search range: ' + str(wEI_min) + ', ' + str(wEI_max) + '\n')

        elapsed_time = time.time() - start_time
        print('Elapsed time for this evolution is : ', elapsed_time)
        cost_log.write('Elapsed time for this evolution is : ' + str(elapsed_time) + '\n')
        print('******************************************')
        cost_log.write('******************************************')
        cost_log.write("\n")

    save_top_k_param(all_parameters, os.path.join(output_path, output_file), k=1000)

    iteration_log.write(str(countloop) + ' Success!')
    iteration_log.close()
    cost_log.close()


def save_top_k_param(all_param, save_dir, k=1000):
    total_cost = all_param[3, :]
    sorted_indicies = np.argsort(total_cost)
    top_k_param = all_param[:, sorted_indicies[:k]]
    np.savetxt(os.path.join(save_dir, 'top_k_param.txt'), top_k_param, delimiter=',')


if __name__ == '__main__':
    group_path = get_path_to_group('train', '5')
    SC_path = os.path.join(group_path, 'group_level_SC.csv')
    myelin_path = os.path.join(group_path, 'group_level_myelin.csv')
    gradient_path = os.path.join(group_path, 'group_level_RSFC_gradient.csv')
    cmaes_wrapper(SC_path, myelin_path, gradient_path)
