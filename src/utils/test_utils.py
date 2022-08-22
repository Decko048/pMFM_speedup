import os
import sys
from tokenize import group
from pytorch_lightning import LightningModule
import torch
import numpy as np
import pandas as pd
from seaborn import histplot
from matplotlib import pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss

from torch_geometric.data.batch import Batch

from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.basic.subject_group import SubjectGroup
from src.models.gnn.gnn_param_dataset import GnnParamDataset, get_data_list
from src.models.naive_net import NaiveNet

from src.utils.init_utils import load_split, set_gpu_device
from src.basic.cost_type import CostType
from src.basic.constants import PATH_TO_FIGURES, PATH_TO_PROJECT, PATH_TO_DATASET, NUM_GROUP_IN_SPLIT, TOTAL_COST_UPPER_BOUND
from src.basic.param_performance import ParamPerformance
# from src.models.ensemble.custom_voting import CustomVoting, voting_ensemble_models_in


def pred_and_actual_cost_corr_dist(split_name: str, k: int = 10000):
    """
    plot the distribution of the correlation between top `k` prediction and their actual costs for all subject groups in `split_name`
    """
    save_dir = os.path.join(PATH_TO_PROJECT, f'reports/testing')

    all_FC_CORR_corr = []
    all_FC_L1_corr = []
    all_FCD_KS_corr = []
    all_total_cost_corr = []
    for group_index in range(1, NUM_GROUP_IN_SPLIT[split_name] + 1):
        group_index = str(group_index)
        FC_CORR_corr, FC_L1_corr, FCD_KS_corr, total_cost_corr = top_k_prediction_vs_actual_cost(split_name,
                                                                                                 group_index,
                                                                                                 k=k)
        all_FC_CORR_corr.append(FC_CORR_corr)
        all_FC_L1_corr.append(FC_L1_corr)
        all_FCD_KS_corr.append(FCD_KS_corr)
        all_total_cost_corr.append(total_cost_corr)

    plot_pred_and_actual_cost_corr_dist(all_total_cost_corr, save_dir,
                                        f'{split_name}_top_{k}_pred_and_actual_total_cost_corr_dist')
    plot_pred_and_actual_cost_corr_dist(all_FC_CORR_corr, save_dir,
                                        f'{split_name}_top_{k}_pred_and_actual_FC_CORR_corr_dist')
    plot_pred_and_actual_cost_corr_dist(all_FC_L1_corr, save_dir,
                                        f'{split_name}_top_{k}_pred_and_actual_FC_L1_corr_dist')
    plot_pred_and_actual_cost_corr_dist(all_FCD_KS_corr, save_dir,
                                        f'{split_name}_top_{k}_pred_and_actual_FCD_KS_corr_dist')


def plot_pred_and_actual_cost_corr_dist(all_corr, save_dir, file_name):
    all_corr = pd.Series(all_corr)

    all_corr.to_csv(os.path.join(save_dir, f'{file_name}.csv'), index=False, header=False)

    histplot(all_corr)
    plt.title(f'distribution of correlation between prediction and their actual costs')
    plt.xlabel(
        f'correlation between prediction and actual costs\n(mean={all_corr.mean():.4f}, std={all_corr.std():.4f})')
    plt.savefig(os.path.join(save_dir, f'{file_name}.png'), dpi=400)
    plt.clf()


def keep_meaningful_only(actual_total_costs: pd.Series, predicted_total_costs: pd.Series):
    """
    only keep costs generated by meaningful params
    """
    meaningful_indices = actual_total_costs < TOTAL_COST_UPPER_BOUND
    meaningful_actual_total_costs = actual_total_costs[meaningful_indices]
    meaningful_predicted_total_costs = predicted_total_costs[meaningful_indices]
    return meaningful_actual_total_costs, meaningful_predicted_total_costs


def top_k_prediction_vs_actual_cost(split_name, group_index, k=10000):
    path_to_prediction_dir = os.path.join(PATH_TO_PROJECT, 'reports/testing/compare_top_k_params', split_name,
                                          str(group_index))

    (actual_FC_CORR, actual_FC_L1, actual_FCD_KS,
     actual_total_costs), (predicted_FC_CORR, predicted_FC_L1, predicted_FCD_KS,
                           predicted_total_costs) = get_actual_and_prediction(path_to_prediction_dir, k=k)

    FC_CORR_corr = plot_top_k_prediction_vs_actual_cost(predicted_FC_CORR, actual_FC_CORR, path_to_prediction_dir,
                                                        f'top_{k}_prediction_vs_actual_FC_CORR')
    FC_L1_corr = plot_top_k_prediction_vs_actual_cost(predicted_FC_L1, actual_FC_L1, path_to_prediction_dir,
                                                      f'top_{k}_prediction_vs_actual_FC_L1')
    FCD_KS_corr = plot_top_k_prediction_vs_actual_cost(predicted_FCD_KS, actual_FCD_KS, path_to_prediction_dir,
                                                       f'top_{k}_prediction_vs_actual_FCD_KS')
    total_cost_corr = plot_top_k_prediction_vs_actual_cost(predicted_total_costs, actual_total_costs,
                                                           path_to_prediction_dir,
                                                           f'top_{k}_prediction_vs_actual_total_cost')
    return FC_CORR_corr, FC_L1_corr, FCD_KS_corr, total_cost_corr


def get_actual_and_prediction(path_to_prediction_dir, k=10000):
    predicted_top_k_file_path = os.path.join(path_to_prediction_dir, f'predicted_top_{k}.csv')
    predicted_top_k_params = np.loadtxt(predicted_top_k_file_path, delimiter=',')

    actual_FC_CORR = pd.Series(predicted_top_k_params[0, :])
    actual_FC_L1 = pd.Series(predicted_top_k_params[1, :])
    actual_FCD_KS = pd.Series(predicted_top_k_params[2, :])
    actual_total_costs = pd.Series(predicted_top_k_params[3, :])

    predicted_FC_CORR = pd.Series(predicted_top_k_params[-4, :])
    predicted_FC_L1 = pd.Series(predicted_top_k_params[-3, :])
    predicted_FCD_KS = pd.Series(predicted_top_k_params[-2, :])
    predicted_total_costs = pd.Series(predicted_top_k_params[-1, :])
    return (actual_FC_CORR, actual_FC_L1, actual_FCD_KS, actual_total_costs), (predicted_FC_CORR, predicted_FC_L1,
                                                                               predicted_FCD_KS, predicted_total_costs)


def plot_top_k_prediction_vs_actual_cost(predicted_costs, actual_costs, save_dir, file_name):
    COUNT_N_MEANINGLESS = False

    assert (actual_costs.shape[0] == predicted_costs.shape[0])
    num_of_meaningless_params = len(actual_costs[actual_costs == TOTAL_COST_UPPER_BOUND])
    corr = actual_costs.corr(predicted_costs)
    prediction_vs_actual_costs = pd.DataFrame({'predicted costs': predicted_costs, 'actual costs': actual_costs})
    prediction_vs_actual_costs.to_csv(os.path.join(save_dir, f'{file_name}.csv'), index=False)
    prediction_vs_actual_costs.plot(x='predicted costs', y='actual costs', kind='scatter', figsize=(10, 5), fontsize=16)

    title = f'prediction vs actual costs (r={corr:.4f})'
    if COUNT_N_MEANINGLESS:
        title += f'\n{100*num_of_meaningless_params/actual_costs.shape[0]:.2f}% predicted params generate NaN during Euler forward simulation'

    plt.title(title, fontsize=17)
    plt.xlabel(f'predicted costs (mean={predicted_costs.mean():.4f}, std={predicted_costs.std():.4f})', fontsize=15)
    plt.ylabel(f'actual costs (mean={actual_costs.mean():.4f}, std={actual_costs.std():.4f})', fontsize=15)
    plt.savefig(os.path.join(save_dir, f'{file_name}.png'), dpi=400)
    plt.clf()
    return corr


def plot_top_k_distribution(split_name, group_index, k=10000):
    group_index = str(group_index)
    ground_truth_top_k_file_path = os.path.join(PATH_TO_DATASET, split_name, group_index, f'top_{k}_params.csv')
    path_to_prediction_dir = os.path.join(PATH_TO_PROJECT, 'reports/testing/compare_top_k_params', split_name,
                                          group_index)
    predicted_top_k_file_path = os.path.join(path_to_prediction_dir, f'predicted_top_{k}.csv')
    actual_top_k_params = np.loadtxt(ground_truth_top_k_file_path, delimiter=',')
    predicted_top_k_params = np.loadtxt(predicted_top_k_file_path, delimiter=',')

    ground_truth_top_k_total_costs = actual_top_k_params[3, :]
    predicted_top_k_actual_total_costs = predicted_top_k_params[3, :]
    num_of_meaningless_params = len(predicted_top_k_actual_total_costs[predicted_top_k_actual_total_costs == 25])
    predicted_top_k_actual_total_costs = predicted_top_k_actual_total_costs[predicted_top_k_actual_total_costs < 25]
    predicted_top_k_predicted_total_costs = predicted_top_k_params[-1, :]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Compare predicted top {k} params with ground truth top {k} params of {split_name} {group_index}',
                 y=1.09,
                 fontsize=18)

    histplot(ground_truth_top_k_total_costs, ax=ax[0]).set_title(
        f'Ground Truth top {k} params\' total costs\n(mean={ground_truth_top_k_total_costs.mean():.6f}, std={ground_truth_top_k_total_costs.std():.6f})'
    )
    ax[0].set_xlabel('total costs')

    histplot(predicted_top_k_actual_total_costs, ax=ax[1]).set_title(
        f'Predicted top {k} params\' actual total costs\n(mean={predicted_top_k_actual_total_costs.mean():.6f}, std={predicted_top_k_actual_total_costs.std():.6f})\n{num_of_meaningless_params} params cannot generate meaningful TC'
    )
    ax[1].set_xlabel('total costs')

    histplot(predicted_top_k_predicted_total_costs, ax=ax[2]).set_title(
        f'Predicted top {k} params\' predicted total costs\n(mean={predicted_top_k_predicted_total_costs.mean():.6f}, std={predicted_top_k_predicted_total_costs.std():.6f})'
    )
    ax[2].set_xlabel('total costs')

    img_file_path = os.path.join(path_to_prediction_dir, f'compare_top_{k}_params.png')
    fig.savefig(img_file_path, bbox_inches='tight')
    plt.clf()


def gnn_get_top_k_prediction(group, model, k=10000, ignore_negative_costs=False):
    batch = Batch.from_data_list(get_data_list(group.split_name, group.group_index))
    y_test = batch.y
    with torch.no_grad():
        model.eval()
        y_pred = model(batch)

    return save_top_k_prediction(group, y_pred, y_test, k=k, ignore_negative_costs=False)


def get_top_k_prediction(group, model, k=10000, ignore_negative_costs=False):
    n = len(group)

    SC = group.SC.to(model.device)
    batched_SC = SC.unsqueeze(0).repeat((n, 1, 1))

    batched_param = torch.stack([p.param.to(model.device) for p in group.param_performances], dim=0)
    y_test = torch.stack([p.performance[:3] for p in group.param_performances], dim=0)

    with torch.no_grad():
        model.eval()
        y_pred = model((batched_SC, batched_param))

    return save_top_k_prediction(group, y_pred, y_test, k, ignore_negative_costs=False)


def save_top_k_prediction(group, y_pred, y_test, k=10000, ignore_negative_costs=False):
    y_pred = y_pred.cpu()
    y_test = y_test.cpu()
    loss = mse_loss(y_pred, y_test)
    print(f'MSE loss: {loss}')

    y_pred = y_pred.numpy()
    y_pred_total_cost = np.sum(y_pred, axis=1)

    sorted_indices = np.argsort(y_pred_total_cost)

    top_k_pred = [
        np.concatenate((group.param_performances[i].to_numpy(), y_pred[i], np.array([y_pred_total_cost[i]])))
        for i in sorted_indices[:k]
    ]
    top_k_pred = np.stack(top_k_pred, axis=1)

    if ignore_negative_costs:
        top_k_pred = top_k_pred[:, top_k_pred[-1, :] > 0]
    save_dir = os.path.join(PATH_TO_PROJECT, 'reports/testing/compare_top_k_params', group.split_name,
                            group.group_index)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(save_dir, f'predicted_top_{k}.csv'), top_k_pred, delimiter=',')
    return top_k_pred


def get_all_subject_group_top_k(split_name: str, k: int = 100):
    """
    Get top k parameters for each subject groups.
    """
    dataset = load_split(split_name)
    for group_index, subject_group in enumerate(dataset.subject_groups, 1):
        group_index = str(group_index)
        top_k_file_path = os.path.join(PATH_TO_DATASET, split_name, group_index, f'top_{k}_params.csv')
        top_k_params = subject_group.get_top_k_params(k)
        np.savetxt(top_k_file_path, top_k_params, delimiter=',')


def test_model(model, test_ds, save_dir=None):
    """
    test model's performance on the given dataset (e.g. validation or test dataset)
    """

    BATCH_SIZE = 256
    NUM_WORKERS = 0
    PIN_MEMORY = True
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, \
                                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    y_test = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            y_test.extend(y_batch.cpu().numpy())

            if isinstance(X_batch, list):
                batch_SC, batch_params = X_batch
                batch_SC = batch_SC.to(model.device)
                batch_params = batch_params.to(model.device)
            else:
                X_batch = X_batch.to(model.device)

            y_test_pred = model(X_batch)
            _, y_pred_batch = torch.max(y_test_pred, dim=1)
            y_pred.extend(y_pred_batch.cpu().numpy())

    # metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0
    accuracy = accuracy_score(y_test, y_pred)
    print("\nprecision     |     recall      |       f1       |     accuracy    ")
    print(f"{precision:.7f}\t{recall:.7f}\t{f1:.7f}\t{accuracy:.7f}")

    if save_dir:
        data = [[precision, recall, f1, accuracy]]
        df = pd.DataFrame(data, columns=['precision', 'recall', 'f1', 'accuracy'])
        df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
        writer = SummaryWriter(log_dir=save_dir)
        writer.add_scalar('precision', precision, 0)
        writer.add_scalar('recall', recall, 0)
        writer.add_scalar('f1', f1, 0)
        writer.add_scalar('acc', accuracy, 0)


class GoodParamCounter():

    def __init__(self, filter=lambda x: x.is_all_GOOD):
        self.filter = filter

        # Note that we are only keeping costs for parameters that can generate meaningful BOLD signal
        self.predicted_all_GOOD_params_total_cost = []
        self.predicted_all_GOOD_params_FC_CORR = []
        self.predicted_all_GOOD_params_FC_L1 = []
        self.predicted_all_GOOD_params_FCD_KS = []
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    @property
    def predicted_meaningful_param_count(self):
        return len(self.predicted_all_GOOD_params_total_cost)

    @property
    def actual_all_GOOD_count(self):
        return self.tp + self.fn

    @property
    def predicted_all_GOOD_count(self):
        return self.tp + self.fp

    def reset(self):
        self.predicted_all_GOOD_params_total_cost = []
        self.predicted_all_GOOD_params_FC_CORR = []
        self.predicted_all_GOOD_params_FC_L1 = []
        self.predicted_all_GOOD_params_FCD_KS = []
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, param_performance: ParamPerformance):
        if self.filter(param_performance):
            if param_performance.is_all_GOOD:
                self.tp += 1
            else:
                self.fp += 1

            if param_performance.is_meaningful:
                self.predicted_all_GOOD_params_total_cost.append(param_performance.total_cost.numpy())
                self.predicted_all_GOOD_params_FC_CORR.append(param_performance.FC_CORR_cost.numpy())
                self.predicted_all_GOOD_params_FC_L1.append(param_performance.FC_L1_cost.numpy())
                self.predicted_all_GOOD_params_FCD_KS.append(param_performance.FCD_KS.numpy())

        else:
            if param_performance.is_all_GOOD:
                self.fn += 1
            else:
                self.tn += 1

    @property
    def precision(self):
        return self.tp / self.predicted_all_GOOD_count if self.predicted_all_GOOD_count else 0

    @property
    def recall(self):
        return self.tp / self.actual_all_GOOD_count if self.actual_all_GOOD_count else 0

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision +
                                                   self.recall) if self.precision and self.recall else 0

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp +
                                      self.fn) if self.tp + self.tn + self.fp + self.fn else 0


def check_all_good_params_costs(split_name: str, fig_path: str = PATH_TO_FIGURES, filter=lambda x: x.is_all_GOOD):
    dataset = load_split(split_name, use_SC=False)
    print(f'{split_name} dataset loaded')
    param_counter = GoodParamCounter(filter)

    print('start counting good params')
    for subject_group in dataset.subject_groups:
        for param_performance in subject_group.param_performances:
            param_counter.update(param_performance)

    predicted_all_GOOD_params_total_cost = np.array(param_counter.predicted_all_GOOD_params_total_cost)
    predicted_all_GOOD_params_FC_CORR = np.array(param_counter.predicted_all_GOOD_params_FC_CORR)
    predicted_all_GOOD_params_FC_L1 = np.array(param_counter.predicted_all_GOOD_params_FC_L1)
    predicted_all_GOOD_params_FCD_KS = np.array(param_counter.predicted_all_GOOD_params_FCD_KS)

    fig_suptitle = f'Among {len(dataset)} params in the {split_name} split, {param_counter.predicted_all_GOOD_count} params are predicted as all GOOD,\n' \
                    + f'among which {param_counter.predicted_meaningful_param_count} params can generate meaningful BOLD signal\n' \
                    + f'Precision: {param_counter.precision:.7f}, Recall: {param_counter.recall:.7f}, F1: {param_counter.f1:.7f}, Accuracy: {param_counter.accuracy:.7f}'

    Path(fig_path).mkdir(parents=True, exist_ok=True)
    fig_file_path = os.path.join(fig_path, f'predicted_all_GOOD_params_costs-{split_name}.png')
    plot_predicted_all_GOOD_params(predicted_all_GOOD_params_total_cost, predicted_all_GOOD_params_FC_CORR,
                                   predicted_all_GOOD_params_FC_L1, predicted_all_GOOD_params_FCD_KS, fig_suptitle,
                                   fig_file_path)
    return predicted_all_GOOD_params_total_cost, predicted_all_GOOD_params_FC_CORR, predicted_all_GOOD_params_FC_L1, predicted_all_GOOD_params_FCD_KS


def plot_predicted_all_GOOD_params(predicted_all_GOOD_params_total_cost, predicted_all_GOOD_params_FC_CORR,
                                   predicted_all_GOOD_params_FC_L1, predicted_all_GOOD_params_FCD_KS, fig_suptitle: str,
                                   fig_file_path: str):

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(fig_suptitle, y=1.14, fontsize=17)

    # plot total cost
    histplot(predicted_all_GOOD_params_total_cost, ax=ax[0]).set_title(
        f'Total cost\n(mean={predicted_all_GOOD_params_total_cost.mean():.4f}, std={predicted_all_GOOD_params_total_cost.std():.4f})'
    )
    ax[0].set_xlabel('Total cost')

    # plot FC_CORR cost
    histplot(predicted_all_GOOD_params_FC_CORR, ax=ax[1]).set_title(
        f'FC_CORR cost\n(mean={predicted_all_GOOD_params_FC_CORR.mean():.4f}, std={predicted_all_GOOD_params_FC_CORR.std():.4f})'
    )
    ax[1].set_xlabel('FC_CORR cost')

    # plot FC_L1 cost
    histplot(predicted_all_GOOD_params_FC_L1, ax=ax[2]).set_title(
        f'FC_L1 cost\n(mean={predicted_all_GOOD_params_FC_L1.mean():.4f}, std={predicted_all_GOOD_params_FC_L1.std():.4f})'
    )
    ax[2].set_xlabel('FC_L1 cost')

    # plot FCD_KS cost
    histplot(predicted_all_GOOD_params_FCD_KS, ax=ax[3]).set_title(
        f'FCD_KS cost\n(mean={predicted_all_GOOD_params_FCD_KS.mean():.4f}, std={predicted_all_GOOD_params_FCD_KS.std():.4f})'
    )
    ax[3].set_xlabel('FCD_KS cost')

    fig.savefig(fig_file_path, bbox_inches='tight')


def get_last_ckpt_model(model_cls: LightningModule, dir_containing_lightning_logs: str):
    ckpt_files_dir = os.path.join(dir_containing_lightning_logs, 'lightning_logs/version_0/checkpoints')
    last_ckpt_files = [f for f in os.listdir(ckpt_files_dir) if f.endswith('last.ckpt')]
    if not last_ckpt_files:
        raise ValueError(f'No last.ckpt file found in {ckpt_files_dir}')
    else:
        return model_cls.load_from_checkpoint(os.path.join(ckpt_files_dir, last_ckpt_files[0]))


def get_voting_ensemble_model(model_cls: LightningModule, dir_containing_lightning_logs: str, num_voting_models):
    ckpt_files_dir = os.path.join(dir_containing_lightning_logs, 'lightning_logs/version_0/checkpoints')
    ckpt_files = [f for f in os.listdir(ckpt_files_dir) if f.endswith('.ckpt')]
    if not ckpt_files:
        raise ValueError(f'No ckpt files found in {ckpt_files_dir}')
    else:
        return voting_ensemble_models_in(model_cls, ckpt_files_dir, num_voting_models)


def get_all_cost_type_models(model_cls: LightningModule, dir: str, num_voting_models=0):
    models = []
    for cost_type in CostType:
        dir_containing_lightning_logs = os.path.join(dir, cost_type.name)
        if num_voting_models:
            print('Getting voting ensemble model...')
            models.append(get_voting_ensemble_model(model_cls, dir_containing_lightning_logs, num_voting_models))
        else:
            print('Getting single model...')
            models.append(get_last_ckpt_model(model_cls, dir_containing_lightning_logs))

    assert len(models) == 3
    return models


def check_all_good_params_costs_for_models_in(dir: str,
                                              model_cls: LightningModule,
                                              split_name: str,
                                              fig_path: str = PATH_TO_FIGURES,
                                              num_voting_models=0,
                                              device=None):
    FC_CORR_model, FC_L1_model, FCD_KS_model = get_all_cost_type_models(model_cls, dir, num_voting_models)

    if device:
        FC_CORR_model.to(device)
        FC_L1_model.to(device)
        FCD_KS_model.to(device)

    def filter_fn(param_perform):
        with torch.no_grad():
            param = param_perform.param.unsqueeze(0).to(FC_CORR_model.device)
            FC_CORR_model.eval()
            FC_L1_model.eval()
            FCD_KS_model.eval()
            has_GOOD_FC_CORR = bool(torch.argmax(FC_CORR_model(param)))
            has_GOOD_FC_L1 = bool(torch.argmax(FC_L1_model(param)))
            has_GOOD_FCD_KS = bool(torch.argmax(FCD_KS_model(param)))
            return has_GOOD_FC_CORR and has_GOOD_FC_L1 and has_GOOD_FCD_KS

    return check_all_good_params_costs(split_name, fig_path, filter_fn)
