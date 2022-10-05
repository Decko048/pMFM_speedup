import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_all_tfevents(path: str, filter_keyword: str):
    """
    Finds all paths to tfevents files in a directory.
    :param path: Path to the directory.
    :param filter_keyword: Keyword to filter the files by.
    :return: List of all paths to tfevents files with filter_keyword.
    """
    tfevents_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ("tfevents" in file) and (filter_keyword in root):
                tfevents_paths.append(os.path.join(root, file))
    return tfevents_paths


# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        # print(f"Extracting data from {path}")
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            # print(tag)
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print(f"Event file possibly corrupt: {path}")
        traceback.print_exc()
    return runlog_data


def get_version_number_from(path):
    version_number = path.split("lightning_logs/version_")[1].split('/')[0]
    return version_number


def get_model_name_from(path):
    model_name = path.split("/lightning")[0].split("training_log/")[1].replace("/", " ")
    return model_name


def get_top_k_version(tfevents_paths, top_k=10, max_version_number=499):
    all_best_models = []
    for p in tfevents_paths:
        version_number = get_version_number_from(p)
        if int(version_number) > max_version_number:
            continue
        model_name = get_model_name_from(p)
        df = tflog2pandas(p)
        #df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
        mse_loss_df = df[(df.metric == 'val_epoch/mse_loss')]

        if not mse_loss_df.empty:
            min_value_index = mse_loss_df['value'].idxmin()
            min_mse_loss = mse_loss_df.iloc[min_value_index]

            assert (len(df[(df.metric == 'val_epoch/FC_CORR_mse') & (df.step == min_mse_loss.step)]) == 1)
            FC_CORR_mse = df[(df.metric == 'val_epoch/FC_CORR_mse') & (df.step == min_mse_loss.step)].iloc[0, 1]
            FC_L1_mse = df[(df.metric == 'val_epoch/FC_L1_mse') & (df.step == min_mse_loss.step)].iloc[0, 1]
            FCD_KS_mse = df[(df.metric == 'val_epoch/FCD_KS_mse') & (df.step == min_mse_loss.step)].iloc[0, 1]

            min_row = pd.Series([
                version_number,
                min_mse_loss.step,
                min_mse_loss.value,
                FC_CORR_mse,
                FC_L1_mse,
                FCD_KS_mse,
            ])
            all_best_models.append(min_row)

    all_best_models_df = pd.DataFrame(all_best_models)
    all_best_models_df.set_axis(['version_number', 'step', 'mse_loss', 'FC_CORR_mse', 'FC_L1_mse', 'FCD_KS_mse'],
                                axis=1,
                                inplace=True)
    all_best_models_df.sort_values(by=['mse_loss'], inplace=True)
    top_k_all_best_models_df = all_best_models_df.iloc[:top_k]
    top_k_all_best_models_df.reset_index(drop=True, inplace=True)

    print(f'saving top k models for {model_name}')
    top_k_all_best_models_df.to_csv(f"{model_name}_best_models.csv", index=False)
    # top_k_all_best_models_df.to_excel(f"{model_name}_best_models.xlsx", float_format="%.4f")


if __name__ == "__main__":
    # path = "/home/ftian/storage/pMFM_speedup/reports/training_log/gnn/gcn_with_mlp"
    # path = "/home/ftian/storage/pMFM_speedup/reports/training_log/basic_models/naive_net/with_SC"
    # path = "/home/ftian/storage/pMFM_speedup/reports/training_log/basic_models/naive_net/no_SC"
    # path = "/home/ftian/storage/pMFM_speedup/reports/training_log/basic_models/naive_net/no_SC_use_coef"
    path = "/home/ftian/storage/pMFM_speedup/reports/training_log/basic_models/naive_net/with_SC_use_coef"
    tfevents_paths = find_all_tfevents(path, "")
    get_top_k_version(tfevents_paths, top_k=10, max_version_number=100)
