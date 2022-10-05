import os
import matplotlib.pyplot as plt
import pandas as pd


def find_csv_under(base_dir: str):
    filenames = os.listdir(base_dir)
    return [filename for filename in filenames if filename.endswith('.csv')]


def csv_paths_to_df(csv_paths):
    return [pd.read_csv(csv_path) for csv_path in csv_paths]


def plot_loss_box_plot(loss_name, all_df, model_names):
    loss_df = pd.concat([df[loss_name] for df in all_df], axis=1)
    loss_df.set_axis(model_names, axis=1, inplace=True)
    loss_df.boxplot()
    plt.savefig(f'{loss_name}_box_plot.png')
    plt.clf()


if __name__ == "__main__":
    base_dir = './'
    csv_file_names = find_csv_under(base_dir)
    csv_paths = [os.path.join(base_dir, csv_path) for csv_path in csv_file_names]
    csv_paths.sort()
    all_df = csv_paths_to_df(csv_paths)
    # model_names = [filename.rstrip('_best_models.csv') for filename in csv_file_names]
    model_names = ['Naive Net\nNo SC', 'Naive Net\nNo SC\nUse Coefficient', 'Naive Net\nWith SC', 'Naive Net\nWith SC\nUse Coefficient', 'GCN']
    plot_loss_box_plot('mse_loss', all_df, model_names)
    plot_loss_box_plot('FC_CORR_mse', all_df, model_names)
    plot_loss_box_plot('FC_L1_mse', all_df, model_names)
    plot_loss_box_plot('FCD_KS_mse', all_df, model_names)
