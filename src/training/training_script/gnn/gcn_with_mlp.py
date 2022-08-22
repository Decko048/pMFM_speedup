import sys

sys.path.insert(0, '/home/ftian/storage/pMFM_speedup/')
from src.training.training_lib import *

if __name__ == '__main__':
    save_dir = os.path.join(PATH_TO_TRAINING_LOG, "gnn/gcn_with_mlp/")
    # tune_gnn(objective_gcn, n_trials=50, timeout=104400, save_dir=save_dir)
    resume_tuning_gnn(objective_gcn, n_trials=25, timeout=50400, save_dir=save_dir)
