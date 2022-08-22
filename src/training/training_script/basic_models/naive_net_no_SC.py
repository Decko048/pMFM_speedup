import sys

sys.path.insert(0, '/home/ftian/storage/pMFM_speedup/')
from src.training.training_lib import *

if __name__ == '__main__':
    save_dir = os.path.join(PATH_TO_TRAINING_LOG, "basic_models/naive_net/no_SC/")
    # tune(objective_naive_net_no_SC, n_trials=50, timeout=86400, save_dir=save_dir)
    resume_tuning(objective_naive_net_no_SC, n_trials=50, timeout=86400, save_dir=save_dir)
