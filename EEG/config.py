from pathlib import Path

# Experiment results dir
eeg_results_dir = Path('/home/user/experiments/EEG/')

# Data Directories
pickle_data_dir = Path('/home/user/data/biodg/eeg/')

clips = {
    'china_train': 9,
    'china_test': 6,
    'fra_train': 12,
    'fra_test': 8,
    'ger_train': 12,
    'ger_test': 6
}

# Clip Information
fs = 200                         # sampling frequency
start_hint = int(5 * 200)        # 5sec * 200Hz = 1000 data points
self_assessment = int(45 * 200)  # 45sec * 200Hz = 9000 data points
rest = int(15 * 200)             # 15 * 200Hz = 3000 data points


experiment_lengths = {
    'ww_eeg1': 47001,
    'ww_eeg2': 46601,
    'ww_eeg3': 41201,
    'ww_eeg4': 47601,
    'ww_eeg5': 37001,
    'ww_eeg6': 39001,
    'ww_eeg7': 47401,
    'ww_eeg8': 43201,
    'ww_eeg9': 53001,
    'ww_eeg10': 47401,
    'ww_eeg11': 47001,
    'ww_eeg12': 46601,
    'ww_eeg13': 47001,
    'ww_eeg14': 47601,
    'ww_eeg15': 41201
}


########## EEG experiment config ##########
w_size = 170
w_size_sec = 10
