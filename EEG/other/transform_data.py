import os
import numpy as np
from EEG.config import eeg_dir_china, export_dir, start_hint, self_assessment, rest
from typing import List
import scipy.io as sio
import pickle



def extract_recording_audio_windows(audio: np.ndarray, wsize: int, wstep: int) -> List[np.ndarray]:
    """
    Creates audio windows from an audio recording.

    :param audio: The audio recording
    :param wsize: The size of the window (in samples)
    :param wstep: The step of the window (in samples)
    :return: A list of audio windows
    """
    recording_audio_windows: List[np.ndarray] = []

    for i in range(0, audio.shape[1] - wsize, wstep):
        recording_audio_windows.append(audio[:, i:i + wsize])

    return recording_audio_windows


def transform_data():
    eeg_file_list = os.listdir(eeg_dir_china)
    eeg_file_list.sort()

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    labels = sio.loadmat(os.path.join(eeg_dir_china, 'label.mat'))
    labels = labels['label'][0] + 1
    count = 0
    for item in eeg_file_list:
        if item in ['label.mat', 'readme.txt']:
            continue

        print(item)
        all_data = sio.loadmat(os.path.join(eeg_dir_china, item))

        signals = []
        split_labels = []
        split_signals = []
        for kk in all_data.keys():
            if 'eeg' in kk:
                x = all_data[kk]
                x = x[:, start_hint:(x.shape[-1] - (self_assessment + rest))]
                signals.append(x)

        for i in range(len(signals)):
            x_tmp = extract_recording_audio_windows(signals[i], 2000, 1600)
            y_tmp = np.zeros(len(x_tmp), dtype=np.int8) + labels[i]
            split_signals.append(x_tmp)
            split_labels.append(y_tmp)

        for w_signal, w_label in zip(split_signals, split_labels):
            assert len(w_signal) == len(w_label)

            for j in range(len(w_signal)):
                window_dict = {
                    'patient_file': item,
                    'data': w_signal[j],
                    'label': w_label[j]
                }

                file = open(export_dir / f'{count}.pkl', 'wb')
                pickle.dump(window_dict, file)
                file.close()
                count += 1

