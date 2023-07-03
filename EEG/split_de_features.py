import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from absl import app, flags
from absl.flags import FLAGS

from EEG.config import w_size, clips

flags.DEFINE_string(
    'eeg_de_features_path', None,
    'The path of the eeg de features',
    required=True)

flags.DEFINE_string(
    'split_data_path', None,
    'The path of the resulting eeg de features',
    required=True)

flags.DEFINE_string(
    'dataset', None,
    'EEG seed dataset. Select one of: "china" , "fra" or "ger"'
)


""" ---General Info/Instructions---

    This script converts the concatenated .npz files of the EEG DE 1s_split features 
    of the SEED datasets to pickle (.pkl) files. The data is distributed from the BCMI
    lab of the Shanghai Jiao Tong University. To access and download the data you must 
    apply at the following page:
    https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/

    ** Data **
    The datasets used for these experiments can be found at:
    -SEED: https://bcmi.sjtu.edu.cn/home/seed/seed.html
    -SEED-FRA: https://bcmi.sjtu.edu.cn/home/seed/seed-FRA.html
    -SEED-GER: https://bcmi.sjtu.edu.cn/home/seed/seed-GER.html
    """


def main(args):
    del args

    feature_1s_dir = Path(FLAGS.eeg_de_features_path)
    file_1s_list = os.listdir(feature_1s_dir)
    file_1s_list.sort()

    for item in file_1s_list:
        name = item.split('.')[0]

        npz_data = np.load(os.path.join(feature_1s_dir, item))

        train_data = pickle.loads(npz_data['train_data'])
        train_labels = npz_data['train_label']

        test_data = pickle.loads(npz_data['test_data'])
        test_labels = npz_data['test_label']

        train_marks_idx = find_clips(train_labels)
        test_marks_idx = find_clips(test_labels)

        train_x_split, train_y_split = split_clip_data(train_data, train_labels, train_marks_idx, w_size)

        test_x_split, test_y_split = split_clip_data(test_data, test_labels, test_marks_idx, w_size)

        output_path = Path(FLAGS.split_data_path)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Split clip names correctly
        dataset = FLAGS.dataset
        if dataset not in ['china', 'fra', 'ger']:
            warnings.warn("Warning! Please select one of the china, ger or fra datasets.")
        train_clips = clips[f'{dataset}_train']
        test_clips = clips[f'{dataset}_test']

        for i in range(train_clips):
            outpath = output_path / f'{dataset}_{name}_{i}.pkl'
            dump_data(train_x_split[i], train_y_split[i], outpath)

        for j in range(test_clips):
            outpath = output_path / f'{dataset}_{name}_{j}.pkl'
            dump_data(test_x_split[j], test_y_split[j], outpath)


def find_clips(labels):
    marks = []
    for i in range(len(labels)):
        if i == len(labels) - 1:
            continue
        if labels[i + 1] != labels[i]:
            marks.append(i)

    return marks


def split_clip_data(data, labels, marks, wsize):
    data = np.array(list(data.values()))
    data = np.swapaxes(data, 0, -1)

    split_data = []
    split_labels = []

    for i in range(len(marks)):
        if i == 0:
            start = 0
        else:
            start = marks[i - 1]

        tmp_data = data[:, start:marks[i] + 1, :]
        length = tmp_data.shape[1]

        if 2 * wsize > length >= wsize:
            tmp_data = tmp_data[:, :wsize, :]

        elif length >= 2 * wsize:
            tmp_data1 = tmp_data[:, :wsize, :]
            tmp_data2 = tmp_data[:, -wsize:, :]

            # Normalize each channel and each wave of the eeg signal
            tmp_data1 = normalize_channels(tmp_data1)
            tmp_data2 = normalize_channels(tmp_data2)

            split_data.append(tmp_data1)
            split_data.append(tmp_data2)

            split_labels.append(int(labels[marks[i]]))
            split_labels.append(int(labels[marks[i]]))

            continue

        else:
            tmp_data = np.pad(tmp_data, ((0, 0), (0, wsize - length), (0, 0)), 'constant')

        #  Normalize each channel and each wave of eeg signal
        tmp_data = normalize_channels(tmp_data)

        split_data.append(tmp_data)
        split_labels.append(int(labels[marks[i]]))

    last_split_data = data[:, marks[-1]:, :]
    length = last_split_data.shape[1]

    if 2 * wsize > length >= wsize:
        last_split_data = last_split_data[:, :wsize, :]

        last_split_data = normalize_channels(last_split_data)

        split_data.append(last_split_data)
        split_labels.append(int(labels[-1]))

    elif length >= 2 * wsize:
        last_split_data1 = last_split_data[:, :wsize, :]
        last_split_data2 = last_split_data[:, -wsize:, :]

        last_split_data1 = normalize_channels(last_split_data1)
        last_split_data2 = normalize_channels(last_split_data2)

        split_data.append(last_split_data1)
        split_data.append(last_split_data2)

        split_labels.append(int(labels[-1]))
        split_labels.append(int(labels[-1]))
    else:
        last_split_data = np.pad(last_split_data, ((0, 0), (0, wsize - length), (0, 0)), 'constant')

        last_split_data = normalize_channels(last_split_data)

        split_data.append(last_split_data)
        split_labels.append(int(labels[-1]))

    return split_data, split_labels


def dump_data(data, labels, filename):
    obj = {
        'data': data,
        'label': labels
    }

    file = open(filename, 'wb')
    pickle.dump(obj, file)

    file.close()


def normalize_channels(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[-1]):
            if np.ptp(data[i, :, j]) == 0:
                data[i, :, j] = data[i, :, j]
            else:
                data[i, :, j] = 2. * (data[i, :, j] - np.min(data[i, :, j])) / (np.ptp(data[i, :, j])) - 1

    return data


if __name__ == '__main__':
    app.run(main)
