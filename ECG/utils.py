import pickle
import sys
sys.path.append('C:\\Users\\telis\\PycharmProjects\\biosignals')
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import wfdb
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from scipy import signal
from scipy.io import loadmat

# from datasets.bio.bioaugmentors import Filter
from commons import Database
from ECG.bioconfig import snomed_classes
from sklearn.metrics import multilabel_confusion_matrix


def normalize_channel(data):
    for i in range(data.shape[0]):
        if np.ptp(data[i]) == 0:
            data[i, :] = data[i, :]
        else:
            data[i, :] = 2. * (data[i] - np.min(data[i])) / (np.ptp(data[i])) - 1

    return data


def normalize_channel_eeg(data):
    ex = [11]
    if type(data) != np.ndarray:
        data = data.to_numpy()
    for i in range(data.shape[1]):
        if i in ex:
            continue
        if np.ptp(data[:, i]) == 0:
            data[:, i] = data[:, i]
        else:
            data[:, i] = 2. * (data[:, i] - np.min(data[:, i])) / (np.ptp(data[:, i])) - 1

    return data


def resample_signal(data, output_sampling_rate, input_sampling_rate):
    if int(output_sampling_rate) == int(input_sampling_rate):
        return data

    else:
        sample = data.astype(np.float32)
        factor = output_sampling_rate / input_sampling_rate
        len_old = sample.shape[1]
        num_of_leads = sample.shape[0]
        new_length = int(factor * len_old)
        f_resample = np.zeros((num_of_leads, new_length))

        for i in range(data.shape[0]):
            f_resample[i, :] = (signal.resample(data[i], new_length, window='hamming'))

        return f_resample


def resample_signal_eeg(data, output_sampling_rate, input_sampling_rate, istraining: bool):
    if int(output_sampling_rate) == int(input_sampling_rate):
        return data

    else:
        sample = data.astype(np.float32)
        factor = output_sampling_rate / input_sampling_rate
        len_old = sample.shape[0]
        num_of_channels = sample.shape[1]
        new_length = int(factor * len_old)
        f_resample = np.zeros((new_length, num_of_channels))

        for i in range(data.shape[1]):
            f_resample[:, i] = (signal.resample(data[i], new_length, window='hamming'))

        return f_resample


def load_ecg_raw_data(filename: Path):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float32)
    new_file = str(filename).replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    # print(input_header_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


"""
The below functions are for PhysioNet's 2018 CinC Challenge: 
Snooze to Win.

The functions are used to read and load the .mat, .hea and -arousal.mat data.
"""


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res


# -----------------------------------------------------------------------------
# import the outcome vector, given the file name.
# e.g. /training/tr04-0808/tr04-0808-arousal.mat
# -----------------------------------------------------------------------------
def eeg_import_arousals(file_name):
    import h5py
    import numpy
    f = h5py.File(file_name, 'r')
    arousals = numpy.array(f['data']['arousals'])
    return arousals


def eeg_import_signals(file_name):
    return np.transpose(scipy.io.loadmat(file_name)['val'])


# -----------------------------------------------------------------------------
# Take a header file as input, and returns the names of the signals
# For the corresponding .mat file containing the signals.
# -----------------------------------------------------------------------------
def eeg_import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples


# -----------------------------------------------------------------------------
# Get a given subject's data
# -----------------------------------------------------------------------------
def eeg_get_subject_data(arousal_file, signal_file, signal_names):
    this_arousal = eeg_import_arousals(arousal_file)
    this_signal = eeg_import_signals(signal_file)
    this_data = np.append(this_signal, this_arousal, axis=1)
    this_data = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data


def eeg_get_subject_data_test(signal_file, signal_names):
    this_signal = eeg_import_signals(signal_file)
    this_data = this_signal
    this_data = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data


def eeg_limit_data(data, dataLimitInHours):
    sampleDataLimit = dataLimitInHours * 3600 * 200
    l_data = []
    if len(data[0].shape) == 1:
        for n in range(len(data)):
            # originalLength = train_data[n].shape[0] / (3600 * 50)
            if data[n].shape[0] < sampleDataLimit:
                # Zero Pad
                neededLength = sampleDataLimit - data[n].shape[0]
                extension = np.zeros(shape=(neededLength))
                extension[::] = -1.0
                data[n] = np.concatenate([data[n], extension], axis=0)
            elif data[n].shape[0] > sampleDataLimit:
                # Chop
                data[n] = data[n][0:sampleDataLimit]
            l_data.append(data[n])
    else:
        for n in range(len(data)):
            # originalLength = train_data[n].shape[0] / (3600 * 50)
            if data[n].shape[0] < sampleDataLimit:
                # Zero Pad
                neededLength = sampleDataLimit - data[n].shape[0]
                extension = np.zeros(shape=(neededLength, data[n].shape[1]))
                extension[::, -3::] = -1.0
                data[n] = np.concatenate([data[n], extension], axis=0)
            elif data[n].shape[0] > sampleDataLimit:
                # Chop
                data[n] = data[n][0:sampleDataLimit, ::]
            l_data.append(data[n])

    return l_data


def eeg_limit_data_single(data, dataLimitInHours):
    sampleDataLimit = dataLimitInHours * 3600 * 200

    if len(data.shape) == 1:
        if data.shape[0] < sampleDataLimit:
            # Zero Pad
            neededLength = sampleDataLimit - data.shape[0]
            extension = np.zeros(shape=(neededLength))
            extension[::] = -1.0
            data = np.concatenate([data, extension], axis=0)
        elif data.shape[0] > sampleDataLimit:
            # Chop
            data = data[0:sampleDataLimit]
        elif data.shape[0] == sampleDataLimit:
            pass

    else:
        if data.shape[0] < sampleDataLimit:
            # Zero Pad
            neededLength = sampleDataLimit - data.shape[0]
            extension = np.zeros(shape=(neededLength, data.shape[1]))
            extension[::] = -1.0
            data = np.concatenate([data, extension], axis=0)
        elif data.shape[0] > sampleDataLimit:
            # Chop
            data = data[0:sampleDataLimit]
        elif data.shape[0] == sampleDataLimit:
            pass

    return data


def split_single_eeg_arousals(signal, result, mask, window_length, arousal):
    x_final = []
    y_final = []
    positions = []
    if len(mask) == 0:
        windows = len(result[0]) // window_length
        for i in range(windows):
            idxs = list(result[0][i * window_length:(i + 1) * window_length])
            x_final.append(signal[idxs])
            if arousal:
                y_final.append(np.ones(window_length, dtype='int32'))
            else:
                y_final.append(np.zeros(window_length, dtype='int32'))
            positions.append((idxs[0], idxs[-1]))
    else:
        for j in range(len(mask)):
            if j == 0:
                windows = mask[j] // window_length
                if windows > 0:
                    for i in range(windows):
                        idxs = list(result[0][i * window_length:(i + 1) * window_length])
                        x_final.append(signal[idxs])
                        if arousal:
                            y_final.append(np.ones(window_length, dtype='int32'))
                        else:
                            y_final.append(np.zeros(window_length, dtype='int32'))
                        positions.append(idxs)
                else:
                    pass
            else:
                samples = mask[j] - mask[j - 1]
                windows = samples // window_length
                if windows > 0:
                    for i in range(windows):
                        idxs = list(result[0][
                                    mask[j - 1] + (window_length * i) + 1:mask[j - 1] + (window_length * (i + 1)) + 1])
                        x_final.append(signal[idxs])
                        y_final.append(np.ones(window_length, dtype='int32'))
                        positions.append(idxs)

    return x_final, y_final, positions


def load_eeg(file):
    header_file = file
    train_arousal = file.replace('.hea', '-arousal.mat')
    train_signal = file.replace('.hea', '.mat')

    x = eeg_import_signal_names(header_file)

    x[0].append('arousals')
    names = x[0].copy()

    data = eeg_get_subject_data(train_arousal, train_signal, names).to_numpy()
    train_data = data[:, 0:13]
    arousals = data[:, 13]

    # train_data = eeg_limit_data_single(train_data, 7)
    # arousals = eeg_limit_data_single(arousals, 7)

    train_data = normalize_channel_eeg(train_data)

    return train_data, arousals


def load_ecg(file: Path):
    wsize = 5000
    x = []
    data, header_data = load_ecg_raw_data(file)
    splitted = header_data[0].split()
    input_sampling_rate = int(splitted[2])
    data = resample_signal(data, 500, input_sampling_rate)
    # filt = Filter(500, 20, 0.01, 0.01, 3)
    # data = filt.augment_single(data)
    data = normalize_channel(data)
    out = pad_sequences(data, maxlen=wsize, dtype='float32', truncating='post', padding="post")
    out = out.reshape(wsize, 12).astype(np.float32)
    x.append(out)

    return x


def load_ecg_incart(file):
    x = []
    data, header_data = load_ecg_raw_data(file)
    splitted = header_data[0].split()
    input_sampling_rate = int(splitted[2])
    data = resample_signal(data, 500, input_sampling_rate)
    data = normalize_channel(data)
    data = data.reshape(len(data[1]), 12).astype(np.float32)
    x.append(data)

    return x


def extract_windows(array, positions, window_size, overlap_percent,
                    arousal, drop_remainding: bool = True):
    """
    Function which uses sliding windows to split an array.
    :param array: array to be split
    :param positions: positions of interest to be split
    :param window_size: window size of signal
    :param overlap_percent: sliding windows overlap percentage
    :param arousal: if EEG signal array contains arousals or not
    :param drop_remainding: boolean value. whether to drop remainding signal points after sliding windows
                            implementation
    :return: split signals, corresponding labels and signal time points to know their origin in the initial signal
    """
    out = []
    label = []
    pos = []
    if len(array) < window_size:
        return out, label, pos
    step = int(window_size * (int(100 - overlap_percent) / 100))
    start = 0
    end = len(array)
    process = True
    while process:
        window = list(range(start, start + window_size))
        cut = array[window]
        out.append(cut)
        pos.append(positions[start:start + window_size])
        if arousal:
            label.append(np.ones(window_size, dtype='int32'))
        else:
            label.append(np.zeros(window_size, dtype='int32'))
        start = start + step
        if (start + window_size) < end:
            if drop_remainding:
                continue
            else:
                window = list(range(end - window_size, end))
                cut = array[window]
                out.append(cut)
                pos.append(positions[end - window_size: end])
                if arousal:
                    label.append(np.ones(window_size, dtype='int32'))
                else:
                    label.append(np.zeros(window_size, dtype='int32'))
        else:
            process = False
    return out, label, pos


def split_sliding_window_eeg(signal, result, mask, window_length, arousal):
    """
    This function along with the 'extract_windows' function,
    uses sliding windows to split the EEG signal, based on the subject's arousal (awake) time periods.
    :param signal: EEG signal to be split
    :param result: binary array of arousal signal points
    :param mask: signal points of arousal time-periods
    :param window_length: length of extracted windows
    :param arousal: boolean value. whether the extracted windows will contain arousal periods or not
    :return: Split signal points, labels and indexes of split signal time points
    """
    x_final = []
    y_final = []
    positions = []
    if len(mask) == 0:
        windows = len(result[0]) // window_length
        for i in range(windows):
            idxs = list(result[0][i * window_length:(i + 1) * window_length])
            x_final.append(signal[idxs])
            if arousal:
                y_final.append(np.ones(window_length, dtype='int32'))
            else:
                y_final.append(np.zeros(window_length, dtype='int32'))
            positions.append((idxs[0], idxs[-1]))
    else:
        for j in range(len(mask)):
            if j == 0:
                idxs = list(result[0][0:mask[j]])
                array = signal[idxs]
                windows, labels, pos_final = extract_windows(array, idxs, window_length,
                                                             80, arousal=True,
                                                             drop_remainding=True)
                x_final.extend(windows)
                y_final.extend(labels)
                positions.extend(pos_final)
            else:
                idxs = list(result[0][mask[j - 1] + 1:mask[j] + 1])
                array = signal[idxs]
                windows, labels, pos_final = extract_windows(array, idxs, window_length,
                                                             80, arousal=True,
                                                             drop_remainding=True)
                x_final.extend(windows)
                y_final.extend(labels)
                positions.extend(pos_final)

    return x_final, y_final, positions


def get_ecg_annotations(file):
    """
    Function that returns a dictionary of the INCART ECG beat annotations.
    :param file: filePath of the patient's annotation file
    :return: Beat annotation dictionary
    """
    ann = wfdb.rdann(str(file), 'atr')
    labels = np.array(ann.__dict__['symbol'])
    sample = ann.__dict__['sample']
    annotations = {'Annotation': labels, 'Sample Index': sample}
    return annotations


def split_incart_ecg(x, label, annotations):
    """
    This function splits the INCART signals to 10 sec windows based on the 'abnormal beat annotations'.
    Specifically, when the first abnormal beat is detected, the window starts 0.5 sec before saif beat and
    ends 9.5 secs after. This way, the window is ensured to contain at least 1 abnormal beat.
    :param x: Incart signal to be splitted
    :param label: Label of signal
    :param annotations: Annotation dictionary, that contains the annotated beats of the ECG signal
    :return: List of splitted 10 sec windows of signal x and their label
    """
    x_final = []
    label_final = []
    ann = annotations
    indexes = ann['Sample Index'][np.where(ann['Annotation'] != 'N')]
    start = indexes[0]
    if start >= 250:
        start = start - 250  # start segmentation 0.5 sec before the abnormal beat
    end = start + 5000
    for i in indexes[1:]:
        if i <= end:
            continue
        else:
            x_final.append(x[start:end])
            label_final.append(label)
            start = i - 250
            end = start + 5000

    return x_final, label_final


def return_database(x: str):
    return {
        'CPSC': Database.CPSC,
        'CPSC_EXTRA': Database.CPSC_EXTRA,
        'GA': Database.GA,
        'PTB': Database.PTB,
        'PTBXL': Database.PTBXL,
        'PTRSB': Database.PTRSB
    }.get(x)


def get_model_summary(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary


def f1_score(history):
    precision = history['Precision']
    recall = history['Recall']

    f1 = 2 * ((precision * recall) / (precision + recall))

    return f1


def calculate_per_class_prediction_metrics(y, y_pred):
    # y_truth = [np.where(x == 1)[0] for x in y]
    # y_hat = [np.where(x == 1)[0] for x in y_pred]

    """Example of prediction and metrics.
    y_truth --> [12]
    y_pred --> [0, 4]

    In this case we have:
    - False-Positives for classes [0, 4]
    - False-Negatives for class 12
    - True-Negatives for each class except [0, 4, 12]
    - Zero True-Positives for class 12
    """

    pred_metrics = np.zeros((y.shape[1], 9), dtype=float)  # FP[0], FN[1], TP[2], TN[3]

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] == 1:
                pred_metrics[j, 8] += 1
            if y[i, j] == 0 and y_pred[i, j] == 1:  # FP
                pred_metrics[j, 0] += 1
            elif y[i, j] == 1 and y_pred[i, j] == 0:  # FN
                pred_metrics[j, 1] += 1
            elif y[i, j] == 1 and y_pred[i, j] == 1:  # TP
                pred_metrics[j, 2] += 1
            elif y[i, j] == 0 and y_pred[i, j] == 0:  # TN
                pred_metrics[j, 3] += 1

    for k in range(len(pred_metrics)):
        fp = pred_metrics[k][0]
        fn = pred_metrics[k][1]
        tp = pred_metrics[k][2]
        tn = pred_metrics[k][3]

        if fn + tp == 0 or fp + tp == 0 or tp == 0:
            pred_metrics[k][4] = str.split(snomed_classes[k], ',')[0]
            pred_metrics[k][5] = 0.0
            pred_metrics[k][6] = 0.0
            pred_metrics[k][7] = 0.0
        else:
            # accuracy = (tp + tn) / (fp + fn + tp + tn)
            precision = tp / (fp + tp)
            recall = tp / (fn + tp)
            f1 = 2 * ((precision * recall) / (precision + recall))

            # metrics_df['Accuracy'][k] = accuracy
            pred_metrics[k][4] = str.split(snomed_classes[k], ',')[0]
            pred_metrics[k][5] = precision
            pred_metrics[k][6] = recall
            pred_metrics[k][7] = f1

    metrics_df = pd.DataFrame(pred_metrics, columns=['FP', 'FN', 'TP', 'TN',
                                                     'class', 'Precision', 'Recall', 'F1', 'Total Labels'], dtype=float)

    return metrics_df


def calculate_per_class_prediction_metrics_new(y, y_pred):
    # y_truth = [np.where(x == 1)[0] for x in y]
    # y_hat = [np.where(x == 1)[0] for x in y_pred]

    """Example of prediction and metrics.
    y_truth --> [12]
    y_pred --> [0, 4]

    In this case we have:
    - False-Positives for classes [0, 4]
    - False-Negatives for class 12
    - True-Negatives for each class except [0, 4, 12]
    - Zero True-Positives for class 12
    """
    confusion_matrices = multilabel_confusion_matrix(y, y_pred)

    pred_metrics = np.zeros((27, 8), dtype=float)  # FP[0], FN[1], TP[2], TN[3]

    for k in range(len(pred_metrics)):
        tn = pred_metrics[k][3] = confusion_matrices[k][0][0]
        fp = pred_metrics[k][0] = confusion_matrices[k][0][1]
        fn = pred_metrics[k][1] = confusion_matrices[k][1][0]
        tp = pred_metrics[k][2] = confusion_matrices[k][1][1]

        if (fn + tp == 0) or (fp + tp == 0) or (tp == 0):
            pred_metrics[k][4] = int(snomed_classes[k])
            pred_metrics[k][5] = 0.0
            pred_metrics[k][6] = 0.0
            pred_metrics[k][7] = 0.0
        else:
            # accuracy = (tp + tn) / (fp + fn + tp + tn)
            precision = tp / (fp + tp)
            recall = tp / (fn + tp)
            f1 = 2 * ((precision * recall) / (precision + recall))

            # metrics_df['Accuracy'][k] = accuracy
            pred_metrics[k][4] = snomed_classes[k]
            pred_metrics[k][5] = precision
            pred_metrics[k][6] = recall
            pred_metrics[k][7] = f1

    metrics_df2 = pd.DataFrame(pred_metrics, columns=['FP', 'FN', 'TP', 'TN',
                                                      'class', 'Precision', 'Recall', 'F1'], dtype=float)

    return metrics_df2


def write_to_file(path, array):
    x = array.tolist()
    with open(path, 'w') as fp:
        for item in x:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
