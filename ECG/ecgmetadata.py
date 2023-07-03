import sys

sys.path.append('/ecg')
import os
import pickle
import platform
import re

import numpy as np
import numpy.core.defchararray as np_f
import pandas as pd

import ECG.bioconfig as biocfg
from commons import DataType, BioSignal, Database
from ECG.utils import find, load_ecg_raw_data


class BioMetadata:
    def __init__(self, datatype: DataType, biosignal: BioSignal, holdout: Database = None):
        assert isinstance(datatype, DataType)
        assert isinstance(holdout, Database) or holdout is None or isinstance(holdout, list)
        assert isinstance(biosignal, BioSignal)

        self.datatype = datatype
        self.biosignal = biosignal
        self.holdout = holdout

        if biosignal is BioSignal.ECG and datatype is not DataType.TFRECORDS:
            if datatype is DataType.PICKLE:
                ecg_datapath = biocfg.get_ecg_data_pickle()
                f = open(ecg_datapath, 'rb')
                self.gender, self.age, self.labels, self.ecg_filenames = pickle.load(f)
                f.close()
                self.ecg_filenames = np_f.replace(self.ecg_filenames, str(biocfg._root_ecg_path_orig),
                                                  str(biocfg._root_ecg_path))
                if platform.system() == 'Linux':
                    self.ecg_filenames = np_f.replace(self.ecg_filenames, '\\', '/')
                self.labels = np.array(self.labels)
            elif datatype is DataType.RAWFILES:
                ecg_datapath = biocfg.get_raw_ecg_data_path()
                print('~~~ Iterating through RAW ECG data Files! ~~~')
                self.gender, self.age, self.ecg_filenames, self.labels = _load_ecg_key_data(ecg_datapath)
                self.labels = np.array(self.labels)
            else:
                raise ValueError("Unsupported data_type: " + str(datatype))

            snomed_codes = pd.read_csv(biocfg.get_snomed_data_path())
            SNOMED_unscored = pd.read_csv(biocfg.get_snomed_unscored_data_path())

            clean_files = []
            labels1hot = []
            for file, labels in zip(self.ecg_filenames, self.labels):
                one_hot_vec = np.zeros(24, dtype=int)
                labels = str.split(labels, ',')
                # clean = True
                for label in labels:
                    if label in biocfg.snomed_classes_dict.keys():
                        index = biocfg.snomed_classes_dict.get(label)
                        one_hot_vec[index] = 1
                if np.max(one_hot_vec) == 1:
                    clean_files.append(file)
                    labels1hot.append(one_hot_vec)

            self.ecg_filenames = clean_files
            self.labels1hot = np.array(labels1hot)


def _load_ecg_key_data(path):
    gender = []
    age = []
    labels = []
    ecg_filenames = []
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_ecg_raw_data(filepath)
                labels.append(header_data[15][5:-1])
                ecg_filenames.append(filepath)
                gender.append(header_data[14][6:-1])
                age.append(header_data[13][6:-1])
                print(filepath, end="\r")
    return gender, age, ecg_filenames, labels


# -----------------------------------------------------------------------------
# returns a list of the training and testing EEG file locations for easier import
# -----------------------------------------------------------------------------
def _eeg_get_files_old(rootDir):
    header_loc, arousal_loc, signal_loc, is_training = [], [], [], []
    for dirName, subdirList, fileList in os.walk(rootDir, followlinks=True):
        if not dirName.endswith('\\data') and not dirName.endswith('\\test') and not dirName.endswith('\\training'):
            if 'training' in dirName:
                is_training.append(True)
                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '\\' + fname)
                    if '-arousal.mat' in fname:
                        arousal_loc.append(dirName + '\\' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '\\' + fname)

            elif 'test' in dirName:
                is_training.append(False)
                arousal_loc.append('')

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '\\' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '\\' + fname)

    # combine into a data frame
    data_locations = {'header': header_loc,
                      'arousal': arousal_loc,
                      'signal': signal_loc,
                      'is_training': is_training
                      }

    # Convert to a data-frame
    df = pd.DataFrame(data=data_locations)

    # Split the data frame into training and testing sets.
    tr_ind = list(find(df.is_training.values))
    te_ind = list(find(df.is_training.values == False))

    training_files = df.loc[tr_ind, :]
    testing_files = df.loc[te_ind, :]

    return training_files, testing_files


def _eeg_get_files(rootDir):
    header_loc, arousal_loc, signal_loc, is_training = [], [], [], []
    for dirName, subdirList, fileList in os.walk(rootDir, followlinks=True):
        if not dirName.endswith('\\data') and not dirName.endswith('\\test') and not dirName.endswith('\\training'):
            if 'training' in dirName:
                is_training.append(True)
                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '\\' + fname)
                    if '-arousal.mat' in fname:
                        arousal_loc.append(dirName + '\\' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '\\' + fname)

            elif 'test' in dirName:
                is_training.append(False)
                arousal_loc.append('')

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '\\' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '\\' + fname)

    # combine into a data frame
    data_locations = {'header': header_loc,
                      'arousal': arousal_loc,
                      'signal': signal_loc,
                      'is_training': is_training
                      }

    # Convert to a data-frame
    df = pd.DataFrame(data=data_locations)

    # Split the data frame into training and testing sets.
    tr_ind = list(find(df.is_training.values))
    te_ind = list(find(df.is_training.values == False))

    training_files = df.loc[tr_ind, :]
    testing_files = df.loc[te_ind, :]

    return training_files, testing_files


def _get_tfrecords(rootDir):
    filenames = []
    for subdir, dirs, files in sorted(
            os.walk(rootDir)):
        for file in files:
            if '.tfrecords' in file:
                filenames.append(subdir + os.sep + file)

    return np.array(filenames)


def _get_tfrecords_ecg(rootDir, holdout):
    filenames = []
    holdout_filenames = []
    if holdout is None:
        for subdir, dirs, files in sorted(
                os.walk(rootDir)):
            for file in files:
                if '.tfrecords' in file:
                    filenames.append(subdir + os.sep + file)

        return np.array(filenames), np.array(holdout_filenames)

    for subdir, dirs, files in sorted(
            os.walk(rootDir)):
        for file in files:
            if '.tfrecords' in file:
                delim = "\\", "/"
                regexPattern = '|'.join(map(re.escape, delim))
                f = re.split(regexPattern, file)[-1]
                if holdout.value not in f:
                    filenames.append(subdir + os.sep + file)
                if holdout.value in f:
                    holdout_filenames.append(subdir + os.sep + file)

    return np.array(filenames), np.array(holdout_filenames)


def _get_tfrecords_ecg_list(rootDir, holdout):
    filenames = []
    holdout_filenames = []
    holdout_values = []
    for i in holdout:
        holdout_values.append(i.value)

    if holdout is None:
        for subdir, dirs, files in sorted(
                os.walk(rootDir)):
            for file in files:
                if '.tfrecords' in file:
                    filenames.append(subdir + os.sep + file)

        return np.array(filenames), np.array(holdout_filenames)

    for subdir, dirs, files in sorted(
            os.walk(rootDir)):
        for file in files:
            if '.tfrecords' in file:
                delim = "\\", "/"
                regexPattern = '|'.join(map(re.escape, delim))
                f = re.split(regexPattern, file)[-1]
                if not any(value in f for value in holdout_values):
                    filenames.append(subdir + os.sep + file)
                if any(value in f for value in holdout_values):
                    holdout_filenames.append(subdir + os.sep + file)

    return np.array(filenames), np.array(holdout_filenames)
