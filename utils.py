import os
import pickle
import time
import copy

import torch
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from torchmetrics import ConfusionMatrix

from ECG.bioconfig import snomed_classes
from commons import BioSignal


def get_files(source_dir, datafile):
    file1 = open(Path(datafile), 'r')
    lines = file1.readlines()
    file1.close()
    lines = [line.rstrip('\n') for line in lines]
    return [os.path.join(source_dir, file) for file in lines]


def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def get_datafile_dir(biosignal: BioSignal = None, holdout: str = None):
    if biosignal == BioSignal.ECG:
        datafile_dir = Path('ECG/datafiles')

    elif biosignal == BioSignal.PCG:
        datafile_dir = Path('pcg_tmp/PCG_new/datafiles')

    elif biosignal == BioSignal.EEG:

        if holdout in ['CHINA', 'china']:
            datafile_dir = Path('EEG/datafiles/h_china')
        elif holdout == 'FRA':
            datafile_dir = Path('EEG/datafiles/h_fra')
        elif holdout == 'GER':
            datafile_dir = Path('EEG/datafiles/h_ger')
        else:
            warnings.warn('Holdout dataset not found. Please select one of the following: CHINA, FRA or GER !')
            datafile_dir = ''
    else:
        warnings.warn('Biosignal not supported. Please select one of the following: ECG, EEG, PCG !')
        datafile_dir = ''

    return datafile_dir


def calculate_per_class_prediction_metrics_ecg(y, y_pred):
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
        # print(i)
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


def torch_conf_matrix(y, y_pred, n_classes, device, multilabel: bool = False):

    """
    Calculates the conf_matrix of predicted vs true labels.
    :param y: true labels
    :param y_pred: predicted labels
    :param n_classes: # of classes
    :param device: torch.device --> 'cuda' or 'cpu'
    :param multilabel: whether the classification problem is mutlilabel
    :return: NxN confusion matrix: Columns are Predicted Conditions and rows are Actual Conditions
    """

    cmat = ConfusionMatrix(num_classes=n_classes, multilabel=multilabel)
    cmat = cmat.to(device)
    y = y.to(device)
    _, y_pred = torch.max(y_pred, 1)
    y_pred = y_pred.to(device)

    confmatrix = cmat(y_pred, y)

    return confmatrix


def dump_to_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)