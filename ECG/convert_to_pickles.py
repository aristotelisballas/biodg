import os
import pickle
import sys
from pathlib import Path

import numpy as np
from absl import app, flags

import ECG.bioconfig as bioconf
import ECG.utils as utils

from commons import DataType, BioSignal
from ECG.ecgmetadata import BioMetadata

FLAGS = flags.FLAGS
flags.DEFINE_string('hostname', 'telis',
                    'The name of the machine to use, to enable executing in different environments')
flags.DEFINE_string('signal', 'ECG', 'The biosignal data to convert to .pkl')
flags.DEFINE_string('outpath', './', 'Output path of the converted data')

""" ---General Info/Instructions---

    This script converts the initial .mat files of the ECG dataset to pickle (.pkl) files.
    To convert the files, change the 'HOSTNAME' flag, which is used for the filePath scheme (filePaths configured
    in 'ECG/config.py').

    ** Data **
    The datasets can be found at:
    -ECG: https://physionetchallenges.org/2020/
    """


def convert_ecg_to_pkl(filepaths, labels):
    sum = 0
    outPath = Path(FLAGS.outpath)

    if not os.path.exists(outPath):
        os.mkdir(outPath)

    for file, label in zip(filepaths, labels):
        file = Path(file)
        split = file.parts[-1]
        origin = np.char.split(split, sep=".").tolist()[0]

        if 'I' in origin and (label[13] == 1):      # Will only split the INCART data according to 'abnormal'
            x = utils.load_ecg_incart(file)[0]      # beats only if the signal has an 'abnormal' label.
            atr = origin[0] + origin[-2:]
            annotation_file = Path(bioconf.get_data_ecg_annotation_files()) / atr
            annotations = utils.get_ecg_annotations(annotation_file)
            x_split, label_split = utils.split_incart_ecg(x, label, annotations)

            noof: int = len(x_split)
            for i in range(noof):
                split_num = origin + f"-{i}"
                name = outPath / split_num
                outfile = f'{name}.pkl'
                file = open(outfile, 'wb')
                obj = {
                    'data': x_split[i],
                    'label': label_split[i]
                }
                pickle.dump(obj, file)
                file.close()
            sum += noof

        else:
            x = utils.load_ecg(file)[0]
            outfile = outPath / f'{origin}.pkl'
            file = open(outfile, 'wb')
            obj = {
                'data': x,
                'label': label
            }
            pickle.dump(obj, file)
            file.close()
            sum += 1

    print(f"Wrote {sum} elements to pickles")

    return sum


def main(args):
    del args

    bioconf.set_host(FLAGS.HOSTNAME)

    md = BioMetadata(DataType.RAWFILES, BioSignal.ECG)
    convert_ecg_to_pkl(md.ecg_filenames, md.labels1hot)


if __name__ == '__main__':
    app.run(main)
