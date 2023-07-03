from pathlib import Path
from typing import NoReturn

import sys

# SNOMED LABELS
# snomed_classes = ['10370003', '111975006', '164889003', '164890007', '164909002', '164917005',
#                   '164934002', '164947007', '17338001', '251146004', '270492004', '284470004',
#                   '39732003', '426177001', '426627000', '426783006', '427084000', '427172004',
#                   '427393009', '445118002', '47665007', '59118001', '59931005', '63593006',
#                   '698252002', '713426002', '713427006']
snomed_classes = ['10370003', '111975006', '164889003', '164890007', '164909002', '164917005',
                  '164934002', '164947007', '251146004', '270492004',
                  '39732003', '426177001', '426627000', '426783006', '427084000', '427172004, 17338001',
                  '427393009', '445118002', '47665007', '59931005', '63593006, 284470004',
                  '698252002', '713426002', '713427006,59118001']

snomed_classes_dict = {
    '10370003': 0,
    '111975006': 1,
    '164889003': 2,
    '164890007': 3,
    '164909002': 4,
    '164917005': 5,
    '164934002': 6,
    '164947007': 7,
    '251146004': 8,
    '270492004': 9,
    '39732003': 10,
    '426177001': 11,
    '426627000': 12,
    '426783006': 13,
    '427084000': 14,
    '427172004': 15,
    '17338001': 15,
    '427393009': 16,
    '445118002': 17,
    '47665007': 18,
    '59931005': 19,
    '63593006': 20,
    '284470004': 20,
    '698252002': 21,
    '713426002': 22,
    '713427006': 23,
    '59118001': 23
}

# Configuration
_hostname: str

# Variables
_root_ecg_path: Path
_data_ecg_root_folder_path: Path
_data_ecg_pickle_path: Path
_data_ecg_snomed_codes: Path
_tfr_ecg_root_path: Path

_root_eeg_path: Path
_data_eeg_root_folder_path: Path
_data_eeg_pickle_path: Path
_data_ecg_snomed_unscored_codes: Path
_data_ecg_annotation_files: Path
_tfr_eeg_root_path: Path
_tfr_eeg_neg_path: Path
_tfr_eeg_pos_path: Path

_sample_rate: int = 500

pickle_data_dir = Path('/home/user/data/biodg/ecg/files/')
ecg_results_dir = Path('/home/user/experiments/biodg/ecg/')


def set_host(hostname: str) -> NoReturn:
    """
    Set the host name to allow easy execution on different machines.

    :param hostname: The name of the host/user. Currently supported values: telis, gpunode0, hua.
    """
    assert isinstance(hostname, str)

    global _hostname

    global _root_ecg_path
    global _data_ecg_root_folder_path
    global _data_ecg_pickle_path
    global _data_ecg_snomed_codes
    global _tfr_ecg_root_path

    global _root_eeg_path
    global _data_eeg_root_folder_path
    global _data_eeg_pickle_path
    global _data_ecg_snomed_unscored_codes
    global _data_ecg_annotation_files
    global _tfr_eeg_root_path
    global _tfr_eeg_neg_path
    global _tfr_eeg_pos_path

    global _results_path

    _hostname = hostname

    if hostname == 'user':
        scripts_root: Path = Path('/home/aballas/git/biosignals/')
        
        ### ECG ###
        _root_ecg_path = Path('/home/aballas/data/biodg/ecg/')

    elif hostname == 'other':

        scripts_root: Path = Path('')

        ### ECG ###
        _root_ecg_path = scripts_root / 'ECG' / 'data'


    else:
        raise ValueError('Unknown host: ' + hostname)

    # ECG
    # _data_ecg_root_folder_path: str = "physionet\\data"
    _data_ecg_root_folder_path = str(_root_ecg_path)
    _data_ecg_snomed_codes = str(scripts_root / 'ECG' / 'datafiles' / 'Dx_map.csv')
    _data_ecg_snomed_unscored_codes = str(_root_ecg_path / 'dx_mapping_unscored.csv')
    _data_ecg_annotation_files = str(_root_ecg_path / 'annotations')


def get_raw_ecg_data_path() -> Path:
    return _data_ecg_root_folder_path


def get_ecg_data_pickle() -> Path:
    return _data_ecg_pickle_path


def get_raw_eeg_data_path() -> Path:
    return _data_eeg_root_folder_path


def get_eeg_data_pickle() -> Path:
    return _data_eeg_pickle_path


def get_snomed_data_path() -> Path:
    return _data_ecg_snomed_codes


def get_snomed_unscored_data_path() -> Path:
    return _data_ecg_snomed_unscored_codes


def get_data_ecg_annotation_files() -> Path:
    return _data_ecg_annotation_files
