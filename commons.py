from enum import Enum

class DataType(Enum):
    RAWFILES = 0
    PICKLE = 1
    TFRECORDS = 2


class SubsetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class BioSignal(Enum):
    ECG = 0
    EEG = 1
    PCG = 2


class Biosignal(Enum):
    ECG = 0
    PHONO_2016 = 1
    PHONO_2022 = 2
    EEG = 3


class Database(Enum):
    """Data File Formats."""
    CPSC = 'A'
    CPSC_EXTRA = 'Q'
    GA = 'E'
    PTB = 'S'
    PTBXL = 'HR'
    PTRSB = 'I'


class EEGHoldout(Enum):
    CHINA = 'h_china'
    FRA = 'h_fra'
    GER = 'h_ger'
