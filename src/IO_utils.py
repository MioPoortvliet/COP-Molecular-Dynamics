import numpy as np
import tables
import os


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def load_and_concat(fpath, file_identifier):
    arrays = []
    for file in [f for f in os.listdir(fpath) if f[:len(file_identifier)] == file_identifier]:
        arrays.append(np.load(fpath+file, allow_pickle=True))
    return np.concatenate(arrays)