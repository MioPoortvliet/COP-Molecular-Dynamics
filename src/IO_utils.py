import numpy as np
import tables
import os
import re


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def load_and_concat(fpath, file_identifier):
    arrays = []
    files = [f for f in os.listdir(fpath) if f[:len(file_identifier)] == file_identifier]
    file_numbers = np.array([int(f.replace(file_identifier, '').replace('-', '').replace('.npy', '')) for f in files])
    files = [files[i] for i in file_numbers.argsort()]

    for file in files:
        arrays.append(np.load(fpath+file, allow_pickle=True))

    return np.concatenate(arrays)


def to_file(fpath, data):
    print("Writing to " + fpath)
    with open(fpath + ".npy", 'wb') as file:
        np.save(file, data)