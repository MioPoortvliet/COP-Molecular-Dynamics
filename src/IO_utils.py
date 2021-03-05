import numpy as np
import os
import json
import shutil


def ensure_dir(file_path: str) -> None:
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def del_dirs(path: str) -> None:
    """REMOVES SPECIFIED DIR!"""
    print(f"Removing {path}.")
    shutil.rmtree(path)


def load_and_concat(fpath: str, file_identifier:str) -> np.ndarray:
    arrays = []
    files = [f for f in os.listdir(fpath) if f[:len(file_identifier)] == file_identifier]
    file_numbers = np.array([int(f.replace(file_identifier, '').replace('-', '').replace('.npy', '')) for f in files])
    files = [files[i] for i in file_numbers.argsort()]

    for file in files:
        arrays.append(np.load(fpath+file, allow_pickle=True))

    return np.concatenate(arrays)


def load_json(fpath: str, fname="00-header.json") -> dict:
    with open(fpath+fname) as json_file:
        data = json.load(json_file)

    return data


def to_file(fpath: str, data: object) -> object:
    with open(fpath + ".npy", 'wb') as file:
        np.save(file, data)