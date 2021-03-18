"""
Helper functions for input/output.
Authors: Mio Poortvliet, Jonah Post
"""
import numpy as np
import os
import json
import shutil


def ensure_dir(file_path: str) -> None:
	if not os.path.exists(file_path):
		os.makedirs(file_path)


def del_dir(path: str) -> None:
	"""
	REMOVES SPECIFIED DIR!
	:param path: dir to be removed
	:type path: str
	:return: None
	:rtype: None
	"""
	print(f"Removing {path}.")
	shutil.rmtree(path)


def load_and_concat(fpath: str, file_identifier:str) -> np.ndarray:
	"""
	Load files in filepath that contain file_identifier in order, then concatenate them to one large array.
	:param fpath: filepath of dir containing files
	:type fpath: str
	:param file_identifier: identifier of specific files
	:type file_identifier: str
	:return: array of loaded files
	:rtype: np.ndarray
	"""
	arrays = []
	files = [f for f in os.listdir(fpath) if f[:len(file_identifier)] == file_identifier]
	file_numbers = np.array([int(f.replace(file_identifier, '').replace('-', '').replace('.npy', '')) for f in files])
	files = [files[i] for i in file_numbers.argsort()]

	for file in files:
		arrays.append(np.load(fpath+file, allow_pickle=True))

	return np.concatenate(arrays)


def load_json(fpath: str, fname="00-header.json") -> dict:
	"""
	Loads a json file from fpath with fname
	:param fpath: path to dir where file is located
	:type fpath: str
	:param fname: filename of json file
	:type fname: str
	:return: contents of json file as a nestled dict
	:rtype: dict
	"""
	with open(fpath+fname) as json_file:
		data = json.load(json_file)

	return data


def to_file(fpath: str, data: object) -> None:
	"""
	Save data to file in fpath
	:param fpath: filepath of file to be written
	:type fpath: str
	:param data: data to be saved
	:type data: object
	:return: None
	:rtype: None
	"""
	with open(fpath + ".npy", 'wb') as file:
		np.save(file, data)



def cleanup_paths(paths):
	"""Deletes dirs in the list paths."""
	for path in paths:
		del_dir(path)