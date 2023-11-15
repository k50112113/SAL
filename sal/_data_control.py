import argparse
import numpy as np
import sys
from ase.io import read, write
from sal._process_control import write_log
import warnings
warnings.filterwarnings("ignore")

model_engine_list = ['nequip']
abinitio_engine_list = ['ase']


def merge(filename_list, nframe_list, random_seed, logfile_path):
    np.random.seed(random_seed)
    merged_data = []
    total_n_frame_list = []
    for i_file, filename in enumerate(filename_list):
        data = read(filename, format = "extxyz", index = ":")
        total_n_frame = len(data)
        total_n_frame_list.append(total_n_frame)
        nframe = nframe_list[i_file]
        if nframe == -1: nframe = len(data)
        else:            nframe = min(nframe_list[i_file], total_n_frame)
        select_frame_indices = np.arange(len(data))
        np.random.shuffle(select_frame_indices)
        select_frame_indices = select_frame_indices[:nframe]
        write_log(f"Select {nframe} frames from {filename}", logfile_path)
        merged_data += [data[v] for v in select_frame_indices]
    
    indices = np.arange(len(merged_data))
    np.random.shuffle(indices)
    merged_data = [merged_data[v] for v in indices]

    return merged_data, total_n_frame_list

def split(data, number_of_sets):
    total_n_frame = len(data)
    number_of_samples = total_n_frame // number_of_sets
    remainder = total_n_frame - number_of_samples * number_of_sets

    splitted_data = []
    start_frame_index = 0
    i_set = 0
    while i_set < number_of_sets:
        end_frame_index = start_frame_index + number_of_samples
        if remainder < i_set: end_frame_index += 1
        end_frame_index = min(end_frame_index, total_n_frame)
        splitted_data.append(data[start_frame_index:end_frame_index])
        start_frame_index = end_frame_index
        i_set += 1

    return splitted_data

def dump(filename, data, logfile_path):
    nframe = len(data)
    write_log(f"Output file saved to: {filename}, {nframe} frames", logfile_path)
    for i in range(nframe):
        data[i].set_array('var', None)
        data[i].set_array('magmoms', None)
        data[i].calc = None
        data[i].info.pop('dipole', None)
        data[i].info.pop('magmom', None)
    write(filename, data, format = "extxyz")
