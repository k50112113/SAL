import numpy as np
import argparse
import sys
import os
from ase.io import read, write
from ase import Atoms

def main():
    
    parser = argparse.ArgumentParser(description='Back track Subascent trajectory')
    parser.add_argument('iteration',  type = int,   help = "iteration")
    parser.add_argument('job',        type = int,   help = "Subascent job index")
    args = parser.parse_args()

    itr = args.iteration
    subascent_index = args.job
    init_itr = itr
    init_subascent_index = subascent_index
    sample_traj_path_list = []
    
    print("Back tracking Subascent trajectory")
    while True:
        sample_traj_path = f"itr_{itr}/sample_{subascent_index}_trajectory.extxyz"
        log_filename = f"itr_{itr}/log"
        assert os.path.isfile(sample_traj_path), "trajectory file not found"
        assert os.path.isfile(log_filename), "log file not found"
        print(f"itr: {itr}, subascent: {subascent_index} -> ", end = "")
        sample_traj_path_list.append(sample_traj_path)

        logline = os.popen(f"grep \"Subascent sampling job {subascent_index}\" {log_filename}").read().split('\n')
        logline.pop()
        loglinetmp = logline[0].strip().split(":")[-1].strip().split(",")
        prev_sample_filename_path = loglinetmp[0]
        prev_frame_index = int(loglinetmp[-1].split(" ")[-1])
        
        if "abinitio_save_output" in prev_sample_filename_path:
            filename_split = prev_sample_filename_path.split('/')
            prev_itr = int(filename_split[0].split("_")[-1])
            prev_subascent_index = int(filename_split[-1].split("_")[1])
        else:
            print(f"{prev_sample_filename_path}, frame {prev_frame_index}")
            print("Back tracking done")
            break

        itr = prev_itr
        subascent_index = prev_subascent_index

    print("\nConcatenating Subascent trajectories...")
    sample_traj_path_list = sample_traj_path_list[::-1]
    data = []
    for sample_traj_path in sample_traj_path_list:
        print("\t" + sample_traj_path)
        datatmp = read(sample_traj_path, format = "extxyz", index = ":")
        data += datatmp
    output_path = f"cat_{init_itr}_{init_subascent_index}.extxyz"
    print(f"Final trajectory saved to {output_path}")
    write(output_path, data, format = "extxyz")