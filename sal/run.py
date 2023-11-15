
import numpy as np
import argparse
import yaml
import sys
import os
import warnings
import datetime
warnings.filterwarnings("ignore")
from ase.io import read, write
from ase import Atoms
from sal._data_control import merge, split, dump
from sal._process_control import wait_all_proc_end, write_log, run_command, go_to_path, next_cuda_card
from sal._process_control import Clock

def main():

    parser = argparse.ArgumentParser(description='Subascent active learning')
    parser.add_argument('filename',  type = str,   help = "*.yaml")
    args = parser.parse_args()

    with open(args.filename) as fin:
        sal_yaml_file = yaml.load(fin, Loader=yaml.FullLoader)
        initial_data_path                         = sal_yaml_file['initial_data_path']
        cuda_card_list                            = sal_yaml_file['cuda_card_list']
        random_seed                               = sal_yaml_file['random_seed']
        dtype                                     = sal_yaml_file['dtype']

        max_wall_time_per_command                 = sal_yaml_file['max_wall_time_per_command']

        start_itr                                 = sal_yaml_file['start_itr']
        end_itr                                   = sal_yaml_file['end_itr']

        stop_if_iteration_exist                   = sal_yaml_file.get("stop_if_iteration_exist", True)

        mlip_train_model_command                  = sal_yaml_file['mlip']['train_model_command']
        mlip_clone_file                           = sal_yaml_file['mlip']['clone_file']
        mlip_model_name                           = sal_yaml_file['mlip']['model_name']
        mlip_number_of_ensemble_model             = sal_yaml_file['mlip']['number_of_ensemble']
        mlip_number_of_frames_from_initial_data   = sal_yaml_file['mlip']['number_of_frames_from_initial_data']
        mlip_number_of_frames_for_itr_0           = sal_yaml_file['mlip']['number_of_frames_for_itr_0']

        subascent_sample_initial_data_probability = sal_yaml_file['subascent']['sample_initial_data_probability']
        subascent_sample_data_probability_distribution = sal_yaml_file['subascent'].get('sample_data_probability_distribution', "uniform")
        subascent_sample_data_exp_coeff = sal_yaml_file['subascent'].get('sample_data_exp_coeff', 1)

        subascent_number_of_frames_to_sample         = sal_yaml_file['subascent']['number_of_frames_to_sample']
        subascent_parameters                         = sal_yaml_file['subascent']['parameters']
        subascent_steepness_parameters               = sal_yaml_file['subascent']['steepness_parameters']
        assert len(subascent_steepness_parameters) > 0, "steepness_parameters cannot be empty"
        assert all(isinstance(v, list) and len(v) > 0 for v in subascent_steepness_parameters), "steepness_parameters should be a list of non-empty lists of steepness parameters"
        subascent_steepness_parameters_sampling_type = sal_yaml_file['subascent'].get('steepness_parameters_sampling_type', "random")
        assert subascent_steepness_parameters_sampling_type in ["random", "exhaust"], "subascent_steepness_parameters_sampling_type not recognized."
        subascent_sample_target_filename             = sal_yaml_file['subascent'].get("sample_target_filename", "")
        
        abinitio = sal_yaml_file['abinitio']

        date_str = datetime.datetime.now().strftime("%Y%m%d_%X").replace("/","")
        saved_filename = args.filename.replace(".yaml","")
        saved_filename = f"{saved_filename}_{date_str}.yaml"
        with open(saved_filename, 'w') as f:
            yaml.dump(sal_yaml_file, f, default_flow_style=False, sort_keys=False)
            
        subascent_parameters['dtype'] = dtype

    np.random.seed(random_seed)
    randint_range = (0, 100000)
    
    wait_interval = 10
    wall_time     = max_wall_time_per_command
    root_dir_path = os.getcwd()

    mlip_ensemble_label = list(range(mlip_number_of_ensemble_model))
    sleep_interval = 10
    cuda_card_idx  = -1
    proc_list = []

    timer = Clock()

    for itr in range(start_itr, end_itr + 1):
        go_to_path(root_dir_path)
        itr_dir_path = f"{root_dir_path}/itr_{itr}"
        if stop_if_iteration_exist == True:
            assert not os.path.isdir(itr_dir_path), f"{itr_dir_path} exists, nothing to do"
        logfile_path = f"{itr_dir_path}/log"
        run_command(f"mkdir -p {itr_dir_path}", wait = True)
        run_command(f"touch {logfile_path}", wait = True)

        # Gather data
        data_filename_list = [initial_data_path]
        nframe_list        = [mlip_number_of_frames_from_initial_data if itr > 0 else mlip_number_of_frames_for_itr_0]
        abinitio_save_output_path = "abinitio_save_output"
        for i_itr in range(itr):
            sample_file_list = os.popen(f"find itr_{i_itr}/{abinitio_save_output_path} -name \"*.extxyz\"").read().split('\n')
            sample_file_list.pop()
            data_filename_list += sample_file_list
            nframe_list += [-1] * len(sample_file_list)
        write_log("List of data:", logfile_path)
        for a_data_filename in data_filename_list:
            write_log(f"{a_data_filename}", logfile_path)
        write_log("", logfile_path)
        
        # Merge and split data
        rseed = np.random.randint(*randint_range)
        write_log(f"merge data random seed: {rseed}", logfile_path)
        merged_data, total_n_frame_list = merge(data_filename_list, nframe_list, rseed, logfile_path)
        splitted_data = split(merged_data, mlip_number_of_ensemble_model)
        ensemble_dir_path_list = [f"{itr_dir_path}/ensemble_{m}" for m in mlip_ensemble_label]
        if all(os.path.isfile(f"{v}/{mlip_model_name}") for v in ensemble_dir_path_list):
            model_exist = True
        else:
            model_exist = False
            for m in mlip_ensemble_label:
                run_command(f"rm -rf {ensemble_dir_path_list[m]}", wait = True)
                run_command(f"mkdir -p {ensemble_dir_path_list[m]}", wait = True)
                dump(f"{ensemble_dir_path_list[m]}/data.extxyz", splitted_data[m], logfile_path)
        
        if os.path.isfile("salend"):
            write_log("Force terminate", logfile_path)
            exit()
        write_log("Merge and split data completed", logfile_path)
        dt = "%.3f"%(timer.get_dt())
        write_log(f"Merge and split data time: {dt} s", logfile_path)
        write_log("", logfile_path)

        if model_exist == False:
            # Prepare input files for model training
            for m in mlip_ensemble_label:
                for a_clone_file in mlip_clone_file:
                    write_log(f"copy {a_clone_file} to {ensemble_dir_path_list[m]}", logfile_path)
                    run_command(f"cp -r {a_clone_file} {ensemble_dir_path_list[m]}", wait = True)
            
            # Train ensemble models
            proc_list = []
            for m in mlip_ensemble_label:
                go_to_path(ensemble_dir_path_list[m])

                outfile = f"{itr_dir_path}/train_ensemble_{m}.out"
                cuda_card_idx, cuda_card = next_cuda_card(cuda_card_idx, cuda_card_list)

                proc = run_command(f"export CUDA_VISIBLE_DEVICES={cuda_card} ; {mlip_train_model_command} > {outfile} 2>&1")
                write_log(f"Train model_{m} using cuda:{cuda_card}", logfile_path)
                
                proc_list.append(proc)
            go_to_path(root_dir_path)
            wait_all_proc_end(proc_list, logfile_path, wait_interval, wall_time)

            if not all(os.path.isfile(f"{v}/{mlip_model_name}") for v in ensemble_dir_path_list):
                write_log("Error: Model does not exist after training, terminated", logfile_path)
                exit()

            if os.path.isfile("salend"):
                write_log("Force terminate", logfile_path)
                exit()
            write_log("Ensemble training completed", logfile_path)
            dt = "%.3f"%(timer.get_dt())
            write_log(f"Ensemble model traing time: {dt} s", logfile_path)
            write_log("", logfile_path)
        else:
            write_log("Ensemble model exists, skip training", logfile_path)

        # Prepare input files for Subascent
        data_filename_for_sample = []
        frame_index_for_sample   = []
        steepness_parameters_index = -1
        if subascent_sample_target_filename != "":
            total_n_frame_target_sample_data = len(read(subascent_sample_target_filename, format = 'extxyz', index = ":"))

        for i_frame in range(subascent_number_of_frames_to_sample):
            if subascent_sample_target_filename == "":

                random_number = np.random.random()
                if itr > 0 and random_number > subascent_sample_initial_data_probability and len(data_filename_list) > 1:
                    if subascent_sample_data_probability_distribution == "uniform":
                        data_index = np.random.randint(1, len(data_filename_list))
                    elif subascent_sample_data_probability_distribution == "exp":
                        selected_fg = 0
                        selected_itr_data_indices = []
                        while selected_fg == 0:
                            delta_itr = int(round(-subascent_sample_data_exp_coeff * np.log(np.random.uniform()), 0))
                            selected_itr = itr - delta_itr
                            if selected_itr == itr: selected_itr -= 1
                            if selected_itr <  0  : selected_itr  = 0
                            for i_data_index, data_filename in enumerate(data_filename_list):
                                if f"itr_{selected_itr}" in data_filename:
                                    selected_itr_data_indices.append(i_data_index)
                            if len(selected_itr_data_indices) > 0:
                                selected_fg = 1
                        data_index = selected_itr_data_indices[np.random.randint(0, len(selected_itr_data_indices))]
                else:
                    data_index = 0

                random_frame_index = np.random.randint(0, total_n_frame_list[data_index])
                data_filename_for_sample.append(data_filename_list[data_index])
            else:

                random_frame_index = np.random.randint(0, total_n_frame_target_sample_data)
                data_filename_for_sample.append(subascent_sample_target_filename)

            frame_index_for_sample.append(random_frame_index)
            atoms = read(data_filename_for_sample[-1], format = 'extxyz', index = frame_index_for_sample[-1])
            data_path = f"{itr_dir_path}/data_for_sample_{i_frame}.extxyz"
            write(data_path, atoms, format = 'extxyz')

            ensemble_shuffle_indices = np.arange(len(ensemble_dir_path_list))
            np.random.shuffle(ensemble_shuffle_indices)
            ensemble_dir_path_list_shuffle = [ensemble_dir_path_list[int(v)] for v in ensemble_shuffle_indices]
            subascent_parameters['model_path_list'] = [f"{v}/{mlip_model_name}" for v in ensemble_dir_path_list_shuffle]

            subascent_parameters['data_path']   = data_path
            output_path                         = f"{itr_dir_path}/sample_{i_frame}"
            subascent_parameters['output_path'] = output_path
            abinitio['data_path']               = f"{output_path}_samples.extxyz"
            
            if subascent_steepness_parameters_sampling_type == "random":
                steepness_parameters_index = np.random.randint(0, len(subascent_steepness_parameters))
            elif subascent_steepness_parameters_sampling_type == "exhaust":
                steepness_parameters_index = (steepness_parameters_index + 1) % len(subascent_steepness_parameters)
                
            subascent_parameters['steepness_parameters'] = [int(v) for v in subascent_steepness_parameters[steepness_parameters_index]]

            with open(f"{itr_dir_path}/sample_{i_frame}.yaml", 'w') as fout:
                yaml.dump(subascent_parameters, fout, sort_keys = False)

            with open(f"{itr_dir_path}/abinitio_{i_frame}.yaml", 'w') as fout:
                yaml.dump(abinitio, fout, sort_keys = False)

        # # for debugging
        # for i_frame in range(subascent_number_of_frames_to_sample): 
        #     write_log(f"{data_filename_for_sample[i_frame]}, frame {frame_index_for_sample[i_frame]}", logfile_path)
            
        # Subascent sampling and ab-initio evaluation
        go_to_path(itr_dir_path)
        proc_list = []
        for i_frame in range(subascent_number_of_frames_to_sample):    
            outfile1 = f"{itr_dir_path}/sample_{i_frame}.out" 
            outfile2 = f"{itr_dir_path}/abinitio_{i_frame}.out"
            cuda_card_idx, cuda_card = next_cuda_card(cuda_card_idx, cuda_card_list)

            proc = run_command(f"export CUDA_VISIBLE_DEVICES={cuda_card} ; sal-sample sample_{i_frame}.yaml > {outfile1} 2>&1 ; sal-abinitio abinitio_{i_frame}.yaml > {outfile2} 2>&1")
            write_log(f"Subascent sampling job {i_frame} using cuda:{cuda_card}: {data_filename_for_sample[i_frame]}, frame {frame_index_for_sample[i_frame]}", logfile_path)
            
            proc_list.append(proc)
        go_to_path(root_dir_path)
        wait_all_proc_end(proc_list, logfile_path, wait_interval, wall_time)

        if os.path.isfile("salend"):
            write_log("Force terminate", logfile_path)
            exit()
        write_log("Subascent sampling completed", logfile_path)
        dt = "%.3f"%(timer.get_dt())
        write_log(f"Subascent sampling and ab-initio evaluation time: {dt} s", logfile_path)
        

