import numpy as np
import torch
import argparse
import yaml
import sys
import os
import warnings
warnings.filterwarnings("ignore")
from ase.io import read, write
from ase import Atoms
import importlib
# from ase.calculators.vasp import Vasp
# from ase.calculators.gaussian import Gaussian
# from ase.calculators.espresso import Espresso

from sal._process_control import run_command
from sal._data_control import abinitio_engine_list

def main():

    parser = argparse.ArgumentParser(description='Subascent active learning')
    parser.add_argument('filename',  type = str,   help = "*.yaml")
    args = parser.parse_args()

    with open(args.filename) as fin:
        abinitio_yaml_file = yaml.load(fin, Loader=yaml.FullLoader)
        abinitio_engine    = abinitio_yaml_file['abinitio_engine'].lower()
        data_path          = abinitio_yaml_file['data_path']

        assert abinitio_engine in abinitio_engine_list, "Error: abinitio_engine not recognized."

        if abinitio_engine == 'ase':
            ase_calc         = abinitio_yaml_file['ase_calc']['name']
            
            try:
                ase_calc_lower = ase_calc.lower()
                ase_calc_module = importlib.import_module(f'ase.calculators.{ase_calc_lower}')
                ase_calc_obj = getattr(ase_calc_module, ase_calc)
            except:
                print("Error: ase_calc name not recognized.")
                exit()

            ase_calc_save_output = abinitio_yaml_file['ase_calc']['save_output']
            ase_calc_delete_output = abinitio_yaml_file['ase_calc']['delete_output']
            ase_calc_kwargs        = abinitio_yaml_file['ase_calc']['kwargs']
    
    abinitio_save_output_path = "abinitio_save_output"
    run_command(f"mkdir -p {abinitio_save_output_path}", wait = True)
    
    data_filename = data_path if "/" not in data_path else data_path[data_path.rindex("/") + 1:]
    data_path_extension = data_path[data_path.rindex('.'):]

    data_filename = data_filename.replace(data_path_extension, "")
    calc_tmp_dir = data_path.replace(data_path_extension, "")
    calc_tmp_dir = f"{calc_tmp_dir}_tmp"
    ase_calc_kwargs['directory'] = calc_tmp_dir

    print(f"Now computing {data_path}...", flush = True)
    if os.path.isfile(data_path) == False:
        print("File not found", flush = True)
        print("ab-initio evaluation completed", flush = True)
        exit()

    data = read(data_path, format = "extxyz", index = ":")
    for i, atoms in enumerate(data):
        atoms.calc = ase_calc_obj(**ase_calc_kwargs)
        atoms.get_potential_energy()
        write(f"{abinitio_save_output_path}/{data_filename}_labeled.extxyz", atoms, append = True if i > 0 else False, format = "extxyz")
        for a_ase_calc_save_output in ase_calc_save_output:
            print(f"{calc_tmp_dir}/{a_ase_calc_save_output} {abinitio_save_output_path}/{data_filename}_{i}_{a_ase_calc_save_output}", flush = True)
            run_command(f"cp {calc_tmp_dir}/{a_ase_calc_save_output} {abinitio_save_output_path}/{data_filename}_{i}_{a_ase_calc_save_output}", wait = True)
        for a_ase_calc_delete_output in ase_calc_delete_output:
            run_command(f"rm -f {calc_tmp_dir}/{a_ase_calc_delete_output}", wait = True)
                            
    run_command(f"rm -rf {calc_tmp_dir}", wait = True)
    print("ab-initio evaluation completed", flush = True)

# export CUDA_VISIBLE_DEVICES=0 ; nohup sal-abinitio abinitio_0.yaml > abinitio_0.out
# export CUDA_VISIBLE_DEVICES=1 ; nohup sal-abinitio abinitio_1.yaml > abinitio_1.out &
# export CUDA_VISIBLE_DEVICES=0 ; nohup sal-abinitio abinitio_4.yaml > abinitio_4.out &
# export CUDA_VISIBLE_DEVICES=1 ; nohup sal-abinitio abinitio_5.yaml > abinitio_5.out &