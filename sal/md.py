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
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from sal._data_control import model_engine_list
from sal._nequip import set_nequip_calc
from sal._uncertainty import ensemble_uncertainty
from sal._process_control import Clock

def main():

    parser = argparse.ArgumentParser(description='Perform regular MD using ASE and select uncertain frames')
    parser.add_argument('filename',  type = str,   help = "*.yaml")
    args = parser.parse_args()

    with open(args.filename) as fin:
        md_yaml_file = yaml.load(fin, Loader=yaml.FullLoader)
        dtype                    = md_yaml_file['dtype']
        model_engine             = md_yaml_file['model_engine'].lower()
        assert model_engine.lower() in model_engine_list, "Error: model_engine not recognized."
        model_path_list          = md_yaml_file['model_path_list']
        data_path                = md_yaml_file['data_path']
        frame_index              = md_yaml_file.get('frame_index', -1)
        output_path              = md_yaml_file['output_path']

        T                        = md_yaml_file['temperature']
        dt                       = md_yaml_file['dt']
        thermo                   = md_yaml_file['thermo']
        eq_step                  = md_yaml_file['equilibration_step']
        prod_step                = md_yaml_file['production_step']

        force_uncertainty        = md_yaml_file.get('force_uncertainty', 1.0)
        energy_uncertainty       = md_yaml_file.get('energy_uncertainty', float('inf'))
        max_energy               = md_yaml_file.get('max_energy', float('inf'))

        minimum_frames_to_select = md_yaml_file.get('minimum_frames_to_select', 10)
    
    assert dtype in ['float64', 'float32'], "Error: dtype not recognized."
    if dtype   == 'float64':
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch_dtype = torch.float64
    elif dtype == 'float32':
        torch.set_default_tensor_type(torch.FloatTensor)
        torch_dtype = torch.float32
    
    def compute_ensemble_uncertainty(atoms_list):
        ensemble_energy = [atoms_list[0].get_potential_energy()]
        ensemble_force  = [atoms_list[0].get_forces().flatten()]
        for j in range(1, len(atoms_list)):
            atoms_list[j].set_positions(atoms_list[0].get_positions())
            ensemble_energy.append(atoms_list[j].get_potential_energy())
            ensemble_force.append(atoms_list[j].get_forces().flatten())
        ensemble_energy = torch.tensor(np.stack(ensemble_energy))
        ensemble_force  = torch.tensor(np.stack(ensemble_force))
        energy_devi, force_devi = ensemble_uncertainty(ensemble_energy, ensemble_force)
        return energy_devi.detach().cpu().numpy(), force_devi.detach().cpu().numpy()

    def printenergy(i, atoms_list, fthermo, timer):
        epot = atoms_list[0].get_potential_energy()
        ekin = atoms_list[0].get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / atoms_list[0].get_number_of_atoms()
        energy_devi, force_devi = compute_ensemble_uncertainty(atoms_list)
        walltime = timer.get_dt()
        print_str = "%d %.3e %.3e %.3e %.3e %.3e %.3f"%(i, temp, epot, ekin, energy_devi, force_devi, walltime)
        print(print_str, flush = True)
        fthermo.write(print_str + '\n')
        fthermo.flush()

    def run_md(atoms_list, dyn, max_step, thermo, thermo_filename, traj_filename, timer):
        if max_step == 0: return
        print_str = "#step temp epot ekin sigma_e sigma_f walltime"
        print(print_str, flush = True)
        fthermo = open(thermo_filename, "w")
        fthermo.write(print_str + '\n')
        for i in range(0, max_step, thermo):
            dyn.run(thermo)
            printenergy(i, atoms_list, fthermo, timer)
            if traj_filename: write(traj_filename, atoms_list[0], format = 'extxyz', append = True if i > 0 else False)
        fthermo.close()

    atoms = read(data_path, format = 'extxyz', index = frame_index)

    if model_engine == 'nequip':
        atoms_list      = [atoms]
        atoms_list     += [atoms.copy() for _ in range(len(model_path_list) - 1)]
        for m in range(len(model_path_list)):
            set_nequip_calc(atoms_list[m], model_path_list[m], 'cuda')
            
    timer = Clock()
    MaxwellBoltzmannDistribution(atoms, temperature_K = T)
    dyn = Langevin(atoms_list[0], dt * units.fs, T * units.kB, 0.002)
    # dyn = VelocityVerlet(atoms, dt * units.fs)
    run_md(atoms_list, dyn, eq_step,   thermo, f"{output_path}-eq.out", None, timer)
    prod_out_filename = f"{output_path}-prod.out"
    prod_traj_filename = f"{output_path}-traj-prod.extxyz"
    run_md(atoms_list, dyn, prod_step, thermo, prod_out_filename, prod_traj_filename, timer)

    data_prod = np.loadtxt(prod_out_filename, delimiter=" ")
    de = data_prod[:, -3]
    df = data_prod[:, -2]
    pe = data_prod[:, 2]

    select_indices = np.where(df >= force_uncertainty)[0]
    select_indices = np.union1d(select_indices, np.where(de >= energy_uncertainty)[0])
    select_indices = np.intersect1d(select_indices, np.where(pe <= max_energy)[0])
    
    if minimum_frames_to_select > len(select_indices):
        df_sort_indices = np.argsort(df)
        select_indices = df_sort_indices[-minimum_frames_to_select:]
        
    print(f"Number of frames selected =", len(select_indices))

    select_atoms_list = read(prod_traj_filename, format = "extxyz", index = ":")
    select_atoms_list = [select_atoms_list[v] for v in select_indices]
    select_traj_filename = f"{output_path}-select.extxyz"
    write(select_traj_filename, select_atoms_list, format = 'extxyz', append = False)

# atoms = read(".extxyz", format = 'extxyz', index = 0)
# write(ini_data_path, atoms, format = 'lammps-data')
# chemical_symbols = atoms.get_chemical_symbols()
# atomic_numbers   = atoms.get_atomic_numbers()
# masses           = atoms.get_masses()
# chemical_symbols_list, chemical_symbols_idx = np.unique(chemical_symbols, return_index = True)
# chemical_symbols_lmp = " ".join(chemical_symbols_list)
# masses_list = [masses[v] for v in chemical_symbols_idx.astype(np.int64)]
# forcefield = LAMMPSWrapper()
# forcefield.lmp.command( "units        metal")
# forcefield.lmp.command( "atom_style   atomic")
# forcefield.lmp.command( "newton       off")
# forcefield.lmp.command( "neighbor     1.0 bin")
# forcefield.lmp.command( "neigh_modify delay 5 every 1")
# forcefield.lmp.command(f"read_data  {ini_data_path}")
# forcefield.lmp.command( "pair_style nequip")
# forcefield.lmp.command(f"pair_coeff * * {model_path} {chemical_symbols_lmp}")
# forcefield.lmp.command( "thermo 1")
# for i_mass, a_mass in enumerate(masses_list):
#     forcefield.lmp.command(f"mass {i_mass + 1} {a_mass}")
# forcefield.lmp.command("minimize 1e-8 1e-8 5 1000")
