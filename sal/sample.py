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
from lammps import lammps
from Subascent.Subascent import Subascent
from Subascent.FunctionWrapper import ASEWrapper
from sal._data_control import model_engine_list
from sal._nequip import set_nequip_calc
from sal._uncertainty import ensemble_uncertainty

def main():

    parser = argparse.ArgumentParser(description='Sample new data using Subascent')
    parser.add_argument('filename',  type = str,   help = "*.yaml")
    args = parser.parse_args()

    with open(args.filename) as fin:
        sample_yaml_file = yaml.load(fin, Loader=yaml.FullLoader)
        dtype                    = sample_yaml_file['dtype']
        model_engine             = sample_yaml_file['model_engine'].lower()
        assert model_engine.lower() in model_engine_list, "Error: model_engine not recognized."
        model_path_list          = sample_yaml_file['model_path_list']
        data_path                = sample_yaml_file['data_path']
        output_path              = sample_yaml_file['output_path']

        steepness_parameters     = sample_yaml_file['steepness_parameters']
        fix_atom                 = sample_yaml_file.get('fix_atom', None)
        move_atom                = sample_yaml_file.get('move_atom', None)
        assert fix_atom is None or move_atom is None, "fix_atom and move_atom cannot be defined at the same time."
        max_number_of_samples    = sample_yaml_file.get('max_number_of_samples', 10)
        max_number_of_ascent_descent_cycles = sample_yaml_file.get('max_number_of_ascent_descent_cycles', 1)
        max_number_of_ascent_steps = sample_yaml_file.get('max_number_of_ascent_steps', 300)
        evaluate_uncertainty_frequency = sample_yaml_file.get('evaluate_uncertainty_frequency', 5)
        initial_random_displacement = sample_yaml_file.get('initial_random_displacement', 0.0)

        force_uncertainty  = sample_yaml_file.get('force_uncertainty', 1.0)
        energy_uncertainty = sample_yaml_file.get('energy_uncertainty', float('inf'))
        max_energy         = sample_yaml_file.get('max_energy', float('inf'))
        
        etol    = sample_yaml_file.get('etol',    1.0e-9)
        ftol    = sample_yaml_file.get('ftol',    1.0e-5)
        dtol    = sample_yaml_file.get('dtol',    0.0005)
        rmin    = sample_yaml_file.get('rmin',    0.0)
        epsilon = sample_yaml_file.get('epsilon', 1.0e-6)
        dmax    = sample_yaml_file.get('dmax',    0.03)
        dinit   = sample_yaml_file.get('dinit',   0.01)

        verbose = sample_yaml_file.get('verbose', 0)
    
    assert dtype in ['float64', 'float32'], "Error: dtype not recognized."
    if dtype   == 'float64':
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch_dtype = torch.float64
    elif dtype == 'float32':
        torch.set_default_tensor_type(torch.FloatTensor)
        torch_dtype = torch.float32
        
    def compute_ensemble_uncertainty(forcefield_list, R):
        ensemble_energy = []
        ensemble_force  = []
        for j in range(len(forcefield_list)):
            f, e = forcefield_list[j](R)
            ensemble_energy.append(e)
            ensemble_force.append(f.flatten())
        ensemble_energy = torch.stack(ensemble_energy)
        ensemble_force  = torch.stack(ensemble_force)
        energy_devi, force_devi = ensemble_uncertainty(ensemble_energy, ensemble_force)
        return energy_devi, force_devi, ensemble_energy
        
    samples_atoms = []
    samples = {
        'steepness_parameter': [],
        'energy':              [],
        'energy_uncertainty':  [],
        'force_uncertainty':   [],
    }
    trajectories_atoms = []
    trajectories = {
        'steepness_parameter': [],
        'energy':              [],
        'energy_uncertainty':  [],
        'force_uncertainty':   [],
        'ascent_or_descent':   [],
        # 'eigenvalues':         [],
        # 'eigenvectors':        [],
    }

    data = read(data_path, format = "extxyz", index = ":")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i_frame in range(len(data)):

        atoms_list      = [data[i_frame].copy() for v in model_path_list]

        if model_engine == 'nequip':
            forcefield_list = [ASEWrapper(atoms) for atoms in atoms_list]
            for m in range(len(model_path_list)):
                set_nequip_calc(forcefield_list[m].atoms, model_path_list[m], device)

        R_save    = forcefield_list[0].get_positions().to(torch_dtype)
        cell_save = forcefield_list[0].get_cell_pbc()[0].to(torch_dtype)

        if verbose == 1: print(f"Frame {i_frame}", flush = True)
        for steepness_parameter in steepness_parameters:    
            if verbose == 1: print(f"steepness_parameter {steepness_parameter}", flush = True)

            R                = R_save.clone()
            cell             = cell_save.clone()
            var_fg           = torch.ones_like(R)
            descent_var_fg   = torch.ones_like(R)
            alternate_fg     = False
            if fix_atom is not None and len(fix_atom) > 0:
                fix_atom         = torch.tensor(fix_atom).to(torch.int64)
                var_fg[fix_atom] = 0
                if len(fix_atom) > 1: alternate_fg = True
            elif move_atom is not None and len(move_atom) > 0:
                var_fg            = torch.zeros_like(R)
                move_atom         = torch.tensor(move_atom).to(torch.int64)
                var_fg[move_atom] = 1
                if len(move_atom) < len(var_fg) - 1: alternate_fg = True

            sa = Subascent(etol    = etol,    \
                           ftol    = ftol,    \
                           dmax    = dmax,    \
                           dtol    = dtol,    \
                           dinit   = dinit,   \
                           epsilon = epsilon, \
                           rmin    = rmin,    \
                           emax    = max_energy)

            # sa.initialize(func          = forcefield_list[0], 
            #               x             = R,
            #               s             = steepness_parameter,
            #               variable_flag = var_fg,
            #               pbc_box       = cell,
            #               verbose       = 0)
            # if verbose == 1: print("Descent to local minimum", flush = True)
            # sa.descent_step(steps = max_number_of_ascent_steps)

            if verbose == 1: print("%7s %13s %13s %13s"%("Step", "E", "dE", "dF"), flush = True)
            for i_cycle in range(max_number_of_ascent_descent_cycles):
                for ascent_or_descent in [0, 1]:

                    if verbose == 1: print("Ascent" if ascent_or_descent == 0 else "Descent", flush = True)

                    if i_cycle == 0 and ascent_or_descent == 0: start_x = R
                    else:                                       start_x = sa.get_current_x()
                    
                    start_x_random_displacement = (2.0 * torch.rand_like(start_x) - 1.0) * initial_random_displacement

                    sa.initialize(func          = forcefield_list[0], 
                                  x             = start_x + start_x_random_displacement,
                                  s             = steepness_parameter,
                                  variable_flag = var_fg if ascent_or_descent == 0 else descent_var_fg,
                                  pbc_box       = cell,
                                  verbose       = 1)

                    for i_step in range(0, max_number_of_ascent_steps, evaluate_uncertainty_frequency):
                        if ascent_or_descent == 0:

                            is_end = sa.ascent_step(steps = evaluate_uncertainty_frequency, hvp_skip = True)
                            if alternate_fg:
                                sa.initialize(func          = forcefield_list[0], 
                                              x             = sa.get_current_x(),
                                              s             = steepness_parameter,
                                              variable_flag = 1 - var_fg,
                                              pbc_box       = cell,
                                              verbose       = 0)
                                sa.descent_step(steps = evaluate_uncertainty_frequency)
                                sa.initialize(func          = forcefield_list[0], 
                                              x             = sa.get_current_x(),
                                              s             = steepness_parameter,
                                              variable_flag = var_fg,
                                              pbc_box       = cell,
                                              verbose       = 1)

                        else:

                            is_end = sa.descent_step(steps = evaluate_uncertainty_frequency)
                        
                        current_R = sa.get_current_x()
                        # current_eigenvalues, current_eigenvectors = sa.get_current_eigens()
                        # current_eigenvalues = current_eigenvalues.detach().cpu().numpy()
                        # current_eigenvectors = current_eigenvectors.detach().cpu().numpy()
                        energy_devi, force_devi, ensemble_energy = compute_ensemble_uncertainty(forcefield_list, current_R)
                        if verbose == 1: print("%7d %13.6e %13.6e %13.6e"%(i_step, ensemble_energy[0].item(), energy_devi.item(), force_devi.item()), flush = True)
                        
                        atom_tmp = Atoms(data[i_frame].symbols, positions=current_R.detach().cpu().numpy(), cell = data[i_frame].get_cell(), pbc = data[i_frame].get_pbc())
                        atom_tmp.set_chemical_symbols(data[i_frame].get_chemical_symbols())
                        atom_tmp.set_array("var", var_fg[:, 0].cpu().numpy().astype(np.int))

                        trajectories_atoms.append(atom_tmp)
                        write(f"{output_path}_trajectory.extxyz", atom_tmp, append = True if len(trajectories_atoms) > 1 else False, format = "extxyz")
                        trajectories['steepness_parameter'].append(steepness_parameter)
                        trajectories['energy'].append(ensemble_energy[0].item())
                        trajectories['energy_uncertainty'].append(energy_devi.item())
                        trajectories['force_uncertainty'].append(force_devi.item())
                        trajectories['ascent_or_descent'].append(ascent_or_descent)
                        # trajectories['eigenvalues'].append(current_eigenvalues)
                        # trajectories['eigenvectors'].append(current_eigenvectors)

                        if ensemble_energy[0].item() > max_energy or \
                           force_devi.item()         > force_uncertainty or \
                           energy_devi.item()        > energy_uncertainty:
                            samples_atoms.append(atom_tmp)
                            write(f"{output_path}_samples.extxyz", atom_tmp, append = True if len(samples_atoms) > 1 else False, format = "extxyz")
                            samples['steepness_parameter'].append(steepness_parameter)
                            samples['energy'].append(ensemble_energy[0].item())
                            samples['energy_uncertainty'].append(energy_devi.item())
                            samples['force_uncertainty'].append(force_devi.item())    
                            if len(samples_atoms) >= max_number_of_samples: break
                        
                        if is_end: break
                    if len(samples_atoms) >= max_number_of_samples: break
                if len(samples_atoms) >= max_number_of_samples: break
            if len(samples_atoms) >= max_number_of_samples: break
        if len(samples_atoms) >= max_number_of_samples: break


    for a_key in trajectories.keys():
        trajectories[a_key] = np.array(trajectories[a_key])
    fout = open(f"{output_path}_trajectory.npz", "wb")
    np.savez(fout, **trajectories)
    fout.close()

    if len(samples_atoms) > 0:
        for a_key in samples.keys():
            samples[a_key] = np.array(samples[a_key])
        fout = open(f"{output_path}_samples.npz", "wb")
        np.savez(fout, **samples)
        fout.close()
    else:
        if verbose == 1: print("No new data", flush = True)

    if verbose == 1: print("End of Subascent sampling", flush = True)
