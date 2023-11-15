import numpy as np
from nequip.ase import NequIPCalculator

def set_nequip_calc(atoms, model_path, device = 'cpu'):    
    atomic_numbers = atoms.get_atomic_numbers()
    chemical_symbols = atoms.get_chemical_symbols()
    chem_map = {int(atomic_numbers[i]): chemical_symbols[i] for i in range(len(atomic_numbers))}
    chemical_symbols_list = [chem_map[v] for v in np.unique(atomic_numbers)]
    species_to_type_name = {v : v for v in chemical_symbols_list}
    atoms.calc = NequIPCalculator.from_deployed_model(model_path=model_path, device = device, species_to_type_name = species_to_type_name, set_global_options = True)
