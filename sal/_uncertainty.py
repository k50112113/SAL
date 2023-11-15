
def ensemble_uncertainty(ensemble_energy, ensemble_force):
    # ensemble_energy: torch.tensor (M)
    # ensemble_force: torch.tensor (M x 3N)
    energy_uncertainty = ensemble_energy.std() #for ensembe = 2, abs((ensemble_energy[1]-ensemble_energy[0]))/2**0.5
    force_uncertainty  = ensemble_force.std(dim = 0).max(dim = 0)[0] #abs((ensemble_force[1]-ensemble_force[0]))/2**0.5
    return energy_uncertainty, force_uncertainty