import numpy as np
import argparse
import yaml
import sys
import os
from ase.io import read, write
from ase import Atoms

def get_hist(quantity, max_val, min_val, bin_width):
    bins = int((max_val - min_val) / bin_width)
    hist, bin_edges = np.histogram(quantity, bins=bins)
    bin_central = np.array([(bin_edges[i] + bin_edges[i + 1])/2 for i in range(len(bin_edges) - 1)])
    return hist, bin_central

def get_EF(filename_list):
    E = []
    F = []
    for filename in filename_list:
        print(f"reading {filename}...")
        data = read(filename, format = "extxyz", index = ":")
        for atoms in data:
            E.append(atoms.get_potential_energy()/atoms.get_number_of_atoms())
            forces = atoms.get_forces()
            try:             var_index = (atoms.get_array('var') == 1).nonzero()[0].astype(np.int32)
            except KeyError: var_index = None
            if var_index is not None: forces = forces[var_index]
            F.append(np.absolute(forces).mean())
    return E, F

def main():

    parser = argparse.ArgumentParser(description='Collective historgram for energy and force')
    parser.add_argument('filename',  type = str,   help = "*.yaml")
    args = parser.parse_args()

    with open(args.filename) as fin:
        sal_yaml_file = yaml.load(fin, Loader=yaml.FullLoader)
        initial_data_path = sal_yaml_file['initial_data_path']

    E_data, F_data = {}, {}
    E_data['Initial data'], F_data['Initial data'] = get_EF([initial_data_path])
    E_data['Subascent'], F_data['Subascent'] = [], []
    itr = 0
    while os.path.isdir(f"itr_{itr}"):
        if os.path.isdir(f"itr_{itr}/abinitio_save_output"):
            filename_list_tmp = os.popen(f"find itr_{itr}/abinitio_save_output -name \"sample*samples_labeled.extxyz\"").read().split("\n")
            filename_list_tmp.pop()
            if len(filename_list_tmp) > 0:
                E, F = get_EF(filename_list_tmp)
                E_data['Subascent'] += E
                F_data['Subascent'] += F
        itr += 1
        if itr == 51: break
    max_itr = itr - 1

    for key in E_data.keys(): E_data[key] = np.array(E_data[key])
    for key in F_data.keys(): F_data[key] = np.array(F_data[key])

    emin  = min(E_data['Initial data'].min(), E_data['Subascent'].min())
    emax  = max(E_data['Initial data'].max(), E_data['Subascent'].max())
    fmin  = min(F_data['Initial data'].min(), F_data['Subascent'].min())
    fmax  = max(F_data['Initial data'].max(), F_data['Subascent'].max())
    ehist = []
    ebin  = []
    fhist = []
    fbin  = []

    fout = open("Histogram.out", "w")
    
    for key in E_data.keys():
        h, b = get_hist(E_data[key], emax, emin, 0.005)
        ehist.append(h)
        ebin.append(b)
        fout.write(f"energy: {key}\n")
        fout.write(f"number of bins: {len(b)}\n")
        for i in range(len(b)): fout.write(f"{b[i]} {h[i]}\n")
    for key in F_data.keys():
        h, b = get_hist(F_data[key], fmax, fmin, 0.005)
        fhist.append(h)
        fbin.append(b)
        fout.write(f"force: {key}\n")
        fout.write(f"number of bins: {len(b)}\n")
        for i in range(len(b)): fout.write(f"{b[i]} {h[i]}\n")
    fout.close()
    
    try:
        from ZLabPlot import ZLabPlot
    except:
        print("ZLabPlot module not found")
        exit()

    zp = ZLabPlot({'legend.fontsize': 12})
    zp.add_subplot()
    zp.add_data(ebin, ehist, ms = 'o', legend = list(E_data.keys()), xlabel = r"$E$ (eV/atom)", ylabel = r"$Count$ (\#)", ylog = True)
    zp.save(f"E_{max_itr}.png", transparent = False)
    zp.clear()
    zp.add_subplot()
    zp.add_data(fbin, fhist, ms = 'o', legend = list(F_data.keys()), xlabel = r"$|\bar{F}|$ (eV/\AA)", ylabel = r"$Count$ (\#)", ylog = True)
    zp.save(f"F_{max_itr}.png", transparent = False)
    zp.clear()
    zp.add_subplot()
    zp.add_data([E_data['Initial data'], E_data['Subascent']], [F_data['Initial data'], F_data['Subascent']], lw = 0, ms = 'o', legend = list(E_data.keys()), xlabel = r"$E$ (eV/atom)", ylabel = r"$|\bar{F}|$ (eV/\AA)")
    zp.save(f"EF_{max_itr}.png", transparent = False)
    zp.clear()