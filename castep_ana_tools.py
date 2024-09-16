import numpy as np
from castep_outputs import *
from typing import List, Optional, Tuple, Dict, Any, DefaultDict
import logging
from collections import defaultdict
import os
import matplotlib.pyplot as plt

class ReadCASTEPOutputs:
    '''
    ###### .castep file format ######
    'build_info', 
    'time_started', 
    'pspot_detail', 
        - reference_electronic_structure
        - pseudopotential_definition
        - solver
        - augmentation_charge_rinner
        - partial_core_correction
    'species_properties', 
    'title', 
    'options', 
    'initial_cell', 
    'initial_positions', 
    'k-points', 
    'symmetries', 
    'constraints', 
    'target_stress', 
    'memory_estimate', 
    'energies',
        - 'final_energy'
        - 'free_energy'
        - 'est_0K'
    'forces', 
    'orbital_popn', 
    'mulliken_popn', 
    'bonds', 
    'initialisation_time', 
    'calculation_time', 
    'finalisation_time', 
    'total_time', 
    'peak_memory_use', 
    'parallel_efficiency'
    ###### .geom file format ######
    energy
        - final energy
        - enthalpy
    lattice_vectors
    '''
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def _read_file(self) -> Dict[Any, Any]:
        output = dict(parse_single(self.filename, loglevel=logging.INFO)[-1])
        return output

class GetData:
    def __init__(self, data_dict: Dict, key: str, keykey: str = None) -> None:
        self.data_dict = data_dict
        self.key = key
        self.keykey = keykey

    def get_data(self) -> Any:
        if self.keykey is None:
            return self.data_dict[self.key]
        else:
            return self.data_dict[self.key][self.keykey]
        
# MultiXCConvTest(seedname, XCs, cut_offs).ConvFig(): default energy is 'free_energy'. 
class MultiXCConvTest:
    def __init__(self,  dir: str, seedname: str, XCs: str, cutoffs: np.array, whichenergy: str = 'free_energy') -> None:
        self.seedname = seedname
        self.XCs = XCs
        self.cutoffs = cutoffs
        self.dir = dir
        self.whichenergy = whichenergy
    
    def _read_energies(self) -> Dict[str, Dict[str, float]]:
        energies = {}
        for XC in self.XCs:
            energies[XC] = {}
            for cutoff in self.cutoffs:
                energies[XC][cutoff] = None
                output = ReadCASTEPOutputs(f'{self.dir}/{XC}/{self.seedname}_{XC}_{cutoff}.castep')._read_file()
                energies[XC][cutoff] = GetData(output, 'energies', self.whichenergy).get_data()
        return energies
    
    def ConvFig(self) -> None:
        energies = self._read_energies()
        num_subplots = len(energies)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 4*num_subplots))
        if num_subplots == 1:
            axes = [axes]

        for idx, (XC, data) in enumerate(energies.items()):
            x = np.array(list(data.keys()))
            y = np.array(list(data.values())).flatten()
            axes[idx].plot(x, y, label=XC)
            axes[idx].set_xlabel('Cutoff Energy (eV)')
            axes[idx].set_ylabel('Energy (eV)')
            axes[idx].legend()
            # axes[idx].set_title(f'{self.whichenergy} vs Cutoff Energy for {XC}')
        plt.savefig(f'{self.seedname}.png')

class GetGeomOpt:
    H2eV = 27.211386246
    Bohr2Ang = 0.529177211

    def __init__(self, seedname: str, XCs: Optional[List] = [None]) -> None:
        self.seedname = seedname
        self.XCs = XCs
        self.geom_opt_info = self._read_geom_files()

    def _read_geom_files(self) -> Dict[str, Dict[str, float]]:
        geom_opt_info = {}
        for XC in self.XCs:
            geom_opt_info[XC] = {}
            if XC is None:
                output = ReadCASTEPOutputs(f'{self.seedname}.geom')._read_file()
            else:
                output = ReadCASTEPOutputs(f'{self.seedname}_{XC}.geom')._read_file()
            geom_opt_info[XC]['energy'] = np.array(GetData(output, 'energy').get_data())
            geom_opt_info[XC]['lattice_vectors'] = np.array(GetData(output, 'lattice_vectors').get_data())
        return geom_opt_info
    
    def get_final_energy(self) -> float:
        final_energy = {}
        for XC, data in self.geom_opt_info.items():
            final_energy[XC] = np.round(data['energy'][0][0] * self.H2eV, decimals=8)
        return final_energy

    def get_enthalpy(self) -> float:
        enthalpy = {}
        for XC, data in self.geom_opt_info.items():
            enthalpy[XC] = np.round(data['energy'][0][1] * self.H2eV, decimals=8)
        return enthalpy
    
    def get_lat_vecs(self) -> np.array:
        lattice_vectors = {}
        for XC, data in self.geom_opt_info.items():
            lattice_vectors[XC] = np.round(data['lattice_vectors'] * self.Bohr2Ang, decimals=8)
        return lattice_vectors