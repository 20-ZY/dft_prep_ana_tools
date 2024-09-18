import numpy as np
from castep_outputs import *
from typing import List, Optional, Tuple, Dict, Any, DefaultDict
import logging
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import castep_prep_tools as pt

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
    atom_info('elem', int)
        - position
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

    def __init__(self, seedname: str, XCs: Optional[List] = None) -> None:
        self.seedname = seedname
        self.XCs = XCs if XCs is not None else ['default']
        self.all_atom_info = self.get_atom_info()
        self.geom_opt_info = self._read_geom_files()
        self.lat_vecs = self.get_lat_vecs()

    def get_atom_info(self) -> Dict[str, List[Tuple[str, int]]]:
        all_atom_info = {}
        for XC in self.XCs:
            if XC == 'default':
                cell_filename = f'{self.seedname}.cell'
            else:
                cell_filename = f'{self.seedname}_{XC}.cell'
            with open(cell_filename, 'r') as cell_file:
                lines = cell_file.readlines()
                for idx, line in enumerate(lines):
                    if '%BLOCK POSITIONS_FRAC' in line:
                        start_idx = idx
                    if '%ENDBLOCK POSITIONS_FRAC' in line:
                        end_idx = idx
                atoms = [(line.split()[0]) for line in lines[start_idx + 1 : end_idx]]
                element_count = {}
                atom_info = []
                for element in atoms:
                    if element not in element_count:
                        element_count[element] = 1
                    else:
                        element_count[element] += 1
                    atom_info.append((element, element_count[element]))
            all_atom_info[XC] = atom_info
        return all_atom_info

    def _read_geom_files(self) -> Dict[str, Dict[Any, np.array]]:
        geom_opt_info = {}
        for XC in self.XCs:
            geom_opt_info[XC] = {}
            if XC == 'default':
                output = ReadCASTEPOutputs(f'{self.seedname}.geom')._read_file()
            else:
                output = ReadCASTEPOutputs(f'{self.seedname}_{XC}.geom')._read_file()
            geom_opt_info[XC]['energy'] = np.array(GetData(output, 'energy').get_data())
            geom_opt_info[XC]['lattice_vectors'] = np.array(GetData(output, 'lattice_vectors').get_data())
            for atom_info in self.all_atom_info[XC]:
                geom_opt_info[XC][atom_info] = np.array(GetData(output, atom_info, 'position').get_data()) * self.Bohr2Ang
        return geom_opt_info
    
    def get_final_energy(self) -> Dict[str, float]:
        final_energy = {}
        for XC, data in self.geom_opt_info.items():
            final_energy[XC] = np.round(data['energy'][0][0] * self.H2eV, decimals=8)
        return final_energy

    def get_enthalpy(self) -> Dict[str, float]:
        enthalpy = {}
        for XC, data in self.geom_opt_info.items():
            enthalpy[XC] = np.round(data['energy'][0][1] * self.H2eV, decimals=8)
        return enthalpy
    
    def get_lat_vecs(self) -> Dict[str, np.array]:
        lattice_vectors = {}
        for XC, data in self.geom_opt_info.items():
            lattice_vectors[XC] = np.round(data['lattice_vectors'] * self.Bohr2Ang, decimals=6)
        return lattice_vectors
    
    def get_coords(self) -> Dict[str, np.array]:
        frac_coords = {}
        for XC, data in self.geom_opt_info.items():
            frac_coords[XC] = {}
            bases = np.array(self.lat_vecs[XC]).T
            for atom in self.all_atom_info[XC]:
                cart_coords = data[atom]
                frac_coords[XC][atom] = np.round(np.linalg.solve(bases, cart_coords), decimals=6)
        return frac_coords
    
    def get_frac_block(self) -> Dict[str, str]:
        frac_coords = self.get_coords()
        frac_block = {}
        for XC, data in frac_coords.items():
            frac_block[XC] = ''
            for atom in self.all_atom_info[XC]:
                frac_block[XC] += f'    {atom[0]}    {data[atom][0]:.6f}    {data[atom][1]:.6f}    {data[atom][2]:.6f}\n'
        return frac_block

    def get_lat_block(self) -> Dict[str, str]:
        lat_block = {}
        for XC, data in self.lat_vecs.items():
            lat_block[XC] = ''
            for vec in data:
                lat_block[XC] += f'    {vec[0]:.6f}    {vec[1]:.6f}    {vec[2]:.6f}\n'
        return lat_block
    
class GetECs:
    Cij_order = {
                'cubic': ['C11', 'C12', 'C44'],
                'hcp': ['C11', 'C12', 'C13', 'C33', 'C44'], 
                'tetr': ['C11', 'C12', 'C13', 'C33', 'C44', 'C66'],
            }
    H2eV = 27.211386246
    eV_A32GPa = 160.21766208
    
    def __init__(self, seedname_prefix: str, cry_sys: str, max_strain: float, steps: int, XCs: Optional[List] = None) -> None:
        self.seednames = {}
        self.cry_sys = cry_sys
        self.max_strain = max_strain
        self.steps = steps
        self.XCs = XCs if XCs is not None else ['default']
        if self.XCs == ['default']:
            self.seednames['default'] = {}
            self.seednames['default']['equil'] = f'./{seedname_prefix}_equil'
            for C_ij in self.Cij_order[self.cry_sys]:
                self.seednames['default'][f'{C_ij}'] = []
                for idx in range(1, self.steps * 2 + 1):
                    self.seednames['default'][f'{C_ij}'].append(f'./{seedname_prefix}_{C_ij}_{idx:02}')
        else:
            for XC in self.XCs:
                self.seednames[XC] = {}
                self.seednames[XC]['equil'] = f'{XC}/{seedname_prefix}_{XC}_equil'
                for C_ij in self.Cij_order[self.cry_sys]:
                    self.seednames[XC][f'{C_ij}'] = []
                    for idx in range(1, self.steps * 2 + 1):
                        self.seednames[XC][f'{C_ij}'].append(f'{XC}/{seedname_prefix}_{XC}_{C_ij}_{idx:02}')
        self.volumes = self.get_volume()
        self.all_energies = self.get_energy()
        self.all_strains = self.get_strain()
        self.all_itas = self.get_itas()

    def get_volume(self) -> Dict[str, float]:
        volumes = {}
        for XC in self.seednames.keys():
            volumes[XC] = []
            equil_cell = pt.ReadTpl(f'{self.seednames[XC]['equil']}').get_cell_lines()
            lat_vecs = np.array(pt.LatVecs(equil_cell).get_lat_vecs())
            lat_vec1, lat_vec2, lat_vec3 = lat_vecs
            volumes[XC] = np.dot(lat_vec1, np.cross(lat_vec2, lat_vec3))
        return volumes
    
    def get_strain(self) -> Dict[str, np.array]:
        strain = {}
        for XC, data in self.all_energies.items():
            strain[XC] = {}
            for C_ij in data.keys():
                strain[XC][C_ij] = []
                for i in range(self.steps):
                    strain[XC][C_ij].append(float((i + 1) / self.steps * self.max_strain))
                    strain[XC][C_ij].append(float((i + 1) / self.steps * self.max_strain * -1))
                strain[XC][C_ij] = np.sort(np.array(strain[XC][C_ij]))
                strain[XC][C_ij] = np.insert(strain[XC][C_ij], self.steps, 0.)
        return strain
    
    def get_energy(self) -> Dict[str, Dict[str, np.array]]:
        energies = {}
        for XC, data in self.seednames.items():
            energies[XC] = {}
            equil_output = ReadCASTEPOutputs(f'{data["equil"]}.geom')._read_file()
            equil_energy = equil_output['energy'][0][0] * self.H2eV
            for C_ij in self.Cij_order[self.cry_sys]:
                energies[XC][C_ij] = []
                for seed in data[C_ij]:
                    output = ReadCASTEPOutputs(f'{seed}.geom')._read_file()
                    energies[XC][C_ij].append(output['energy'][0][0] * self.H2eV)
                energies[XC][C_ij] = np.array(energies[XC][C_ij])
                energies[XC][C_ij] = np.insert(energies[XC][C_ij], self.steps, equil_energy)
                energies[XC][C_ij] = ((energies[XC][C_ij] - equil_energy) / self.volumes[XC]) * self.eV_A32GPa
        return energies
    
    def get_itas(self) -> Dict[str, Dict[str, float]]:
        itas = {}
        for XC, data in self.all_energies.items():
            itas[XC] = {}
            for C_ij, energy in data.items():
                itas[XC][C_ij] = np.polyfit(self.all_strains[XC][C_ij], energy, 2)[0] * 2
        return itas

    def get_Cij(self) -> Dict[str, float]:
        Cij = {}
        for XC, data in self.all_itas.items():
            Cij[XC] = {}
            if self.cry_sys == 'cubic':
                Cij[XC]['C11'] = data['C11']
                Cij[XC]['C12'] = data['C12'] / 2 - data['C11']
                Cij[XC]['C44'] = data['C44']
            elif self.cry_sys == 'hcp':
                Cij[XC]['C11'] = data['C11']
                Cij[XC]['C12'] = data['C12'] / 2 - data['C11']
                Cij[XC]['C13'] = (data['C13'] - data['C11'] - data['C33']) / 2
                Cij[XC]['C33'] = data['C33']
                Cij[XC]['C44'] = data['C44']
            elif self.cry_sys == 'tetr':
                Cij[XC]['C11'] = data['C11']
                Cij[XC]['C12'] = data['C12'] / 2 - data['C11']
                Cij[XC]['C13'] = (data['C13'] - data['C11'] - data['C33']) / 2
                Cij[XC]['C33'] = data['C33']
                Cij[XC]['C44'] = data['C44']
                Cij[XC]['C66'] = data['C66']
            # ... add more crystal systems
        return Cij