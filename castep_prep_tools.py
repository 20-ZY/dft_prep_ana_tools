import numpy as np
from typing import List, Optional, Tuple, Dict
import os

four_space = '    '

class ReadTpl:
    default_cell_lines = ['%BLOCK LATTICE_CART\n', 
                        '%ENDBLOCK LATTICE_CART\n', '\n', 
                        '%BLOCK POSITIONS_FRAC\n', 
                        '%ENDBLOCK POSITIONS_FRAC\n', '\n', 
                        '%BLOCK SPECIES_MASS\n', 
                        '%ENDBLOCK SPECIES_MASS\n', '\n', 
                        '%BLOCK SPECIES_POT\n', 
                        '%ENDBLOCK SPECIES_POT\n', '\n', 
                        'kpoints_mp_grid 1 1 1\n']
    
    default_param_lines = ['TASK : SinglePoint\n', '\n', 
                           'XC_FUNCTIONAL : PBE\n', '\n', 
                           'CUT_OFF_ENERGY : 300 eV\n', '\n', 
                           'MAX_SCF_CYCLES : 100\n', '\n', 
                           'MIXING_SCHEME : Pulay\n', '\n', 
                           'ELEC_ENERGY_TOL : 1.0e-5\n']

    def __init__(self, seedname: str) -> None:
        # Initialize with the seedname and check if the template files exist.
        self.seedname = seedname
        self.tpl_cell_exist_ok, self.tpl_param_exist_ok = self.tpl_exist_ok()
        self.cell_lines = None
        self.param_lines = None

    def _read_file(self, filename: str) -> List[str]:
        with open(f'{filename}', 'r') as file:
            return file.readlines()

    def get_cell_lines(self) -> List[str]:
        if self.cell_lines is None:
            if self.tpl_cell_exist_ok:
                self.cell_lines =  self._read_file(f'{self.seedname}.cell')
            else:
                self.cell_lines = self.default_cell_lines
        return self.cell_lines
    
    def get_param_lines(self) -> List[str]:
        if self.param_lines is None:
            if self.tpl_param_exist_ok:
                self.param_lines = self._read_file(f'{self.seedname}.param')
            else:
                self.param_lines = self.default_param_lines
        return self.param_lines

    def tpl_exist_ok(self) -> Tuple[bool, bool]:
        tpl_cell_exist_ok = os.path.exists(f'./{self.seedname}.cell')
        tpl_param_exist_ok = os.path.exists(f'./{self.seedname}.param')
        return tpl_cell_exist_ok, tpl_param_exist_ok
    
def write_file(filename: str, cell_lines: List[str], param_lines: List[str]) -> None:
    with open(f'{filename}.cell', 'w') as cell_file, open(f'{filename}.param', 'w') as param_file:
        cell_file.writelines(cell_lines)
        param_file.writelines(param_lines)

class AddKeyword:
    def __init__(self, tpl_lines: List[str], keyword: str, value: str, unit: Optional[str] = None, is_block: bool = False) -> None:
        self.lines = tpl_lines.copy()
        if is_block:
            self.add_block(keyword, value)
        else:
            self.add_keyword(keyword, value, unit)

    def add_keyword(self, keyword: str, value: str, unit: Optional[str] = None) -> List[str]:
        unit_str = f' {unit}' if unit else ''
        self.lines.append(f'\n{keyword} {value}{unit_str}\n' if self.lines[-1].strip() else f'{keyword} {value}{unit_str}\n')
        print(f'This parameter, {keyword}, was not found. It has been added to this file and its values is {value}.')
        return self.lines

    def add_block(self, keyword: str, value: str) -> List[str]:
        start_key, end_key = f'%BLOCK {keyword}', f'%ENDBLOCK {keyword}'
        self.lines.append(f'\n{start_key}\n{value}{end_key}\n' if self.lines[-1].strip() else f'{start_key}\n{value}{end_key}\n')
        print(f'This block, {keyword}, was not found. It has been added to this file.')
        return self.lines

    def get_lines(self) -> List[str]:
        return self.lines

class Modify:
    def __init__(self, filename: str, tpl_lines: List[str], keyword: str, value: str, unit: Optional[str] = None, is_block: bool = False) -> None:
        # Initialize with filename, template lines, and modify the keyword or block
        self.lines = tpl_lines.copy()
        self.filename = filename
        if is_block:
            self.modify_block(keyword, value)
        else:
            self.modify_line(keyword, value, unit)
            self.filename += f'_{value}'
    
    @staticmethod
    def get_keyword_idx(lines: List[str], keyword: str) -> int:
        # Get the index of the line containing the keyword
        for idx, line in enumerate(lines):
            if keyword in line:
                return idx
            
    def modify_line(self, keyword: str, value: str, unit: Optional[str] = None) -> List[str]:
        # Modify a line containing the keyword with the new value
        line_idx = self.get_keyword_idx(self.lines, keyword)
        unit_str = f' {unit}' if unit else ''
        self.lines[line_idx] = f'{keyword} {value}{unit_str}\n'
        return self.lines
    
    def modify_block(self, keyword: str, value: str) -> List[str]:
        # Modify a block containing the keyword with the new value
        start_key, end_key = f'%BLOCK {keyword}', f'%ENDBLOCK {keyword}'
        start_idx = self.get_keyword_idx(self.lines, start_key)
        end_idx = self.get_keyword_idx(self.lines, end_key)
        self.lines[start_idx : end_idx + 1] = [f'{start_key}\n', f'{value}', f'{end_key}\n']
        return self.lines
    
    def get_lines(self) -> List[str]:
        return self.lines
    
    def get_filename(self) -> str:
        return self.filename
    
class LatVecs:
    # Extract the lattice vectors from the given cell file. 
    def __init__(self, tpl_lines) -> None:
        self.lines = tpl_lines.copy()
        if self.is_abc():
            self.lat_vecs = self.conv_lat_vecs_abc()
        else:
            self.lat_vecs = self.ext_lat_vecs() 
        

    def ext_lat_vecs(self) -> np.ndarray:
        lat_vecs = []
        # Extract the lattice vectors from the initial cell file of LATTICE_CART. 
        for i, line in enumerate(self.lines):
            if f'%BLOCK LATTICE_CART' in line:
                lat_vecs = [self.lines[i + 1].split(), self.lines[i + 2].split(), self.lines[i + 3].split()]
                lat_vecs = np.array(lat_vecs, dtype=float)
                lat_vecs = np.round(lat_vecs, decimals=8)
        return lat_vecs
    
    def conv_lat_vecs_abc(self) -> np.ndarray:
        # Compute the lattice vectors from the initial cell file of LATTICE_ABC.
        for i, line in enumerate(self.lines):
            if f'%BLOCK LATTICE_ABC' in line:
                A, B, C = map(float, self.lines[i + 1].split())
                alpha, beta, gamma = map(lambda x: np.radians(float(x)), self.lines[i + 2].split())
                A_x = A
                A_y = 0.
                A_z = 0.
                B_x = B * np.cos(gamma)
                B_y = B * np.sin(gamma)
                B_z = 0.
                C_x = C * np.cos(beta)
                C_y = C * (np.cos(alpha) - (np.sin(gamma) * np.cos(beta)) / np.sin(gamma)) / np.sin(gamma)
                C_z = np.sqrt(C ** 2 - (C * np.cos(beta)) ** 2 - (C * (np.cos(alpha) - (np.sin(gamma) * np.cos(beta)) / np.sin(gamma)) / np.sin(gamma)) ** 2)
        lat_vecs = np.round([[A_x, A_y, A_z], [B_x, B_y, B_z], [C_x, C_y, C_z]], decimals=8)
        return lat_vecs

    def is_abc(self) -> bool:
        # Check if the cell file is in LATTICE_CART format. 
        return any('%BLOCK LATTICE_ABC' in line for line in self.lines)
    
    def get_lat_vecs(self) -> np.ndarray:
        return self.lat_vecs

class InputForEC:
    # A seedname must be provided at the beginning. 
    def __init__(self, seedname: str, tpl_cell_lines: List[str], tpl_param_lines: List[str], cry_sys: str, max_strain: float, steps: int) -> None:
        self.filename = seedname
        self.cell_lines = tpl_cell_lines.copy()
        self.param_lines = tpl_param_lines.copy()
        self.init_lat_vecs = LatVecs(self.cell_lines).get_lat_vecs()
        self.cell_lines = self.check_modify()
        write_file(f'{self.filename}_equil', self.cell_lines, self.param_lines)
        all_def_lat_vecs = self.DeformLat(self.init_lat_vecs, cry_sys, max_strain, steps).get_all_def_lat_vecs()
        for Cij, set_def_lat_vecs in all_def_lat_vecs.items():
            for idx, def_lat_vecs in enumerate(set_def_lat_vecs, start=1):
                new_cell_lines = self.cell_lines.copy()
                for i, line in enumerate(new_cell_lines):
                    if '%BLOCK LATTICE_CART' in line:
                        new_cell_lines[i + 1 : i + 4] = self.format_lat_vecs(def_lat_vecs)
                filename = f'{self.filename}_{Cij}_{idx:02}'
                write_file(f'{filename}', new_cell_lines, self.param_lines)

    def format_lat_vecs(self, lat_vecs: np.ndarray) -> List[str]:
        return [
            f'{four_space}{lat_vecs[0][0]:.8f}{four_space}{lat_vecs[0][1]:.8f}{four_space}{lat_vecs[0][2]:.8f}\n', 
            f'{four_space}{lat_vecs[1][0]:.8f}{four_space}{lat_vecs[1][1]:.8f}{four_space}{lat_vecs[1][2]:.8f}\n', 
            f'{four_space}{lat_vecs[2][0]:.8f}{four_space}{lat_vecs[2][1]:.8f}{four_space}{lat_vecs[2][2]:.8f}\n']
    
    def check_modify(self) -> List[str]:
        param_found = False
        for i, line in enumerate(self.cell_lines):
            if 'FIX_ALL_CELL' in line:
                param_found = True
                print('This parameter, FIX_ALL_CELL, was found in the cell file. Its value has been modified to TRUE.')
                self.cell_lines[i] = 'FIX_ALL_CELL : TRUE\n'
            if '%BLOCK LATTICE_ABC' in line:
                self.cell_lines[i : i + 4] = [f'%BLOCK LATTICE_CART\n'] + self.format_lat_vecs(self.init_lat_vecs) + [f'%ENDBLOCK LATTICE_CART\n']
            if '%BLOCK LATTICE_CART'in line:
                self.cell_lines[i : i + 5] = [f'%BLOCK LATTICE_CART\n'] + self.format_lat_vecs(self.init_lat_vecs) + [f'%ENDBLOCK LATTICE_CART\n']
        if not param_found:
            self.cell_lines.append('\nFIX_ALL_CELL : TRUE' if self.cell_lines[-1].strip() else 'FIX_ALL_CELL : TRUE\n')
            print('This parameter, FIX_ALL_CELL, was not found in this cell file. It has been added to this file and its values is TRUE.')
        return self.cell_lines

    class DeformLat:
        def __init__(self, lat_vecs: np.ndarray, cry_sys: str, max_strain: float, steps: int) -> None:
            self.strain_matrices = self.get_strain_mats(cry_sys)
            self.Cij_order = self.get_Cij_order(cry_sys)
            self.all_strain_mats = self.get_all_strain_mats(max_strain, steps)
            self.all_def_lat_vecs = {}
            for C_ij, strain_mats in self.all_strain_mats.items():
                def_lat_vecs = np.array([self.deformed_lat_vecs(lat_vecs, strain_mat) for strain_mat in strain_mats])
                self.all_def_lat_vecs[C_ij] = np.round(def_lat_vecs, decimals=8)


        def get_strain_mats(self, cry_sys: str) -> List[np.ndarray]:
            # Extract the strain matrices needed according to the crystal system. 
            strain_mat_01 = [[1., 0., 0.], [0., 0., 0.], [0., 0., 0]] # C_11
            strain_mat_02 = [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]] # C_22
            strain_mat_03 = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]] # C_33
            strain_mat_04 = [[0., 0.5, 0.], [0.5, 0., 0.], [0., 0., 0.]] # C_66
            strain_mat_05 = [[0., 0., 0.5], [0., 0., 0.], [0.5, 0., 0.]] # C_55
            strain_mat_06 = [[0., 0., 0.], [0., 0., 0.5], [0., 0.5, 0.]] # C_44
            strain_mat_07 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]] # C_12
            strain_mat_08 = [[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]] # C_13
            strain_mat_09 = [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]] # C_23
            strain_mat_10 = [[1., 0.5, 0.], [0.5, 0., 0.], [0., 0., 0.]] # C_16
            strain_mat_11 = [[1., 0., 0.5], [0., 0., 0.], [0.5, 0., 0.]] # C_15
            strain_mat_12 = [[1., 0., 0.], [0., 0., 0.5], [0., 0.5, 0.]] # C_14
            strain_mat_13 = [[0., 0.5, 0.], [0.5, 1., 0.], [0., 0., 0.]] # C_26
            strain_mat_14 = [[0., 0., 0.5], [0., 1., 0.], [0.5, 0., 0.]] # C_25
            strain_mat_15 = [[0., 0., 0.], [0., 1., 0.5], [0., 0.5, 0.]] # C_24
            strain_mat_16 = [[0., 0.5, 0.], [0.5, 0., 0.], [0., 0., 1.]] # C_36
            strain_mat_17 = [[0., 0., 0.5], [0., 0., 0.], [0.5, 0., 1.]] # C_35
            strain_mat_18 = [[0., 0., 0.], [0., 0., 0.5], [0., 0.5, 1.]] # C_34
            strain_mat_19 = [[0., 0.5, 0.5], [0.5, 0., 0.], [0.5, 0., 0.]] # C_56
            strain_mat_20 = [[0., 0.5, 0.], [0.5, 0., 0.5], [0., 0.5, 0.]] # C_46
            strain_mat_21 = [[0., 0., 0.5], [0., 0., 0.5], [0.5, 0.5, 0.]] # C_45

            strain_matrices = {
                'cubic': [strain_mat_01, strain_mat_07, strain_mat_06],
                'hcp': [strain_mat_01, strain_mat_07, strain_mat_08, strain_mat_03, strain_mat_06],
                'tetr': [strain_mat_01, strain_mat_07, strain_mat_08, strain_mat_03, strain_mat_06, strain_mat_04],
            }
            return strain_matrices.get(cry_sys, [])
        
        def get_Cij_order(self, cry_sys: str) -> List[str]:
            # Extract the sequence of C_ij to be computed using the generated strain matrices. 
            Cij_order = {
                'cubic': ['C11', 'C12', 'C44'],
                'hcp': ['C11', 'C12', 'C13', 'C33', 'C44'], 
                'tetr': ['C11', 'C12', 'C13', 'C33', 'C44', 'C66'],
            }
            return Cij_order.get(cry_sys, []) 
        
        def deformed_lat_vecs(self, lat_vecs: np.ndarray, strain_mat: np.ndarray) -> np.ndarray:
            def_mat = np.eye(3) + strain_mat
            return np.dot(def_mat, lat_vecs.T).T

        def get_all_strain_mats(self, max_strain: float, steps: int) -> Dict[str, List[np.ndarray]]:
            # Compute all the strain matrices and place them into a dictionary according to C_ij. 
            strain_scaling = []
            for i in range(steps):
                strain_scaling.append(float((i + 1) / steps * max_strain))
                strain_scaling.append(float((i + 1) / steps * max_strain * -1.))
            strain_scaling = np.array(sorted(strain_scaling))
            all_strain_mats = {}
            for index, strain_mat in enumerate(self.strain_matrices):
                all_strain_mats[self.Cij_order[index]] = []
                for i in range(len(strain_scaling)):
                    all_strain_mats[self.Cij_order[index]].append(strain_scaling[i] * np.array(strain_mat))
            return all_strain_mats
        
        def get_all_def_lat_vecs(self) -> Dict[str, np.ndarray]:            
            return self.all_def_lat_vecs



'''
The modify_block function will never change the filename; instead, it will add the value as a suffix. It is recommended to edit only one type of input, either cell or param, to ensure the filename remains well-ordered.

seedname = '***'
tpl_cell_lines = ReadTpl(seedname).get_cell_lines()
tpl_param_lines = ReadTpl(seedname).get_param_lines()
e.g.
modify_param_1 = Modify(tpl_parm_lines, keyword, value, unit, is_block)
new_param_lines = modify_param_1.get_lines()
filename = modify_param_1.get_filename()
modify_param_2 = Modify(filename, new_param_lines, keyword, value, unit, is_block)
new_param_lines = modify_param_2.get_lines()
filename = modify_param_2.get_filename()
...
new_cell_lines = ...
write_file(filename, new_cell_lines, new_param_lines)
'''

'''
InputForEC is used to generate a series of input files for calculating elastic constants.

seedname = '***'
tpl_cell_lines = ReadTpl(seedname).get_cell_lines()
tpl_param_lines = ReadTpl(seedname).get_param_lines()
InputForEC(seedname, tpl_cell_lines, tpl_param_lines, cry_sys, max_strain, steps)

This will automatically generate the necessary files.
'''