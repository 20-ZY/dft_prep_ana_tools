import numpy as np
from decimal import *
from typing import List, Optional, Dict, Tuple

class GenInitStruct:
    # 创建超胞的初始结构。
    def __init__(self, atom_coords_uc: Dict[str, List[List[float]]], init_CBs: List[List[float]], sc_dir: List[List[int]], size: Tuple[List[float], List[bool]], multiple: List[int] = [10, 10, 10], radius: Optional[float] = 0) -> None:
        self.atom_coords_uc = atom_coords_uc
        self.init_CBs = np.array(init_CBs)
        self.init_CB1, self.init_CB2, self.init_CB3 = self.init_CBs[0], self.init_CBs[1], self.init_CBs[2]
        self.sc_dir = sc_dir
        self.size = size
        self.sc_bases = self.get_sc_bases()
        self.len_ABC = self.get_len_ABC() 
        self.multiple = np.array(multiple)
        self.cped_atom_coords = self.duplicate_atoms()
        self.cped_coords_sc = self.coords_sc()
        self.radius = radius
        self.fin_atom_coords = self.cut_supercell() if self.radius == 0 else self.cut_sphere()

    def get_sc_bases(self) -> np.ndarray[np.ndarray[np.float64]]:
        ABC_vecs = np.dot(self.sc_dir, self.init_CBs.T) # 不是最终超胞的基，而是将超胞的三个晶向表示的基，转化为笛卡尔坐标表示。
        if self.size[1][0] == True:
            sc_basis1 = self.size[0][0] * ABC_vecs[0] / np.linalg.norm(ABC_vecs[0])
        else:
            sc_basis1 = self.size[0][0] * ABC_vecs[0]
        if self.size[1][1] == True:
            sc_basis2 = self.size[0][1] * ABC_vecs[1] / np.linalg.norm(ABC_vecs[1])
        else:
            sc_basis2 = self.size[0][1] * ABC_vecs[1]
        if self.size[1][2] == True:
            sc_basis3 = self.size[0][2] * ABC_vecs[2] / np.linalg.norm(ABC_vecs[2])
        else:
            sc_basis3 = self.size[0][2] * ABC_vecs[2]
        sc_bases = np.array([sc_basis1, sc_basis2, sc_basis3])
        return sc_bases

    def get_len_ABC(self) -> Tuple[np.float64]:
        return np.round(np.linalg.norm(self.sc_bases[0]), decimals=6), np.round(np.linalg.norm(self.sc_bases[1]), decimals=6), np.round(np.linalg.norm(self.sc_bases[2]), decimals=6)
    
    def duplicate_atoms(self) -> Dict[str, List[List[float]]]:
        # Duplicate atoms along the three basis directions. 
        all_cped_coords = {} # {element: [[x1, y1, z1], [x2, y2, z2], ...]} absolute coordinates of atoms in the duplicated cell.
        for elem, coordss in self.atom_coords_uc.items(): # elem: element, coordss: [[x1, y1, z1], [x2, y2, z2], ...]
            all_cped_coords[elem] = [] # Initialize the list of coordinates of the element in the duplicated cell.
            for coords in coordss: # coords: [x, y, z]
                for i in range(-self.multiple[0] + 1, self.multiple[0]): # Direction of basis 1
                    for j in range(-self.multiple[1] + 1, self.multiple[1]): # Directiron of basis 2
                        for k in range(-self.multiple[2] + 1, self.multiple[2]): # Direction of basis 3
                            cped_coords = (coords[0] + i) * self.init_CB1 + (coords[1] + j) * self.init_CB2 + (coords[2] + k) * self.init_CB3
                            all_cped_coords[elem].append(cped_coords)
        return all_cped_coords

    def coords_sc(self) -> Dict[str, List[List[float]]]:
        # Calculate all coordinates with three new bases of the supercell. 
        inverse_supercell_bases = np.linalg.inv(self.sc_bases.T) # Inverse of the transpose of the supercell basis vectors.
        all_coords_sc = {} # {element: [[a1, b1, c1], [a2, b2, c2], ...]} fractional coordinates of atoms w.r.t. the supercell basis.
        for elem, coordss in self.cped_atom_coords.items():
            all_coords_sc[elem] = []
            for original_coords in coordss:
                new_coords = np.dot(inverse_supercell_bases, np.array(original_coords))
                all_coords_sc[elem].append(new_coords)
        for elem, coordss in all_coords_sc.items():
            for coords in coordss:
                for i in range(3):
                    coords[i] = np.round(coords[i], decimals=10)
        return all_coords_sc

    def cut_supercell(self) -> Dict[str, List[List[float]]]:
        coords_in_sc = {}
        for elem, coordss in self.cped_coords_sc.items(): 
            coords_in_sc[elem] = [[a, b, c] for a, b, c in coordss if 0. <= a < 1. and 0. <= b < 1. and 0. <= c < 1.] # Exclude atoms outside the supercell.
        return coords_in_sc

    def cut_sphere(self) -> Dict[str, List[List[float]]]:
        corrds_in_sph = {}
        for elem, coordss in self.cped_coords_sc.items():
            corrds_in_sph[elem] = [[a, b, c] for a, b, c in coordss if a ** 2 + b ** 2 + c ** 2 < self.radius ** 2] # Exclude atoms outside the sphere.
        return corrds_in_sph

    def get_atom_coords(self) -> Dict[str, List[List[float]]]:
        return self.fin_atom_coords

    def save_atom_coords(self, filename: str) -> None:
        coordsss = []
        for elem, coordss in self.fin_atom_coords.items():
            coordsss.extend([[elem, a, b, c] for a, b, c in coordss])
        coordsss.sort(key = lambda coords: (coords[3], coords[1], coords[2]))
        with open(filename, 'w') as f:
            for elem, a, b, c in coordsss:
                f.write(f'    {elem}    {a:.6f}    {b:.6f}    {c:.6f}\n')