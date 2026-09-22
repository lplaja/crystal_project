import numpy as np
from dataclasses import dataclass

import os
import string
from grid import Grid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _compute_reciprocal_coordinates(vectors_coordinates, inverse=False):
    # computes the reciprocal vectors from the direct vectors or the reverse
    # reciprocal vectors are ginen in units of 2pi
    # direct vectors are given in units of a
    A = np.stack(vectors_coordinates).T  # d x d matrix
    # print(f'vprint in line: 9 --> {A.shape=}')
    # print(f'vprint in line: 10 --> {inverse=}')
    if inverse:
        B =  np.linalg.inv(A).T
    else:
        B =  np.linalg.inv(A.T)
    return list(B.T)

def vector_in_cart(vector, basis):
        return  vector @ basis


species = {
    'Simple 1D Chain': {
        'name': 'Simple 1D Chain',
        'type': 'linear',
        'direct vectors': [np.array([1.0])],
        'direct lattice unit': 1.0,
        'reciprocal vectors': None,
        'reciprocal symmetry points': {
            'Gamma': np.array([0]),
            'M': np.array([0.5])
        },
        'orbitals': [
            {
                'name': 'A',
                'position': np.array([0.0]),
                'element': 'A',
                'orbital': 's',
                'energy': 0, 
                'neighbors': [
                    {'name': 'A', 'delta': np.array([1.0]), 'hopping': -1.0},
                    {'name': 'A', 'delta': np.array([-1.0]), 'hopping': -1.0}
                ]
            }
        ]
    },

    'graphene': {
        'name': 'graphene',
        'type': 'hexagonal',
        'direct vectors':[np.array([np.sqrt(3)/2, 0.5]), np.array([np.sqrt(3)/2, -0.5])], # direct vectors  [a_1, a_2, a_3] coordinates in cartesian 
        'direct lattice unit': 2.46e-10,
        'reciprocal vectors': None,   # reciprocal vectors [b_1, b_2, b_3] coordinates in cartesian 
        'reciprocal symmetry points': {  # symmetry points in the reciprocal lattice expressed in the reciprocal basis {b_1, b_2, b_3}
            'Gamma': np.array([0, 0]),
            'K': np.array([2/3, +1/3]),
            'Kp': np.array([1/3, 2/3]),
            'Kpv': np.array([1/3, -1/3]),
            'M': np.array([1/2, 1/2]),
            'Gamma01': np.array([1, 1]),
            'Gamma10': np.array([1, 0]),
        },
        'orbitals': [
            {
                'name': 'A',
                'position': np.array([1/6, 1/6]),  #orbital position in the direct vector basis {a_1, a_2, a_3}
                #'position': np.array([0, 0]),  #orbital position in the direct vector basis {a_1, a_2, a_3}
                'element': 'C',
                'orbital': 'pz',
                'energy': -0.28*0, 
                'neighbors': [
                    {'name': 'B', 'delta': np.array([-1/3, -1/3]),  'hopping': 2.97},  # delta is the relative position of the neighbor in the direct vector basis
                    {'name': 'B', 'delta': np.array([ 2/3, -1/3]), 'hopping': 2.97},
                    {'name': 'B', 'delta': np.array([-1/3,  2/3]), 'hopping': 2.97},
                ]
            },
            {
                'name': 'B',
                'position': np.array([-1/6, -1/6]),  #orbital position in the direct vector basis {a_1, a_2, a_3}
                #'position': np.array([-1/3, -1/3]),  #orbital position in the direct vector basis {a_1, a_2, a_3}
                'element': 'C',
                'orbital': 'pz',
                'energy': -0.28*0, 
                'neighbors': [
                    {'name': 'A', 'delta': np.array([1/3, 1/3]), 'hopping': 2.97},  # delta is the relative position of the neighbor in the direct vector basis
                    {'name': 'A', 'delta': np.array([-2/3, 1/3]),  'hopping': 2.97},
                    {'name': 'A', 'delta': np.array([1/3, -2/3]), 'hopping': 2.97},
                ]
            },
        ]
    },
    'aluminium': {
        'name': 'aluminium',
        'type': 'FCC',
        'direct vectors': [np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.0, 0.5])],
        'direct lattice unit': 4.0495,
        'reciprocal vectors': None,
        'reciprocal symmetry points': {
            'Gamma': np.array([0, 0, 0]),
            'X': np.array([0.5, 0, 0]),
            'K': np.array([3/8, 3/4, 3/8]),
            'L': np.array([0.5, 0.5, 0.5])
        },
        'orbitals': [
            {
                'name': 'Al1',
                'position': np.array([0.0, 0.0, 0.0]),
                'element': 'Al',
                'orbital': 'sp3',
                'energy': 0,
                'neighbors': [
                    {'delta': np.array([0.0, 0.5, 0.5]), 'name': 'Al2', 'hopping': -1.79},
                    {'delta': np.array([0.5, 0.0, 0.5]), 'name': 'Al3', 'hopping': -1.79},
                    {'delta': np.array([0.5, 0.5, 0.0]), 'name': 'Al4', 'hopping': -1.79}
                ]
            },
            {
                'name': 'Al2',
                'position': np.array([0.0, 0.5, 0.5]),
                'element': 'Al',
                'orbital': 'sp3',
                'energy': 0,
                'neighbors': [
                    {'delta': np.array([0.0, -0.5, -0.5]), 'name': 'Al1', 'hopping': -1.79},
                    {'delta': np.array([0.5, 0.0, 0.0]), 'name': 'Al3', 'hopping': -1.79},
                    {'delta': np.array([0.5, -0.5, -0.5]), 'name': 'Al4', 'hopping': -1.79}
                ]
            },
            {
                'name': 'Al3',
                'position': np.array([0.5, 0.0, 0.5]),
                'element': 'Al',
                'orbital': 'sp3',
                'energy': 0,
                'neighbors': [
                    {'delta': np.array([-0.5, 0.0, -0.5]), 'name': 'Al1', 'hopping': -1.79},
                    {'delta': np.array([-0.5, 0.5, -0.5]), 'name': 'Al2', 'hopping': -1.79},
                    {'delta': np.array([0.0, 0.5, 0.0]), 'name': 'Al4', 'hopping': -1.79}
                ]
            },
            {
                'name': 'Al4',
                'position': np.array([0.5, 0.5, 0.0]),
                'element': 'Al',
                'orbital': 'sp3',
                'energy': 0,
                'neighbors': [
                    {'delta': np.array([-0.5, -0.5, 0.0]), 'name': 'Al1', 'hopping': -1.79},
                    {'delta': np.array([-0.5, 0.0, -0.5]), 'name': 'Al2', 'hopping': -1.79},
                    {'delta': np.array([0.0, -0.5, 0.5]), 'name': 'Al3', 'hopping': -1.79}
                ]
            }
        ]
    }
}

for key in species:
    if species[key]['direct vectors'] is not None and species[key]['reciprocal vectors'] is None:
        species[key]['reciprocal vectors'] = _compute_reciprocal_coordinates(species[key]['direct vectors'])
    elif species[key]['direct vectors'] is None and species[key]['reciprocal vectors'] is not None:
        species[key]['direct vectors'] = _compute_reciprocal_coordinates(species[key]['reciprocal vectors'], inverse=True)


@dataclass
class crystal:
    name: str
    grid: Grid
    direct_lattice_unit: float
    reciprocal_lattice_unit: float
    direct_vectors: np.ndarray # (3, 3)
    reciprocal_vectors: np.ndarray # (3, 3)
    num_wann: int # number of wannier functions or orbitals
    nrpts: int # number of points in the Brillouin zone
    ir0: int # index of the central wigner-seitz cell
    orbital_names: list # (num_wann,) list of the names of the orbitals
    deg_weights: np.ndarray # (nrpts,) weights for the integration over the Brillouin zone
    R_vectors: np.ndarray # (nrpts, 3) R vectors for the integration over the Brillouin zone
    h_Rnm: np.ndarray # (nrpts, num_wann, num_wann) hopping matrix for the integration over the Brillouin zone
    r_Rnm: np.ndarray # (nrpts, num_wann, num_wann, 3) position matrix for the integration over the Brillouin zone     
    deltas: np.ndarray # (num_wann, 3) position of the orbitals
    max_hopping: float
    threshold_hopping: float

    @classmethod
    def from_W90_TB_file(cls, filename:str, grid: Grid, species_name:str=None, threshold_hopping:float=1):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Wannier90 file not found: {filename}")
        if filename[-6:].lower() != 'tb.dat':
            raise ValueError(f"File is not a Wannier90 TB file: {filename}")

        with open(filename, 'r') as f:
            lines = f.readlines()

        logging.info(f"Read Wannier90 TB file: {filename}")


        name=species_name
        direct_lattice_unit = 1.0e-10
        reciprocal_lattice_unit = 2*np.pi/direct_lattice_unit

        idx = 0
        # --- Header ---
        idx += 1  # skip comment/timestamp line

        # Lattice vectors (3 lines, 3 floats each), in Angstrom
        lattice_vectors = []
        for _ in range(3):
            lattice_vectors.append(np.array([float(x) for x in lines[idx].split()]))
            idx += 1
        lattice_vectors = np.array(lattice_vectors)  # shape (3, 3), rows are a1, a2, a3
        direct_vectors = lattice_vectors # direct vectors  [a_1, a_2, a_3] coordinates in cartesian 
        reciprocal_vectors = _compute_reciprocal_coordinates(direct_vectors) # reciprocal vectors  [b_1, b_2, b_3] coordinates in cartesian

        # num_wann and nrpts
        num_wann = int(lines[idx].strip()); idx += 1 # Number of Wannier functions
        nrpts    = int(lines[idx].strip()); idx += 1 # Number of R-points

        orbital_names = list(string.ascii_uppercase[:num_wann])

        # Degeneracy weights: nrpts integers, may span multiple lines
        deg_weights = []
        while len(deg_weights) < nrpts:
            deg_weights.extend([int(x) for x in lines[idx].split()])
            idx += 1
        deg_weights= np.array(deg_weights)  # shape (nrpts,)

        # --- Hamiltonian h_Rnm blocks ---

        R_vectors = np.zeros((nrpts, 3), dtype=np.int32)
        h_Rnm       = np.zeros((nrpts, num_wann, num_wann), dtype=np.complex128)

        for ir in range(nrpts):
            while idx < len(lines) and lines[idx].strip() == '':
                idx += 1
            R_vectors[ir] = [int(x) for x in lines[idx].split()]
            if np.all(R_vectors[ir] == 0):
                ir0=ir  # index of the central wigner-seitz cell

            idx += 1
            for _ in range(num_wann * num_wann):
                parts = lines[idx].split()
                m, n = int(parts[0]) - 1, int(parts[1]) - 1  # convert to 0-indexed
                h_Rnm[ir, m, n] = complex(float(parts[2]), float(parts[3]))
                idx += 1

        # --- Position r_Rnm blocks in cartesian coordinates---
        r_Rnm = np.zeros((nrpts, num_wann, num_wann, 3), dtype=np.complex128)
        for ir in range(nrpts):
            while idx < len(lines) and lines[idx].strip() == '':
                idx += 1
            idx += 1  # skip R-vector line (same R-points as H_R)
            for _ in range(num_wann * num_wann):
                parts = lines[idx].split()
                m, n = int(parts[0]) - 1, int(parts[1]) - 1
                r_Rnm[ir, m, n, 0] = complex(float(parts[2]), float(parts[3]))  # a1
                r_Rnm[ir, m, n, 1] = complex(float(parts[4]), float(parts[5]))  # a2
                r_Rnm[ir, m, n, 2] = complex(float(parts[6]), float(parts[7]))  # a3
                idx += 1

        # Displacement of the orbitals in the wigner-seitz cell
        deltas = np.zeros((num_wann, 3), dtype=np.float64)
        for iwann in range(num_wann):
            deltas[iwann]=np.real(r_Rnm[ir0,iwann,iwann])

        logging.info(f"Found {num_wann} orbitals in each of {nrpts} R-points")

        # max hopping
        _h_Rnm = h_Rnm.copy()  
        # for n in range(num_wann):
        #     _h_Rnm[:,n,n] = 0   #exclude diagonals

        max_hopping = np.max(np.abs(_h_Rnm))
        logging.info(f"Max. inter-site hopping={max_hopping} (absolute value)")

        if threshold_hopping>0:
            logging.info(f"Retaining hoppings above {threshold_hopping*100}% --> threshold={threshold_hopping*max_hopping} (absolute value)")
            
            ir_to_keep=[ir for ir in range(nrpts) if np.abs(_h_Rnm[ir]).max() > threshold_hopping*max_hopping]
            
            h_Rnm=h_Rnm[ir_to_keep]
            r_Rnm=r_Rnm[ir_to_keep]
            R_vectors=R_vectors[ir_to_keep]
            deg_weights=deg_weights[ir_to_keep]

            nrpts = h_Rnm.shape[0]  

            logging.info(f"Keeping {nrpts} R-points")

            for ir in range(nrpts):
                if np.all(R_vectors[ir] == 0):
                    ir0=ir  # index of the central wigner-seitz cell
 
        return cls(
            name=name,
            grid=grid,
            direct_lattice_unit= direct_lattice_unit,
            reciprocal_lattice_unit= reciprocal_lattice_unit,
            direct_vectors= direct_vectors,
            reciprocal_vectors= reciprocal_vectors,
            num_wann= num_wann,
            nrpts= nrpts,
            ir0= ir0,
            orbital_names= orbital_names,
            deg_weights= deg_weights,
            R_vectors= R_vectors,
            h_Rnm= h_Rnm,
            r_Rnm= r_Rnm,
            deltas= deltas,
            max_hopping=max_hopping,
            threshold_hopping=threshold_hopping,
        )
    

