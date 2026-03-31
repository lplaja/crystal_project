import numpy as np
from grid import Grid


def _compute_reciprocal_coordinates(vectors_coordinates, inverse=False):
    # computes the reciprocal vectors from the direct vectors or the reverse
    # reciprocal vectors are ginen in units of 2pi/a
    # direct vectors are given in units of a
    A = np.stack(vectors_coordinates).T  # d x d matrix
    if inverse:
        B =  np.linalg.inv(A).T
    else:
        B =  np.linalg.inv(A.T)
    return list(B.T)

species = {
    'Simple 1D Chain': {
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
                'position': np.array([-1/6, -1/6]),  #orbital position in the direct vector basis {a_1, a_2, a_3}
                'element': 'C',
                'orbital': 'pz',
                'energy': -0.28, 
                'neighbors': [
                    {'name': 'B', 'delta': np.array([-1/3, -1/3]),  'hopping': -2.97},  # delta is the relative position of the neighbor in the direct vector basis
                    {'name': 'B', 'delta': np.array([ 2/3, -1/3]), 'hopping': -2.97},
                    {'name': 'B', 'delta': np.array([-1/3,  2/3]), 'hopping': -2.97},
                ]
            },
            {
                'name': 'B',
                'position': np.array([1/6, 1/6]),  #orbital position in the direct vector basis {a_1, a_2, a_3}
                'element': 'C',
                'orbital': 'pz',
                'energy': -0.28, 
                'neighbors': [
                    {'name': 'A', 'delta': np.array([1/3, 1/3]), 'hopping': -2.97},  # delta is the relative position of the neighbor in the direct vector basis
                    {'name': 'A', 'delta': np.array([-2/3, 1/3]),  'hopping': -2.97},
                    {'name': 'A', 'delta': np.array([1/3, -2/3]), 'hopping': -2.97},
                ]
            }
        ]
    },
    'aluminium': {
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

# Postprocessing of species

for key in species:
    if species[key]['direct vectors'] is not None and species[key]['reciprocal vectors'] is None:
        species[key]['reciprocal vectors'] = _compute_reciprocal_coordinates(species[key]['direct vectors'])
    elif species[key]['direct vectors'] is None and species[key]['reciprocal vectors'] is not None:
        species[key]['direct vectors'] = _compute_reciprocal_coordinates(species[key]['reciprocal vectors'], inverse=True)

class crystal:
    
    def __init__(self, crystal_species: dict= species['graphene'], space:str='reciprocal', grid:Grid=None):
        
        self.parameters= crystal_species 
        self.grid = grid
        self.space = space  #  'direct' or 'reeciprocal'
        self.direct_lattice_unit = self.parameters['direct lattice unit']

        # # Direct ad reciprocal vector coordinates are expresed in the cartesian basis normalized to lattice units (i.e. taking a=1
        # # A vector is expresses in coordinates as [a_1,a_2.a_3] for the vector bais {e_1, e_2, e_3}
        # direct_vector_coordinates = self.parameters['direct vectors']
        # reciprocal_vectors_coordinates = self.parameters['reciprocal vectors']
        # if direct_vector_coordinates is not None:
        #     reciprocal_vectors_coordinates = self._compute_reciprocal_coordinates(direct_vector_coordinates)
        # elif reciprocal_vectors_coordinates is not None:
        #     direct_vector_coordinates = self._compute_reciprocal_coordinates(reciprocal_vectors_coordinates, inverse=True)
        # else:
        #     raise ValueError("Must provide either direct_vectors or reciprocal_vectors.")
        
        self.direct_vectors= self.parameters['direct vectors']
        self.reciprocal_vectors= self.parameters['reciprocal vectors']

        if self.space == 'direct':
            self.vectors = np.array(self.direct_vectors)
        elif self.space == 'reciprocal':
            self.vectors = np.array(self.reciprocal_vectors)
       
    def vector_in_cart(self, vector, basis):
        return  basis @ vector

    # def _coordinates_to_vector(self, coordinates):
    #     # vectors are columns with the first coordinate in the highest row 
    #     # coord=[x,y,z]  --> vec=np.array([x][y][z])

    #     return np.array(coordinates)

    # def _compute_reciprocal_coordinates(self, vectors_coordinates, inverse=False):
    #     # computes the reciprocal vectors from the direct vectors or the reverse
    #     # reciprocal vectors are ginen in units of 2pi/a
    #     # direct vectors are given in units of a
    #     A = np.stack(vectors_coordinates).T  # d x d matrix
    #     if inverse:
    #         B =  np.linalg.inv(A).T
    #     else:
    #         B =  np.linalg.inv(A.T)
    #     return list(B.T)


    
