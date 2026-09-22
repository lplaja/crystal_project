import sys
sys.path.append("/home/lplaja/crystal_project/src")
from scipy.constants import eV
import numpy as np
import TBcrystal as cr
import grid as grid

def test_crystal_parameters():
    # Create an hexagonal grid
    filter=grid.polygonfilter
    filter_args={'nsides':6, 'radius': 2/3}
    ggr=grid.UniformCartesianGrid(2,[[-0.75,0.75],[-0.75,0.75]],[500,500], origin=(0.0,0), filter=filter, filter_args= filter_args)

    # Create the graphene crystal

    TBcr=cr.crystal.from_W90_TB_file(filename='/home/lplaja/crystal_project/calculations/Wannier90 data/gr1NN_tb.dat' , 
                                    grid=ggr, species_name='graphene', threshold_hopping=0)
    
    assert np.allclose(TBcr.direct_vectors, [[2.1304224930000002, 1.2300000000000000,0.0000000000000000],     
            [2.1304224930000002, -1.2300000000000000, 0.0000000000000000],     
            [0.0000000000000000, 0.0000000000000000, 15.000000000000000]]) #rows are a1, a2, a3
    
    a=2.46
    b=2/np.sqrt(3)/a
    b1=b*np.array([1/2, np.sqrt(3)/2,0])
    b2=b*np.array([1/2, -np.sqrt(3)/2,0])
    b3=np.array([0, 0, 1/15])
    assert np.allclose(TBcr.reciprocal_vectors[0], b1)
    assert np.allclose(TBcr.reciprocal_vectors[1], b2)
    assert np.allclose(TBcr.reciprocal_vectors[2], b3)
    
    assert TBcr.num_wann == 2
    assert TBcr.nrpts == 5
    assert TBcr.ir0 == 2
    assert TBcr.orbital_names == ['A', 'B']
    assert np.allclose(TBcr.deg_weights, [1, 1, 1, 1, 1])

    assert np.allclose(TBcr.R_vectors, [[-1, 0, 0], [0, -1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert np.allclose(TBcr.h_Rnm[0,:,:], [[0. + 0.j, -2.8879914 + 0.j], [0. + 0.j, 0. + 0.j]])
    assert np.allclose(TBcr.h_Rnm[2,:,:], [[0. + 0.j, -2.8879914 + 0.j], [ -2.8879914 + 0.j, 0. + 0.j]])
    
    assert np.allclose(TBcr.r_Rnm[0,0,1], [0,0,0])
    assert np.allclose(TBcr.r_Rnm[2,1,1], [0.14202816E+01,0,0])

    assert np.allclose(TBcr.deltas[0],[0,0,0])
    assert np.allclose(TBcr.deltas[1],[0.14202816E+01,0,0])


