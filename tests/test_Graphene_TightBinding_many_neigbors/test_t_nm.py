import sys
sys.path.append("/home/lplaja/crystal_project/src")
from scipy.constants import eV
import numpy as np
import TBcrystal as cr
import grid as grid
from TBevolution import _fk, TB_graphene_CMCP
from Field import polarizedElectricField, lambda2T, env_sin2

def test_t_nm():
    # Create an hexagonal grid
    filter=grid.polygonfilter
    filter_args={'nsides':6, 'radius': 2/3}
    ggr=grid.UniformCartesianGrid(2,[[-0.75,0.75],[-0.75,0.75]],[500,500], origin=(0.0,0), filter=filter, filter_args= filter_args)

    # Create the graphene crystal
    gcr=cr.crystal(cr.species['graphene'], grid=ggr, space='reciprocal')

    # Create the Field
    lambda0=3000e-9
    T0=lambda2T(lambda0)
    #npt=1024*32
    npt=1024*16
    tt=grid.UniformCartesianGrid(1,limits=[0,8*T0], nptx=npt)

    Efield=polarizedElectricField(tt,I_W__cm2=5e10,lambda0_nm=lambda0*1e9, env=env_sin2, phi_rad=-np.pi/2, chi_rad=np.pi/2,ellip=0)

    # Create the TightBinding object
    gtb=TB_graphene_CMCP(gcr, Efield)

    gamma0=-2.97
    kx=np.array([0, 1/np.sqrt(3), 1/np.sqrt(3)]) # points Gamma, M and K
    ky=np.array([0, 0, 1/3]) # points Gamma, M and K

    fg0k=-gamma0*_fk(kx,ky, gtb.crystal.direct_vectors)*np.exp(1j*2*np.pi*kx/np.sqrt(3)/2)
    tnm=gtb.tnm(kx,ky)/eV

    # print('Gamma point')
    # print(f'\t {fg0k[0]=}')
    # print(f'\t {tnm[:,:,0]=}')

    eps0=gtb.crystal.parameters['orbitals'][0]['energy']
    eps1=gtb.crystal.parameters['orbitals'][1]['energy']

    assert np.allclose(tnm[0,0,0], eps0)
    assert np.allclose(tnm[1,1,0], eps1)
    assert np.allclose(tnm[0,1,0], fg0k[0])
    assert np.allclose(tnm[1,0,0], np.conjugate(fg0k[0]))

    # print('M point')
    # print(f'\t {fg0k[1]=}')
    # print(f'\t {tnm[:,:,1]=}')

    assert np.allclose(tnm[0,0,1], eps0)
    assert np.allclose(tnm[1,1,1], eps1)
    assert np.allclose(tnm[0,1,1], fg0k[1])
    assert np.allclose(tnm[1,0,1], np.conjugate(fg0k[1]))

    # print('K point')
    # print(f'\t {fg0k[2]=}')
    # print(f'\t {tnm[:,:,2]=}')

    assert np.allclose(tnm[0,0,2], eps0)
    assert np.allclose(tnm[1,1,2], eps1)
    assert np.allclose(tnm[0,1,2], fg0k[2])
    assert np.allclose(tnm[1,0,2], np.conjugate(fg0k[2]))
