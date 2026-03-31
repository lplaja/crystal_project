import numpy as np
import TBcrystal as cr
import grid as grid
from Graphene_TightBinding import TB_graphene_CMCP, qe, epsk
from Field import electricField, lambda2T, env_sin2

def test_fk_Ek_phik_grads_Dk():

    # Create an hexagonal grid

    filter=grid.polygonfilter
    filter_args={'nsides':6, 'radius': 2/3}
    ggr=grid.UniformCartesianGrid(2,[[-0.75,0.75],[-0.75,0.75]],[100,100], filter=filter, filter_args= filter_args)

    # Create the graphene crystal
    gcr=cr.crystal(cr.species['graphene'], grid=ggr, space='reciprocal')

    # Create the Field
    T0=lambda2T(800e-9)
    tt=grid.UniformCartesianGrid(1,limits=[0,8*T0], nptx=1024*8)

    Efield=electricField(tt,I_W__cm2=5e10,lambda0_nm=800, env=env_sin2)

    # Create the TightBinding object

    gtb=TB_graphene_CMCP(gcr, Efield)

    assert gtb.crystal.direct_lattice_unit==2.46e-10
    assert gtb.crystal.reciprocal_lattice_unit==2*np.pi/gtb.crystal.direct_lattice_unit
    assert np.allclose(gtb.crystal.direct_vectors, np.array([[np.sqrt(3)/2,0.5],[np.sqrt(3)/2,-0.5]]))

    i_k=533
    kx=gtb.k[i_k,0]
    ky=gtb.k[i_k,1]

    # fk
    assert gtb.fk(gtb.k[i_k])==np.exp(-1j*2*np.pi*kx/np.sqrt(3))*(1+2*np.exp(1j*2*np.pi*np.sqrt(3)*kx/2)*np.cos(2*np.pi*ky/2))
    # Ek
    assert gtb.Ek(gtb.k[i_k])[0]==np.abs(gtb.fk(gtb.k[i_k]))*(-2.97*1.602176634e-19 )
    # phik
    assert np.tan(gtb.phik(gtb.k[i_k]))== np.imag(gtb.fk(gtb.k[i_k]))/np.real(gtb.fk(gtb.k[i_k]))
    # grads phi
    epskx=np.array([[1,0]])*epsk
    epsky=np.array([[0,1]])*epsk
    assert gcr.reciprocal_lattice_unit == 2*np.pi/gcr.direct_lattice_unit
    assert np.abs((gtb.grad_phik_x(gtb.k[i_k])-(gtb.phik(gtb.k[i_k]+epskx)-gtb.phik(gtb.k[i_k]-epskx))/2/epsk/2/np.pi*gcr.direct_lattice_unit)/gtb.grad_phik_x(gtb.k[i_k]))<1e-5 
    assert np.abs((gtb.grad_phik_y(gtb.k[i_k])-(gtb.phik(gtb.k[i_k]+epsky)-gtb.phik(gtb.k[i_k]-epsky))/2/epsk/2/np.pi*gcr.direct_lattice_unit)/gtb.grad_phik_y(gtb.k[i_k]))<1e-5
    # Dk
    Dkx=-gcr.direct_lattice_unit*qe/2/np.sqrt(3)*(1+np.cos(2*np.pi*ky/2)*(np.cos(2*np.pi*kx*np.sqrt(3)/2)-2*np.cos(2*np.pi*ky/2)))/ \
        (1+4*np.cos(2*np.pi*ky/2)*(np.cos(2*np.pi*kx*np.sqrt(3)/2)+np.cos(2*np.pi*ky/2)))
    Dky=-gcr.direct_lattice_unit*qe/2*np.sin(2*np.pi*ky/2)*np.sin(2*np.pi*kx*np.sqrt(3)/2)/ \
        (1+4*np.cos(2*np.pi*ky/2)*(np.cos(2*np.pi*kx*np.sqrt(3)/2)+np.cos(2*np.pi*ky/2)))
    assert np.abs((gtb.Dk(gtb.k[i_k])[0]-Dkx)/gtb.Dk(gtb.k[i_k])[0])<1e-5
    assert np.abs((gtb.Dk(gtb.k[i_k])[1]-Dky)/gtb.Dk(gtb.k[i_k])[1])<1e-5