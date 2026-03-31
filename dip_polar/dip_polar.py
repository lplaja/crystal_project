
import sys
sys.path.append("/home/lplaja/crystal_project/src")

import numpy as np
import TBcrystal as cr
import grid as grid
from Graphene_TightBinding import TB_graphene_CMCP, qe, epsk, hbar
from Field import polarizedElectricField, lambda2T, env_sin2
import logging
import time
from datetime import datetime

logging.basicConfig(
    filename="dipole_polar.log",
    filemode="w", 
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

now=str(datetime.now())
logging.info(f"Started at {now}")

# Create two polar grids grid
#filter=grid.polygonfilter
#filter_args={'nsides':6, 'radius': 2/3}
ggr_p=grid.UniformCylindricalGrid(2,[[0,1/3],[-np.pi,np.pi]],[200,72], offset=(0,2/3), cartesian=True)
ggr_m=grid.UniformCylindricalGrid(2,[[0,1/3],[-np.pi,np.pi]],[200,72], offset=(0,-2/3), cartesian=True)

#create the comlpementary mesh

def Complementaryfilter(x, *args):
    def distance(a, b): # auxiliar funtion to define the cross product of 2D vectors
        v=b-a
        return np.sqrt(v[0]**2+v[1]**2)
    phi=np.arange(0, 2*np.pi, np.pi/3)
    ylim=2/3
    radius=1/3
    polygon_vertices=np.column_stack((ylim*np.sin(phi), ylim*np.cos(phi)))
    x_test=np.zeros(x.shape[0], dtype=bool)
    for n, ix in enumerate(x):
        test=[distance(polygon_vertices[i],ix)>radius for i in range(6)]
        x_test[n]=np.all(test)
    return x_test

xlim=1/np.sqrt(3)
ylim=1/np.sqrt(3)*np.cos(np.pi/6)
ggr_c=grid.UniformCartesianGrid(2,[[-xlim,xlim],[-ylim, ylim]],[400,400],filter=Complementaryfilter)


filename='grid_p.txt'
with open(filename, "w") as f:
    f.write('######  SPATIAL GRID')
    f.write("# x y \n")
    for ix, iy in zip(ggr_p.x[:,0], ggr_p.x[:,1]):
        f.write(f"{ix:.16e} \t {iy:.16e} \n")

filename='grid_m.txt'
with open(filename, "w") as f: 
    f.write('######  SPATIAL GRID')
    f.write("# x y \n")
    for ix, iy in zip(ggr_m.x[:,0], ggr_m.x[:,1]):
        f.write(f"{ix:.16e} \t {iy:.16e} \n")

filename='grid_c.txt'
with open(filename, "w") as f:
    f.write('######  SPATIAL GRID')
    f.write("# x y \n")
    for ix, iy in zip(ggr_c.x[ggr_c.inGrid][:,0], ggr_c.x[ggr_c.inGrid][:,1]):
        f.write(f"{ix:.16e} \t {iy:.16e} \n")

# Create the graphene crystal
gcr_p=cr.crystal(cr.species['graphene'], grid=ggr_p, space='reciprocal')
gcr_m=cr.crystal(cr.species['graphene'], grid=ggr_m, space='reciprocal')
gcr_c=cr.crystal(cr.species['graphene'], grid=ggr_c, space='reciprocal')


# Create the Field
lambda0=3000e-9
T0=lambda2T(lambda0)
#npt=1024*32
npt=1024*64
tt=grid.UniformCartesianGrid(1,limits=[0,8*T0], nptx=npt)

Efield=polarizedElectricField(tt,I_W__cm2=5e10,lambda0_nm=lambda0*1e9, env=env_sin2, phi_rad=-np.pi/2, chi_rad=np.pi/2,ellip=0)

# Create the TightBinding object
gtb_p=TB_graphene_CMCP(gcr_p, Efield)
gtb_m=TB_graphene_CMCP(gcr_m, Efield)
gtb_c=TB_graphene_CMCP(gcr_c, Efield)

#calculate dipole
time_dip, dipole_x_p,dipole_y_p=gtb_p.rk_dipole(npt=500)
time_dip, dipole_x_m,dipole_y_m=gtb_m.rk_dipole(npt=500)
time_dip, dipole_x_c,dipole_y_c=gtb_c.rk_dipole(npt=500)

dipole_x=dipole_x_p+dipole_x_m+dipole_x_c
dipole_y=dipole_y_p+dipole_y_m+dipole_y_c

#write results

filename='dipole_polar.txt'

with open(filename, "w") as f:
    f.write('######  SPATIAL GRIDS')
    f.write(str(gcr_p))
    f.write(str(gcr_m))
    f.write(str(gcr_c))
    f.write('######  TEMPORAL GRID')
    f.write(str(tt))
    f.write(str(Efield))
    f.write('######  TIGHTBINDIG OBJECTS')
    f.write(str(gtb_p))
    f.write(str(gtb_m))
    f.write(str(gtb_c))
    f.write("# t  dipole_x    dipole_y\n")
    for i_t, i_dipole_x, i_dipole_y, i_Ex, i_Ey in zip(time_dip, dipole_x, dipole_y,Efield.E[:,0],  Efield.E[:,1]):
        f.write(f"{i_t:.16e} \t {i_dipole_x:.16e} \t {i_dipole_y:.16e} \n")

filename='field.txt'
with open(filename, "w") as f:
    f.write(str(gcr_p))
    f.write(str(tt))
    f.write(str(Efield))
    f.write(str(gtb_p))
    f.write("# t  Efield_x  Efield_y\n")
    for i_t, i_Ex, i_Ey, i_Ax, i_Ay in zip(Efield.t[:,0], Efield.E[:,0],  Efield.E[:,1], Efield.A[:,0],  Efield.A[:,1] ):
        f.write(f"{i_t:.16e} \t {i_Ex:.16e} \t {i_Ey:.16e} \t {i_Ax:.16e} \t {i_Ay:.16e}  \n")


