
import sys
sys.path.append("/home/lplaja/crystal_project/src")

import numpy as np
import TBcrystal as cr
import grid as grid
from TBevolution import TB_graphene_CMCP, qe, epsk, hbar
from Field import polarizedHarmonicElectricField, lambda2T, env_sin2
import logging
import time
from datetime import datetime

logging.basicConfig(
    filename="dipole.log",
    filemode="w", 
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

now=str(datetime.now())
logging.info(f"Started at {now}")



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
npt=1024*64
tt=grid.UniformCartesianGrid(1,limits=[0,8*T0], nptx=npt)

Efield=polarizedHarmonicElectricField(tt,I_W__cm2=5e10,lambda0_nm=lambda0*1e9, env=env_sin2, phi_rad=-np.pi/2, chi_rad=np.pi/2,ellip=0)

# Create the TightBinding object
gtb=TB_graphene_CMCP(gcr, Efield)

#calculate dipole
dipole_x,dipole_y=gtb.rk_dipole(npt=500)

#write results

filename='dipole.txt'

with open(filename, "w") as f:
    f.write(str(gcr))
    f.write(str(tt))
    f.write(str(Efield))
    f.write(str(gtb))
    f.write("# t  dipole_x    dipole_y\n")
    for i_t, i_dipole_x, i_dipole_y in zip(Efield.t, dipole_x, dipole_y):
        f.write(f"{i_t[0]:.16e} \t {i_dipole_x:.16e} \t {i_dipole_y:.16e} \n")


