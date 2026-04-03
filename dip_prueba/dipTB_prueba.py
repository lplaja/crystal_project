
import sys
sys.path.append("/home/lplaja/crystal_project/src")

import numpy as np
import TBcrystal as cr
import grid as grid
from TBevolution import TBevolution_CMCP, qe, epsk, hbar
from Field import polarizedHarmonicElectricField, lambda2T, env_sin2
import logging
import time
from datetime import datetime
from scipy.constants import hbar, elementary_charge, eV


logging.basicConfig(
    filename="dipTB_prueba.log",
    filemode="w", 
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

now=str(datetime.now())
logging.info(f"Started at {now}")

# Create an hexagonal grid
a=2.46
filter=grid.polygonfilter
filter_args={'nsides':6, 'radius': 2/3/a}
# ggr=grid.UniformCartesianGrid(2,[[-0.75,0.75],[-0.75,0.75]],[700,700], origin=(0.0,0), filter=filter, filter_args= filter_args)
ggr=grid.UniformCartesianGrid(2,[[-0.75/a,0.75/a],[-0.75/a,0.75/a]],[70,70], origin=(0.0,0), filter=filter, filter_args= filter_args)


# Create the graphene crystal
gcr=cr.crystal.from_W90_TB_file(filename='/home/lplaja/crystal_project/dip_prueba/Wannier90 data/gr_tb.dat', grid=ggr, species_name='graphene1NN')

# Create the Field
lambda0=3000e-9
T0=lambda2T(lambda0)
npt=1024*32
#npt=1024*64
tt=grid.UniformCartesianGrid(1,limits=[0,8*T0], nptx=npt)

Efield=polarizedHarmonicElectricField(tt,I_W__cm2=5e10,lambda0_nm=lambda0*1e9, env=env_sin2, phi_rad=-np.pi/2, chi_rad=np.pi/2,ellip=0)

# Create the TightBinding evolution
gtb=TBevolution_CMCP(gcr, Efield)

# import time
# kx, ky, kz = gtb.k[:,0], gtb.k[:,1], gtb.k[:,2]

# # warm-up
# _ = gtb.tnm(kx, ky, kz)

# t0 = time.perf_counter()
# for _ in range(100):
#     _ = gtb.tnm(kx, ky, kz)
# print(f"_t_nm: {(time.perf_counter()-t0)/100*1000:.2f} ms")

#calculate dipole
time_dip,dipole_x,dipole_y=gtb.rk_dipole(npt=350)

#print(f'vprint in line: 48 --> {time_dip=}')

#write results

filename='dipTB_prueba.txt'

with open(filename, "w") as f:
    #f.write(str(gcr))
    #f.write(str(tt))
    #f.write(str(Efield))
    #f.write(str(gtb))
    f.write("# t  dipole_x    dipole_y\n")
    for i_t, i_dipole_x, i_dipole_y in zip(time_dip, dipole_x, dipole_y):
        f.write(f"{i_t:.16e} \t {i_dipole_x:.16e} \t {i_dipole_y:.16e} \n")