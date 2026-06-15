import argparse
import tomllib

import sys
sys.path.append("/home/lplaja/crystal_project/src")

import numpy as np
import TBcrystal as cr
import grid as grid
import TBevolution
import Field as Field
import logging
import time
from datetime import datetime
from scipy.constants import eV

import pprint

def load_config(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)
    
grid_type={
    'UniformCartesianGrid':grid.UniformCartesianGrid,
    'UniformCylindricalGrid':grid.UniformCylindricalGrid
}

filter_type={
    'polygonfilter':grid.polygonfilter
}

evolution_type={
    'TBevolution_CMCP':TBevolution.TBevolution_CMCP,
    'TBevolution_Bloch':TBevolution.TBevolution_Bloch
}   

field_type={
    'polarizedHarmonicElectricField':Field.polarizedHarmonicElectricField,
    'polarizedHarmonicPotentialVectorField':Field.polarizedHarmonicPotentialVectorField
}

field_env_type={
    'env_sin2':Field.env_sin2
}

def main():
    parser = argparse.ArgumentParser(description='Tight-binding calculation')
    parser.add_argument('config', type=str, help='Path to config TOML file')
    args = parser.parse_args()

    config = load_config(args.config)

    outputfilename=config['crystal']['name']+'_'+config['calculation']['ID']
    logging.basicConfig(
        filename=outputfilename+".log",
        filemode="w", 
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    now=str(datetime.now())
    log_msg=f"Tight-binding calculation started at {now}"
    logging.info(log_msg)
    log_msg="="*len(log_msg)
    logging.info(log_msg)

    log_msg='\n '+pprint.pformat(config, indent=2)
    logging.info(log_msg)

    logging.info('Constructing the BZ')

    ndim=config['crystal']['dim']
    a=config['crystal']['a']
    W90filename=config['crystal']['W90filename']

    if ndim==1:
        limits=[config['BZ']['kx_min'],config['BZ']['kx_max']]  
        nk=[config['BZ']['nkx']]
    elif ndim==2:
        limits=[[config['BZ']['kx_min'],config['BZ']['kx_max']],[config['BZ']['ky_min'],config['BZ']['ky_max']]]  
        nk=[config['BZ']['nkx'],config['BZ']['nky']]
    else:
        limits=[[config['BZ']['kx_min'],config['BZ']['kx_max']],[config['BZ']['ky_min'],config['BZ']['ky_max']],[config['BZ']['kz_min'],config['BZ']['kz_max']]]  
        nk=[config['BZ']['nkx'],config['BZ']['nky'],config['BZ']['nkz']]

    if 'filter' in config['BZ']:
        if config['BZ']['filter']['type']=='polygonfilter':
            filter=filter_type[config['BZ']['filter']['type']]
            filter_args=config['BZ']['filter']['args']
        else:
            filter=None
            filter_args={}
    else:
        filter=None
        filter_args={}

    gr=grid.UniformCartesianGrid(ndim,limits,nk, origin=(0.0,0), filter=filter, filter_args= filter_args)

    print(f'vprint in line: 97 --> {ndim=}')
    print(f'vprint in line: 97 --> {limits=}')
    print(f'vprint in line: 97 --> {nk=}')
    print(f'vprint in line: 97 --> {filter=}')
    print(f'vprint in line: 97 --> {filter_args=}')

    logging.info('Constructing the TB crystal')

    filename=config['crystal']['W90filename']
    species_name=config['crystal']['name']
    threshold_hopping=config['crystal']['threshold_hopping']

    TBcr=cr.crystal.from_W90_TB_file(filename=filename , 
                                    grid=gr, species_name=species_name, threshold_hopping=threshold_hopping)
    
    print(f'vprint in line: 109 --> {filename=}')
    print(f'vprint in line: 109 --> {species_name=}')
    print(f'vprint in line: 109 --> {threshold_hopping=}')

    logging.info('Constructing the temporal grid')

    lambda0=config['Field']['lambda']*1e-9
    T0=Field.lambda2T(lambda0)
    tini=config['time']['tini']*T0
    tfin=config['time']['tfin']*T0
    limits=[tini,tfin]
    nptx=config['time']['nt']
    tt=grid.UniformCartesianGrid(1,limits=limits, nptx=nptx)

    print(f'vprint in line: 120 --> {lambda0=}')
    print(f'vprint in line: 120 --> {T0=}')
    print(f'vprint in line: 120 --> {tini=}')
    print(f'vprint in line: 120 --> {tfin=}')
    print(f'vprint in line: 120 --> {nptx=}')

    logging.info('Constructing the Field')

    I_W__cm2=config['Field']['I_W__cm2']
    env=field_env_type[config['Field']['env']['type']]
    env_params=config['Field']['env']['args']
    phi_rad=config['Field']['phi']
    chi_rad=config['Field']['chi']
    ellip=config['Field']['ellip']

    Efield=Field.polarizedHarmonicElectricField(tt,I_W__cm2=I_W__cm2,lambda0_nm=lambda0*1e9, env=env, env_parameters=env_params, phi_rad=phi_rad, chi_rad=chi_rad,ellip=ellip)
    print(f'vprint in line: 143 --> {I_W__cm2=}')
    print(f'vprint in line: 143 --> {env=}')
    print(f'vprint in line: 143 --> {env_params=}')
    print(f'vprint in line: 143 --> {phi_rad=}')
    print(f'vprint in line: 143 --> {chi_rad=}')
    print(f'vprint in line: 143 --> {ellip=}')
    print(f'vprint in line: 46 --> {max(Efield.E[:,0])=}')
    print(f'vprint in line: 46 --> {max(Efield.E[:,1])=}')


    logging.info('Constructing the TB evolver')

    evolcallable=evolution_type[config['evolver']['type']]
    TBev=evolcallable(TBcr, Efield)

    formatted = pprint.pformat(config, indent=2)
    commented = '\n'.join('# ' + line for line in formatted.splitlines())

    if config['evolver']['calculation']['type']=='rk_dipole':
        logging.info('Starting the calculation: rk_dipole')
        nt=config['evolver']['calculation']['args']['nt']
        print(f'vprint in line: 162 --> {nt=}')
        caltime_dip,dipole_x,dipole_y=TBev.rk_dipole(npt=nt) #callable = TBev.rk_dipole

        filename=outputfilename+'_'+config['evolver']['calculation']['type']+'.txt'

        logging.info('Calculation finished. Writing to '+filename)

        with open(filename, "w") as f:
            msg1='# CALCULATION: '+config['evolver']['calculation']['type']
            f.write(msg1+'\n')
            msg2='# DATE: '+datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(msg2+'\n')
            msg='='*len(max(msg1,msg2))
            f.write('#'+msg+'\n')
            f.write(commented+'\n')
            f.write('#'+msg+'\n')
            f.write("# t  dipole_x    dipole_y\n")
            for i_t, i_dipole_x, i_dipole_y in zip(caltime_dip, dipole_x, dipole_y):
                f.write(f"{i_t:.16e} \t {i_dipole_x:.16e} \t {i_dipole_y:.16e} \n")


if __name__ == '__main__':
    main()