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
    
def list_params(config, sep='.'):
    """Aplana el config y devuelve {ruta: longitud} solo para los parametros
    definidos como lista (candidatos a barrido de varios calculos)."""
    out = {}
    def walk(d, prefix=''):
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, key)
            elif isinstance(v, list):
                out[key] = len(v)
    walk(config)
    return out

def check_consistent(config):
    lp = list_params(config)
    if not lp:
        return 1                      # ningun parametro en lista: un solo calculo
    n = set(lp.values())
    if len(n) > 1:
        detalle = ', '.join(f"{k}={v}" for k, v in lp.items())
        raise ValueError(f"Listas con longitudes distintas: {detalle}")
    return n.pop()

def take_element_from_config_input(config_input, index):
    def walk(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = walk(v)
            elif isinstance(v, list):
                out[k] = v[index]
            else:
                out[k] = v
        return out
    return walk(config_input)
    
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

    config_input = load_config(args.config)

    list_in_config_length = check_consistent(config_input)

    if list_in_config_length > 1:
        IDs = config_input['calculation']['ID']
        if not isinstance(IDs, list) or len(IDs) != list_in_config_length:
            raise ValueError(f"IDs must be a list of length {list_in_config_length}")

    config = [take_element_from_config_input(config_input, i) for i in range(list_in_config_length)]

    for i_config in config:
        outputfilename=i_config['crystal']['name']+'_'+i_config['calculation']['ID']
        logging.basicConfig(
            filename=outputfilename+".log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            force=True,
        )

        now=str(datetime.now())
        log_msg=f"Tight-binding calculation started at {now}"
        logging.info(log_msg)
        log_msg="="*len(log_msg)
        logging.info(log_msg)

        log_msg='\n '+pprint.pformat(i_config, indent=2)
        logging.info(log_msg)

        logging.info('Constructing the BZ')

        ndim=i_config['crystal']['dim']
        a=i_config['crystal']['a']
        
        if ndim==1:
            limits=[i_config['BZ']['kx_min'],i_config['BZ']['kx_max']]  
            nk=[i_config['BZ']['nkx']]
        elif ndim==2:
            limits=[[i_config['BZ']['kx_min'],i_config['BZ']['kx_max']],[i_config['BZ']['ky_min'],i_config['BZ']['ky_max']]]  
            nk=[i_config['BZ']['nkx'],i_config['BZ']['nky']]
        else:
            limits=[[i_config['BZ']['kx_min'],i_config['BZ']['kx_max']],[i_config['BZ']['ky_min'],i_config['BZ']['ky_max']],[i_config['BZ']['kz_min'],i_config['BZ']['kz_max']]]  
            nk=[i_config['BZ']['nkx'],i_config['BZ']['nky'],i_config['BZ']['nkz']]

        if 'filter' in i_config['BZ']:
            if i_config['BZ']['filter']['type']=='polygonfilter':
                filter=filter_type[i_config['BZ']['filter']['type']]
                filter_args=i_config['BZ']['filter']['args']
            else:
                filter=None
                filter_args={}
        else:
            filter=None
            filter_args={}

        dx=[(ilimit[1]-ilimit[0])/nk[i_n] for i_n,ilimit in enumerate(limits)]
        origin=tuple(d/2 for d in dx)
        gr=grid.UniformCartesianGrid(ndim,limits,nk, origin=origin, filter=filter, filter_args= filter_args)
        logging.info(f'Number of points in the original Grid: {gr.npts}')
        logging.info(f'Number of points in the filtered Grid: {gr.inGrid.sum()}')


        logging.info('Constructing the TB crystal')

        filename=i_config['crystal']['W90filename']
        species_name=i_config['crystal']['name']
        threshold_hopping=i_config['crystal']['threshold_hopping']

        TBcr=cr.crystal.from_W90_TB_file(filename=filename , 
                                        grid=gr, species_name=species_name, threshold_hopping=threshold_hopping)

        logging.info('Constructing the temporal grid')

        lambda0=i_config['Field']['lambda']*1e-9
        T0=Field.lambda2T(lambda0)
        tini=i_config['time']['tini']*T0
        tfin=i_config['time']['tfin']*T0
        limits=[tini,tfin]
        nptx=i_config['time']['nt']
        tt=grid.UniformCartesianGrid(1,limits=limits, nptx=nptx)

        logging.info('Constructing the Field')

        I_W__cm2=i_config['Field']['I_W__cm2']
        env=field_env_type[i_config['Field']['env']['type']]
        env_params=i_config['Field']['env']['args']
        varphi_rad=i_config['Field']['phi']
        chi_rad=i_config['Field']['chi']
        ellip=i_config['Field']['ellip']

#        logging.info(f'\t Intensity \t {I_W__cm2}  W/cm^2')
#        logging.info(f'\t Envelope \t {env}')
#        logging.info(f'\t \t env_params \t {env}')
#        logging.info(f'\t \t phi \t {varphi_rad}')
#        logging.info(f'\t \t chi \t {chi_rad}')
#        logging.info(f'\t \t ellip \t {ellip}')


        Efield=Field.polarizedHarmonicElectricField(tt,I_W__cm2=I_W__cm2,lambda0_nm=lambda0*1e9, 
                                                    env=env, env_parameters=env_params, 
                                                    varphi_rad=varphi_rad, 
                                                    chi_rad=chi_rad,ellip=ellip)

        logging.info('Constructing the TB evolver')

        evolcallable=evolution_type[i_config['evolver']['type']]
        TBev=evolcallable(TBcr, Efield)

        formatted = pprint.pformat(i_config, indent=2)
        commented = '\n'.join('# ' + line for line in formatted.splitlines())

        if i_config['evolver']['calculation']['type']=='dipole_velocity':
            logging.info('Starting the calculation: dipole_velocity')
            nt=i_config['evolver']['calculation']['args']['nt']
            caltime_dip, vx, vy, vz = TBev.rk_dipole_velocity(npt=nt)

            filename=outputfilename+'_'+i_config['evolver']['calculation']['type']+'.txt'

            logging.info('Calculation finished. Writing to '+filename)

            with open(filename, "w") as f:
                msg1='# CALCULATION: '+i_config['evolver']['calculation']['type']
                f.write(msg1+'\n')
                msg2='# DATE: '+datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(msg2+'\n')
                msg='='*len(max(msg1,msg2))
                f.write('#'+msg+'\n')
                f.write(commented+'\n')
                f.write('#'+msg+'\n')
                f.write("# t  vx  vy      (v = sum_k dV v_k, dV en 1/A^2, v en m/s)\n")
                for i_t, i_vx, i_vy in zip(caltime_dip, vx.real, vy.real):
                    f.write(f"{i_t:.16e} \t {i_vx:.16e} \t {i_vy:.16e} \n") 


if __name__ == '__main__':
    main()