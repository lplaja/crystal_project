import sys
sys.path.append("/home/lplaja/crystal_project/src")
from scipy.constants import eV
import numpy as np
import TBcrystal as cr
import grid as grid
import Field as Field
import TBevolution as TBevolution


def test_TBevolution_Bloch_parameters():
        # Create an hexagonal grid
        filter=grid.polygonfilter
        filter_args={'nsides':6, 'radius': 2/3}
        ggr=grid.UniformCartesianGrid(2,[[-0.75,0.75],[-0.75,0.75]],[500,500], origin=(0.0,0), filter=filter, filter_args= filter_args)

        # Create the graphene crystal

        TBcr=cr.crystal.from_W90_TB_file(filename='/home/lplaja/crystal_project/calculations/Wannier90 data/gr1NN_tb.dat' , 
                                                                        grid=ggr, species_name='graphene', threshold_hopping=0)
        
        lambda0=3000*1e-9
        T0=Field.lambda2T(lambda0)
        tini=0*T0
        tfin=8*T0
        limits=[tini,tfin]
        nptx=32768
        tt=grid.UniformCartesianGrid(1,limits=limits, nptx=nptx)

        fieldcallable=Field.polarizedHarmonicElectricField        
        I_W__cm2=5e10
        env_params={'start' : 0, 'end' : 1, 'ton'   : 0.5, 'toff'  : 0.5 } # in units of the time grid
        phi_rad=-1.5707963267948966
        chi_rad=1.5707963267948966
        ellip=0
        Efield=fieldcallable(tt,I_W__cm2=I_W__cm2,lambda0_nm=lambda0*1e9, env=Field.env_sin2, env_parameters=env_params, phi_rad=phi_rad, chi_rad=chi_rad,ellip=ellip)

        TBev=TBevolution.TBevolution_Bloch(TBcr, Efield)
        

        # self.h_Rnm = self.crystal.h_Rnm * eV / self.crystal.deg_weights[:, None, None]
        assert np.allclose(TBev.h_Rnm[0,:,:], np.array([[0. + 0.j, -2.8879914 + 0.j], [0. + 0.j, 0. + 0.j]])*eV)
        assert np.allclose(TBev.h_Rnm[2,:,:], np.array([[0. + 0.j, -2.8879914 + 0.j], [ -2.8879914 + 0.j, 0. + 0.j]])*eV)

        # self.R=cr.vector_in_cart(self.crystal.R_vectors, self.crystal.direct_vectors) # coordinate of the WZ cell in cartesian
        assert np.allclose(TBev.R[0], np.array([-2.1304224930000002, -1.2300000000000000,0]))
        assert np.allclose(TBev.R[1], np.array([-2.1304224930000002, 1.2300000000000000,0]))
        assert np.allclose(TBev.R[2], np.array([0, 0,0]))
        assert np.allclose(TBev.R[3], np.array([2.1304224930000002, -1.2300000000000000,0]))
        assert np.allclose(TBev.R[4], np.array([2.1304224930000002, 1.2300000000000000,0]))



def test_TBevolution_Bloch_bands():
        # Create an hexagonal grid
        filter=grid.polygonfilter
        filter_args={'nsides':6, 'radius': 2/3}
        ggr=grid.UniformCartesianGrid(2,[[-0.75,0.75],[-0.75,0.75]],[500,500], origin=(0.0,0), filter=filter, filter_args= filter_args)

        # Create the graphene crystal

        TBcr=cr.crystal.from_W90_TB_file(filename='/home/lplaja/crystal_project/calculations/Wannier90 data/gr1NN_tb.dat' , 
                                                                        grid=ggr, species_name='graphene', threshold_hopping=0)
        
        lambda0=3000*1e-9
        T0=Field.lambda2T(lambda0)
        tini=0*T0
        tfin=8*T0
        limits=[tini,tfin]
        nptx=32768
        tt=grid.UniformCartesianGrid(1,limits=limits, nptx=nptx)

        fieldcallable=Field.polarizedHarmonicElectricField        
        I_W__cm2=5e10
        env_params={'start' : 0, 'end' : 1, 'ton'   : 0.5, 'toff'  : 0.5 } # in units of the time grid
        phi_rad=-1.5707963267948966
        chi_rad=1.5707963267948966
        ellip=0
        Efield=fieldcallable(tt,I_W__cm2=I_W__cm2,lambda0_nm=lambda0*1e9, env=Field.env_sin2, env_parameters=env_params, phi_rad=phi_rad, chi_rad=chi_rad,ellip=ellip)

        TBev=TBevolution.TBevolution_Bloch(TBcr, Efield)

        a=1.42
        kx=np.array([0,2/3/a*np.sqrt(3)/2, 2/3/a*np.sqrt(3)/2] )
        ky=np.array([0,2/3/a*1/2,0])
        kz=np.array([0,0,0])
        en, eig=TBev.bands(kx,ky,kz)

        assert np.allclose(en[:,0], np.array([-8.6639742, 8.6639742])*eV)  # Gamma
        assert np.allclose(en[:,1], np.array([0, 0])*eV)  # K
        assert np.allclose(en[:,2], np.array([-2.888, 2.888])*eV)  # M

        # G = un vector primitivo de la red recíproca (en las mismas unidades que k)
        b = TBev.crystal.reciprocal_vectors      # lista de b_1, b_2, b_3 (sin 2π)
        G = np.array(b[0])                        # primer vector primitivo

        kx = np.array([0.13]); ky = np.array([0.27]); kz = np.array([0.0])
        t_k  = TBev.tnm(kx,      ky,      kz,)
        t_kG = TBev.tnm(kx+G[0], ky+G[1], kz+G[2])
  
        err = np.max(np.abs(t_kG - t_k))
        assert err < 1e-12*eV, f"t(k) no es periódico en G: max|Δ|={err}"

