import sys
sys.path.append("/home/lplaja/crystal_project/src")
from scipy.constants import eV
import numpy as np
import TBcrystal as cr
import grid as grid
import Field as Field
import TBevolution as TBevolution

A_LATT = 2.46   # Å, constante de red del grafeno (gr1NN_tb.dat)

def hexagonal_BZ_grid(n=200):
    """Malla cuyo filtro deja UNA zona de Brillouin de grafeno (k en 1/Å, sin 2π)."""
    R = 2/(3*A_LATT)                    # |Γ-K| = 0.271 1/Å
    L = 1.1*R
    dx = 2*L/n
    return grid.UniformCartesianGrid(2, [[-L, L], [-L, L]], [n, n], origin=(dx/2, dx/2),
                                     filter=grid.polygonfilter, filter_args={'nsides': 6, 'radius': R})

def test_TBevolution_Bloch_parameters():
        # Create an hexagonal grid
        ggr = hexagonal_BZ_grid()

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
        varphi_rad=-1.5707963267948966
        chi_rad=1.5707963267948966
        ellip=0
        Efield=fieldcallable(tt,I_W__cm2=I_W__cm2,lambda0_nm=lambda0*1e9, env=Field.env_sin2, env_parameters=env_params, varphi_rad=varphi_rad, chi_rad=chi_rad,ellip=ellip)

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
        ggr = hexagonal_BZ_grid()

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
        varphi_rad=-1.5707963267948966
        chi_rad=1.5707963267948966
        ellip=0
        Efield=fieldcallable(tt,I_W__cm2=I_W__cm2,lambda0_nm=lambda0*1e9, env=Field.env_sin2, env_parameters=env_params, varphi_rad=varphi_rad, chi_rad=chi_rad,ellip=ellip)

        TBev=TBevolution.TBevolution_Bloch(TBcr, Efield)

        a=2.46
        kx=np.array([0,2/3/a*np.sqrt(3)/2, 2/3/a*np.sqrt(3)/2] )
        ky=np.array([0,2/3/a*1/2,0])
        kz=np.array([0,0,0])
        en, eig=TBev.bands(kx,ky,kz)

        print(en[:,0])

        assert np.allclose(en[:,0]/eV, np.array([-8.6639742, 8.6639742]), atol=1e-6) # Gamma
        assert np.allclose(en[:,1]/eV, np.array([0, 0] ), atol=1e-6) # K
        assert np.allclose(en[:,2]/eV, np.array([-2.888, 2.888]), atol=1e-6)   # M

        # G = un vector primitivo de la red recíproca (en las mismas unidades que k)
        b = TBev.crystal.reciprocal_vectors      # lista de b_1, b_2, b_3 (sin 2π)
        G = np.array(b[0])                        # primer vector primitivo

        kx = np.array([0.13]); ky = np.array([0.27]); kz = np.array([0.0])
        t_k  = TBev.tnm(kx,      ky,      kz,)
        t_kG = TBev.tnm(kx+G[0], ky+G[1], kz+G[2])
  
        err = np.max(np.abs(t_kG - t_k))
        assert err < 1e-12*eV, f"t(k) no es periódico en G: max|Δ|={err}"

def test_grad_t_nm_matches_finite_difference():

       # Create an hexagonal grid
        ggr = hexagonal_BZ_grid()

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
        varphi_rad=-1.5707963267948966
        chi_rad=1.5707963267948966
        ellip=0
        Efield=fieldcallable(tt,I_W__cm2=I_W__cm2,lambda0_nm=lambda0*1e9, env=Field.env_sin2, env_parameters=env_params, varphi_rad=varphi_rad, chi_rad=chi_rad,ellip=ellip)

        TBev=TBevolution.TBevolution_Bloch(TBcr, Efield)

        # --- puntos de prueba: genéricos, lejos de simetrías donde grad=0 ---
        a = 1.42
        kx = np.array([0.13, 0.27, 0.05]) / a
        ky = np.array([0.31, 0.08, 0.19]) / a
        kz = np.array([0.0, 0.0, 0.0])

        dk = 1e-6
        for dim, (dkx, dky, dkz) in enumerate([(dk,0,0), (0,dk,0), (0,0,dk)]):
                t_plus  = TBev.tnm(kx+dkx, ky+dky, kz+dkz)
                t_minus = TBev.tnm(kx-dkx, ky-dky, kz-dkz)
                grad_num = (t_plus - t_minus) / (2*dk)
                grad_ana = TBev.grad_tnm(dim, kx, ky, kz)

                print(f"dim={dim}: grad_num={grad_num}")
                print(f"dim={dim}: grad_ana={grad_ana}")

                if np.max(np.abs(grad_ana)) > 1e-30:
                        err = np.max(np.abs(grad_num - grad_ana)) / np.max(np.abs(grad_ana))
                        print(f"{err=}")
                        assert err < 1e-6, f"dim={dim}: error relativo {err}"