import numpy as np
import grid as gr
import TBcrystal as cr
from Field import PolarizedHarmonicField
from scipy.constants import hbar, elementary_charge, eV
import logging
import time
from datetime import timedelta
from numba import njit, set_num_threads, prange

# logging.basicConfig(
#     filename="graphene_TightBinding.log",
#     filemode="w", 
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s: %(message)s",
# )

set_num_threads(8)

logger = logging.getLogger(__name__)

qe=-elementary_charge

epsk=1e-13 # small interval to compute the derivative


# This function is necessary only with the purpose of testing fuction _t_nm
def _fk(kx,ky,a):
    kdota0a1=2*np.pi*(kx*(a[0][0]+a[1][0])+ky*(a[0][1]+a[1][1]))
    kdota0=2*np.pi*(kx*a[0][0]+ky*a[0][1])
    kdota1=2*np.pi*(kx*a[1][0]+ky*a[1][1])    
    fk=np.exp(-1j*kdota0a1/3)*(1+np.exp(1j*kdota0)+np.exp(1j*kdota1))
    return fk

def precompute_R_delta(R, deltas):
    n_orb = len(deltas)
    n_R = R.shape[0]
    R_delta = np.zeros((n_orb, n_orb, n_R, 3))
    for n in range(n_orb):
        for m in range(n_orb):
            R_delta[n, m] = R + deltas[m] - deltas[n]
    return R_delta

@njit(parallel=True)
def _t_nm(kx, ky, kz, h_Rnm_T, R_delta):
    n_orb = h_Rnm_T.shape[0]
    n_k = len(kx)

    k = np.stack((kx, ky, kz))
    t = np.zeros((n_orb, n_orb, n_k), dtype=np.complex128)

    for n in prange(n_orb):
        for m in range(n_orb):
            tnm_vector = h_Rnm_T[n, m]
            arg_exp = 1j * 2 * np.pi * (R_delta[n, m] @ k)
            t[n, m] = tnm_vector @ np.exp(arg_exp)

    return t


@njit(parallel=False)
def _rk_evolveCMCP(CM, CP, kx, ky, kz, Ax, Ay, Az, dt, h_Rnm_T, R_delta, from_it, to_it):
#def _rk_evolveCMCP(CM, CP, kx,ky, kz, Ax, Ay, Az, dt,  neighbor_hoppings, neighbor_positions, from_it:int, to_it:int, a):

    itmax=len(Ax)

    to_it=min(to_it, itmax)

    c_qe__hbar=qe/hbar
    c1=-dt*1j/2/hbar

    for it in range(from_it,to_it):
        norm_At_x=c_qe__hbar*Ax[it]
        norm_At_y=c_qe__hbar*Ay[it]
        norm_At_z=c_qe__hbar*Az[it]
        if it==itmax:
            norm_Atdt_x=c_qe__hbar*(2*Ax[itmax]-Ax[itmax-1]) # extrapolation
            norm_Atdt_y=c_qe__hbar*(2*Ay[itmax]-Ay[itmax-1]) # extrapolation
            norm_Atdt_z=c_qe__hbar*(2*Az[itmax]-Az[itmax-1]) # extrapolation
        else:
            norm_Atdt_x=c_qe__hbar*Ax[it+1]
            norm_Atdt_y=c_qe__hbar*Ay[it+1]
            norm_Atdt_z=c_qe__hbar*Az[it+1]
            
        norm_Atdt2_x=(norm_Atdt_x+norm_At_x)/2
        norm_Atdt2_y=(norm_Atdt_y+norm_At_y)/2
        norm_Atdt2_z=(norm_Atdt_z+norm_At_z)/2

        ktx=kx-norm_At_x
        kty=ky-norm_At_y
        ktz=kz-norm_At_z
        tnm = _t_nm(ktx, kty, ktz, h_Rnm_T, R_delta)
        # tnm=_t_nm(np.array([ktx[100], ktx[100]]),np.array([kty[100], kty[100]]),np.array([ktz[100], ktz[100]]), neighbor_hoppings, neighbor_positions)
        # print(f'vprint in line: 92 --> {tnm[0,1][0]/eV=}')
        # print(f'vprint in line: 92 --> {type(tnm[0,1][0]/eV)=}')
        # print(f'vprint in line: 93 --> {2.97*_fk(ktx[100],kty[100],a)=}')
        sumE=np.real(tnm[0,0]+tnm[1,1])
        diffE=2*tnm[0,1]
        K1M=c1*(sumE*CM+diffE*CP)
        K1P=c1*(sumE*CP+np.conjugate(diffE)*CM)

        ktx=kx-norm_Atdt2_x
        kty=ky-norm_Atdt2_y
        ktz=kz-norm_Atdt2_z
        tnm = _t_nm(ktx, kty, ktz, h_Rnm_T, R_delta)
        sumE=np.real(tnm[0,0]+tnm[1,1])
        diffE=2*tnm[0,1]
        K2M=c1*(sumE*(CM+K1M/2)+diffE*(CP+K1P/2))
        K2P=c1*(sumE*(CP+K1P/2)+np.conjugate(diffE)*(CM+K1M/2))
                
        K3M=c1*(sumE*(CM+K2M/2)+diffE*(CP+K2P/2))
        K3P=c1*(sumE*(CP+K2P/2)+np.conjugate(diffE)*(CM+K2M/2))

        ktx=kx-norm_Atdt_x
        kty=ky-norm_Atdt_y
        ktz=kz-norm_Atdt_z
        tnm = _t_nm(ktx, kty, ktz, h_Rnm_T, R_delta)
        sumE=np.real(tnm[0,0]+tnm[1,1])
        diffE=2*tnm[0,1]
        K4M=c1*(sumE*(CM+K3M)+diffE*(CP+K3P))
        K4P=c1*(sumE*(CP+K3P)+np.conjugate(diffE)*(CM+K3M))

        CM+=K1M/6+K2M/3+K3M/3+K4M/6
        CP+=K1P/6+K2P/3+K3P/3+K4P/6

    return CM, CP

class TBevolution_CMCP:

    def __init__(self, crystal:cr.crystal, Field:PolarizedHarmonicField):

        if not crystal.grid.cartesian:
            raise ValueError("crystal.grid must be cartesian")

        self.crystal = crystal

        self.Field=Field

        self.k=self.crystal.grid.x[self.crystal.grid.inGrid] # note that k are points from the filtered grid
        self.dV=self.crystal.grid.dV[self.crystal.grid.inGrid]
        self.V=np.sum(self.dV)

        if self.crystal.grid.ndim == 2:
            if Field.s_direction[0]!= 0 or Field.s_direction[1]!= 0:
                raise ValueError("s_direction must be [0,0,1], orthogonal to the xy plane")
            # Añadir columna de ceros para kz
            kz = np.zeros((self.k.shape[0], 1), dtype=np.float64)
            self.k = np.column_stack([self.k, kz])
        elif self.crystal.grid.ndim == 1:
            if Field.s_direction[0]!= 0:
                raise ValueError("s_direction must be orthogonal to x")
            # Añadir columnas de ceros para kz y ky
            kz = np.zeros((self.k.shape[0], 1), dtype=np.float64)
            ky = np.zeros((self.k.shape[0], 1), dtype=np.float64)
            self.k = np.column_stack([self.k, ky, kz])    

        self.h_Rnm = self.crystal.h_Rnm * eV / self.crystal.deg_weights[:, None, None] # convert to eV and apply degeneracy weights
        self.h_Rnm_T = np.ascontiguousarray(self.h_Rnm.transpose(1, 2, 0))  # (n_orb, n_orb, n_R)
        self.R=cr.vector_in_cart(self.crystal.R_vectors, self.crystal.direct_vectors) # coordinate of the WZ cell in cartesian
        self.deltas=self.crystal.deltas
        self.R_delta = precompute_R_delta(self.R, self.deltas)

        self.num_wann=self.crystal.num_wann
        self.nrpts=self.crystal.nrpts

        kx=self.k[:,0]
        ky=self.k[:,1]
        kz=self.k[:,2]
        sqrt_norm=np.sqrt(2*self.V) # como están definidos CM^2+CP^2=2
        self.CM   =-np.ones(kx.shape[0],dtype=np.complex128)/sqrt_norm
        self.CM_px=-np.ones(kx.shape[0],dtype=np.complex128)/sqrt_norm
        self.CM_mx=-np.ones(kx.shape[0],dtype=np.complex128)/sqrt_norm
        self.CM_py=-np.ones(kx.shape[0],dtype=np.complex128)/sqrt_norm
        self.CM_my=-np.ones(kx.shape[0],dtype=np.complex128)/sqrt_norm

        self.CP   =np.ones(kx.shape[0],dtype=np.complex128)*np.exp(-1j*self.phik(kx,ky,kz))/sqrt_norm    
        self.CP_px=np.ones(kx.shape[0],dtype=np.complex128)*np.exp(-1j*self.phik(kx+epsk,ky,kz))/sqrt_norm
        self.CP_mx=np.ones(kx.shape[0],dtype=np.complex128)*np.exp(-1j*self.phik(kx-epsk,ky,kz))/sqrt_norm
        self.CP_py=np.ones(kx.shape[0],dtype=np.complex128)*np.exp(-1j*self.phik(kx, ky+epsk,kz))/sqrt_norm
        self.CP_my=np.ones(kx.shape[0],dtype=np.complex128)*np.exp(-1j*self.phik(kx, ky-epsk,kz))/sqrt_norm

    def __repr__(self):
        info=f"# {self.__class__.__name__}:  id= {id(self):x} \n"
        info+=f"# \n"
        return info
    
    def tnm(self, kx, ky, kz):
        return _t_nm(kx, ky, kz, self.h_Rnm_T, self.R_delta)    
    def phik(self,kx,ky,kz):
        tnm=self.tnm(kx,ky,kz)
        return np.angle(tnm[0,1])
        
    def Cv(self):
        kx=self.k[:,0]
        ky=self.k[:,1]
        kz=self.k[:,2]
        exp_phik=np.exp(1j*self.phik(kx,ky,kz))
        return (self.CP*exp_phik-self.CM)/2
    
    def Cc(self):
        kx=self.k[:,0]
        ky=self.k[:,1]
        kz=self.k[:,2]
        exp_phik=np.exp(1j*self.phik(kx,ky,kz))
        return (self.CP*exp_phik+self.CM)/2
        
    def rk_evolve(self,CM, CP, kx, ky, kz, from_it:int, to_it:int):
        if to_it>len(self.Field.A[:,0])-1:
            to_it=len(self.Field.A[:,0])-1

        # Assume that the parallel directioon is in the plana x,z and perp in the direction y
        Ax=self.Field.A[:,0]/self.crystal.reciprocal_lattice_unit*self.Field.s_direction[2]
        sinus=np.sqrt(1-self.Field.s_direction[2]**2)
        Az=self.Field.A[:,0]/self.crystal.reciprocal_lattice_unit*sinus

        ## add the axial part
        Ax+=self.Field.A[:,2]/self.crystal.reciprocal_lattice_unit*sinus
        Az+=self.Field.A[:,2]/self.crystal.reciprocal_lattice_unit*self.Field.s_direction[2]
    
        Ay=self.Field.A[:,1]/self.crystal.reciprocal_lattice_unit

        CM, CP = _rk_evolveCMCP(CM, CP, kx, ky, kz, Ax, Ay, Az, self.Field.dt, self.h_Rnm_T, self.R_delta, from_it, to_it)
        # CM, CP=_rk_evolveCMCP(CM, CP, kx,ky,kz, Ax, Ay, Az, self.Field.dt, self.neighbor_hoppings, self.neighbor_positions,
        #                       from_it, to_it, self.crystal.direct_vectors)
        return CM, CP
    
    def rk_dipole(self, npt:int):
        start_time = time.time()

        imax=len(self.Field.t)+1
        istep=imax//npt

        CMw=self.CM.copy()
        CPw=self.CP.copy()
        CM_px=self.CM_px.copy()
        CP_px=self.CP_px.copy()
        CM_mx=self.CM_mx.copy()
        CP_mx=self.CP_mx.copy()
        CM_py=self.CM_py.copy()
        CP_py=self.CP_py.copy()
        CM_my=self.CM_my.copy()
        CP_my=self.CP_my.copy()
        dk=epsk*self.crystal.reciprocal_lattice_unit

        time_dip=np.zeros(npt,dtype=np.float64)
        dipole_x=np.zeros(npt,dtype=np.complex128)
        dipole_y=np.zeros(npt,dtype=np.complex128)


        gradCM_x=(CM_px-CM_mx)/(2*dk)
        gradCP_x=(CP_px-CP_mx)/(2*dk)
        gradCM_y=(CM_py-CM_my)/(2*dk)
        gradCP_y=(CP_py-CP_my)/(2*dk)
        dipole_x[0]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_x+np.conj(CPw)*gradCP_x)*self.dV)
        dipole_y[0]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_y+np.conj(CPw)*gradCP_y)*self.dV)


        c_qe__hbar=qe/hbar

        kx=self.k[:,0]
        ky=self.k[:,1]
        kz=self.k[:,2]

        for it in range(0,imax,istep):
            if it+istep>=imax:
                break
            elapsed = time.time() - start_time
            if it!=0:
                remaining=elapsed/it*(imax-it)
                formatted_elapsed = str(timedelta(seconds=elapsed))
                formatted_remaining = str(timedelta(seconds=remaining))
                logging.info(f"elap. time {formatted_elapsed} \t rem. time {formatted_remaining} step= {it//istep}/{imax//istep}")

            CMw, CPw=self.rk_evolve(CMw, CPw, kx, ky, kz, it, it+istep)
            CM_px, CP_px=self.rk_evolve(CM_px, CP_px, kx+epsk, ky, kz, it, it+istep)
            CM_mx, CP_mx=self.rk_evolve(CM_mx, CP_mx, kx-epsk, ky, kz, it, it+istep)
            CM_py, CP_py=self.rk_evolve(CM_py, CP_py, kx, ky+epsk, kz, it, it+istep)
            CM_my, CP_my=self.rk_evolve(CM_my, CP_my, kx, ky-epsk, kz, it, it+istep)

            gradCM_x=(CM_px-CM_mx)/(2*dk)
            gradCP_x=(CP_px-CP_mx)/(2*dk)
            gradCM_y=(CM_py-CM_my)/(2*dk)
            gradCP_y=(CP_py-CP_my)/(2*dk)

            time_dip[min(it//istep+1, len(dipole_x)-1)]=self.Field.t[it+istep,0]
            dipole_x[min(it//istep+1, len(dipole_x)-1)]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_x+np.conj(CPw)*gradCP_x)*self.dV)
            dipole_y[min(it//istep+1, len(dipole_x)-1)]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_y+np.conj(CPw)*gradCP_y)*self.dV)
        
        return time_dip, dipole_x, dipole_y
            