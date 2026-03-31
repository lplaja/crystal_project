import numpy as np
import grid as gr
import TBcrystal as cr
from Field import Field
from scipy.constants import hbar, elementary_charge, eV
import logging
import time
from datetime import timedelta
from numba import njit

# logging.basicConfig(
#     filename="graphene_TightBinding.log",
#     filemode="w", 
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s: %(message)s",
# )

logger = logging.getLogger(__name__)

qe=-elementary_charge

epsk=1e-3 # small interval to compute the derivative



@njit
def _fk(kx,ky,a):
    kdota0a1=2*np.pi*(kx*(a[0][0]+a[1][0])+ky*(a[0][1]+a[1][1]))
    kdota0=2*np.pi*(kx*a[0][0]+ky*a[0][1])
    kdota1=2*np.pi*(kx*a[1][0]+ky*a[1][1])
    fk=np.exp(-1j*kdota0a1/3)*(1+np.exp(1j*kdota0)+np.exp(1j*kdota1))
    return fk

@njit
def _Ek(kx,ky,a,gamma0):
    en=gamma0*np.abs(_fk(kx,ky,a))
    return np.column_stack((en,-en))# energy[0] valence, energy[1] conduction
    
@njit
def _phik(kx,ky,a):
    return np.angle(_fk(kx,ky,a))

@njit
def _phikEk(kx,ky,a,gamma0):
    fk=_fk(kx,ky,a)
    en=gamma0*np.abs(fk)
    return np.angle(fk), np.column_stack((en,-en))
    
@njit
def _rk_evolveCMCP(CM, CP, kx,ky, Ax, Ay, dt, a, gamma0, from_it:int, to_it:int):

    itmax=len(Ax)

    to_it=min(to_it, itmax)

    c_qe__hbar=qe/hbar
    c1=-dt*1j/2/hbar

    for it in range(from_it,to_it):
        norm_At_x=c_qe__hbar*Ax[it]
        norm_At_y=c_qe__hbar*Ay[it]
        if it==itmax:
            norm_Atdt_x=c_qe__hbar*(2*Ax[itmax]-Ax[itmax-1]) # extrapolation
            norm_Atdt_y=c_qe__hbar*(2*Ay[itmax]-Ay[itmax-1]) # extrapolation
        else:
            norm_Atdt_x=c_qe__hbar*Ax[it+1]
            norm_Atdt_y=c_qe__hbar*Ay[it+1]
            
        norm_Atdt2_x=(norm_Atdt_x+norm_At_x)/2
        norm_Atdt2_y=(norm_Atdt_y+norm_At_y)/2

        ktx=kx-norm_At_x
        kty=ky-norm_At_y

        g0fk=_fk(ktx,kty,a)*gamma0
        sumE=0
        diffE=-2*g0fk
        K1M=c1*(sumE*CM+diffE*CP)
        K1P=c1*(sumE*CP+np.conjugate(diffE)*CM)

        ktx=kx-norm_Atdt2_x
        kty=ky-norm_Atdt2_y
        g0fk=_fk(ktx,kty,a)*gamma0
        sumE=0
        diffE=-2*g0fk
        K2M=c1*(sumE*(CM+K1M/2)+diffE*(CP+K1P/2))
        K2P=c1*(sumE*(CP+K1P/2)+np.conjugate(diffE)*(CM+K1M/2))
                
        K3M=c1*(sumE*(CM+K2M/2)+diffE*(CP+K2P/2))
        K3P=c1*(sumE*(CP+K2P/2)+np.conjugate(diffE)*(CM+K2M/2))

        ktx=kx-norm_Atdt_x
        kty=ky-norm_Atdt_y
        g0fk=_fk(ktx,kty,a)*gamma0
        sumE=0
        diffE=-2*g0fk
        K4M=c1*(sumE*(CM+K3M)+diffE*(CP+K3P))
        K4P=c1*(sumE*(CP+K3P)+np.conjugate(diffE)*(CM+K3M))

        CM+=K1M/6+K2M/3+K3M/3+K4M/6
        CP+=K1P/6+K2P/3+K3P/3+K4P/6

    return CM, CP

class TB_graphene_CMCP:

    def __init__(self, crystal:cr.crystal, Field:Field):

        if not crystal.grid.cartesian:
            raise ValueError("crystal.grid must be cartesian")

        self.crystal = crystal

        self.Field=Field

        self.k=self.crystal.grid.x[self.crystal.grid.inGrid] # note that k are points from the filtered grid
        self.dV=self.crystal.grid.dV[self.crystal.grid.inGrid]
        self.V=np.sum(self.dV)


        self.gamma0=crystal.parameters['orbitals'][0]['neighbors'][0]['hopping']*eV

        kx=self.k[:,0]
        ky=self.k[:,1]
        sqrt_norm=np.sqrt(2*self.V*self.crystal.reciprocal_lattice_unit**2) # como están definidos CM^2+CP^2=1
        self.CM   =-np.ones(kx.shape[0],dtype=complex)/sqrt_norm
        self.CM_px=-np.ones(kx.shape[0],dtype=complex)/sqrt_norm
        self.CM_mx=-np.ones(kx.shape[0],dtype=complex)/sqrt_norm
        self.CM_py=-np.ones(kx.shape[0],dtype=complex)/sqrt_norm
        self.CM_my=-np.ones(kx.shape[0],dtype=complex)/sqrt_norm

        self.CP   =np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx,ky))/sqrt_norm    
        self.CP_px=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx+epsk,ky))/sqrt_norm
        self.CP_mx=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx-epsk,ky))/sqrt_norm
        self.CP_py=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx, ky+epsk))/sqrt_norm
        self.CP_my=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx, ky-epsk))/sqrt_norm

    def __repr__(self):
        info=f"# {self.__class__.__name__}:  id= {id(self):x} \n"
        info+=f"# \n"
        return info
    
    def fk(self,kx,ky):
        return _fk(kx,ky,self.crystal.direct_vectors)

    def Ek(self,kx,ky):
        return _Ek(kx,ky,self.crystal.direct_vectors,self.gamma0)

    def phik(self,kx,ky):
        return _phik(kx,ky,self.crystal.direct_vectors)
    
    def grad_phik_x(self,kx,ky):
        kxp=kx+epsk
        kxm=kx-epsk
        phikp=self.phik(kxp,ky)
        phikm=self.phik(kxm,ky)
        gradphik=(phikp-phikm)/(2*epsk)/self.crystal.reciprocal_lattice_unit
        return gradphik

    def grad_phik_y(self,kx,ky):
        kyp=ky+epsk
        kym=ky-epsk
        phikp=self.phik(kx,kyp)
        phikm=self.phik(kx, kym)
        gradphik=(phikp-phikm)/(2*epsk)/self.crystal.reciprocal_lattice_unit
        return gradphik
    
    def Dk(self,kx,ky):
        gradphik_x=self.grad_phik_x(kx,ky)
        gradphik_y=self.grad_phik_y(kx,ky)
        if kx.ndim==1:
            return qe/2*np.array([gradphik_x,gradphik_y])
        else:
            return qe/2*np.column_stack((gradphik_x,gradphik_y))
        
    def Cv(self):
        kx=self.k[:,0]
        ky=self.k[:,1]
        exp_phik=np.exp(1j*self.phik(kx,ky))
        return (self.CP*exp_phik-self.CM)/2
    
    def Cc(self):
        kx=self.k[:,0]
        ky=self.k[:,1]
        exp_phik=np.exp(1j*self.phik(kx,ky))
        return (self.CP*exp_phik+self.CM)/2
        
    def rk_evolve(self,CM, CP, kx, ky, from_it:int, to_it:int):
        if to_it>len(self.Field.A[:,0])-1:
            to_it=len(self.Field.A[:,0])-1
        Ax=self.Field.A[:,0]/self.crystal.reciprocal_lattice_unit
        Ay=self.Field.A[:,1]/self.crystal.reciprocal_lattice_unit

        CM, CP=_rk_evolveCMCP(CM, CP, kx,ky, Ax, Ay, self.Field.dt, self.crystal.direct_vectors, 
                              self.gamma0, from_it, to_it)
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

        time_dip=np.zeros(npt, dtype=np.float64)
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

        for it in range(0,imax,istep):
            elapsed = time.time() - start_time
            if it!=0:
                remaining=elapsed/it*(imax-it)
                formatted_elapsed = str(timedelta(seconds=elapsed))
                formatted_remaining = str(timedelta(seconds=remaining))
                logging.info(f"elap. time {formatted_elapsed} \t rem. time {formatted_remaining} step= {it//istep}/{imax//istep}")

            CMw, CPw=self.rk_evolve(CMw, CPw, kx, ky, it, it+istep)
            CM_px, CP_px=self.rk_evolve(CM_px, CP_px, kx+epsk*0, ky, it, it+istep)
            CM_mx, CP_mx=self.rk_evolve(CM_mx, CP_mx, kx-epsk*0, ky, it, it+istep)
            CM_py, CP_py=self.rk_evolve(CM_py, CP_py, kx, ky+epsk, it, it+istep)
            CM_my, CP_my=self.rk_evolve(CM_my, CP_my, kx, ky-epsk, it, it+istep)

            gradCM_x=(CM_px-CM_mx)/(2*dk)
            gradCP_x=(CP_px-CP_mx)/(2*dk)
            gradCM_y=(CM_py-CM_my)/(2*dk)
            gradCP_y=(CP_py-CP_my)/(2*dk)

            time_dip[min(it//istep+1, len(dipole_x)-1)]=self.Field.t[it,0]
            # dipole_x[min(it//istep+1, len(dipole_x)-1)]=qe/2*np.sum(np.conj(CMw)*gradCM_x*self.dV)
            # dipole_y[min(it//istep+1, len(dipole_x)-1)]=qe/2*np.sum(np.conj(CMw)*gradCM_y*self.dV)
            #dipole_x[min(it//istep+1, len(dipole_x)-1)]=qe/2*np.sum(np.conj(CPw)*gradCP_x*self.dV)
            #dipole_y[min(it//istep+1, len(dipole_x)-1)]=qe/2*np.sum(np.conj(CPw)*gradCP_y*self.dV)
            dipole_x[min(it//istep+1, len(dipole_x)-1)]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_x+np.conj(CPw)*gradCP_x)*self.dV)
            dipole_y[min(it//istep+1, len(dipole_x)-1)]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_y+np.conj(CPw)*gradCP_y)*self.dV)

            #dipole_x[min(it//istep+1, len(dipole_x)-1)]=1j*qe/2*np.conj(CMw)*gradCM_x*self.dV


        return time_dip, dipole_x*self.crystal.reciprocal_lattice_unit, dipole_y*self.crystal.reciprocal_lattice_unit
            



