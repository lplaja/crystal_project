import numpy as np
import grid as gr
import TBcrystal as cr
from Field import Field
from scipy.constants import hbar, elementary_charge, eV

from numba import njit


qe=-elementary_charge

epsk=1e-5 # small interval to compute the derivative

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
def _rk_evolveCMCP(CM, CP, kx,ky, A, dt, a, gamma0, from_it:int, to_it:int):
    itmax=len(A)
    to_it=min(to_it, itmax)

    c_qe__hbar=qe/hbar
    c1=-dt*1j/2/hbar

    for it in range(from_it,to_it):
        norm_At=c_qe__hbar*A[it]
        if it==itmax:
            norm_Atdt=c_qe__hbar*(2*A[itmax]-A[itmax-1]) # extrapolation
        else:
            norm_Atdt=c_qe__hbar*A[it+1]
            
        norm_Atdt2=(norm_Atdt+norm_At)/2

        ktx=kx-norm_At
        kty=ky
        Ener=_Ek(ktx,kty,a,gamma0)
        Ev=Ener[:,0]
        Ec=Ener[:,1]

        exp_phikt=np.exp(1j*_phik(ktx,kty,a))
        K1M=-c1*((Ec+Ev)*CM+(Ec-Ev)*exp_phikt*CP)
        K1P=-c1*((Ec+Ev)*CP+(Ec-Ev)*np.conj(exp_phikt)*CM)

        ktx=kx-norm_Atdt2
        kty=ky
        Ener=_Ek(ktx,kty,a,gamma0)
        Ev=Ener[:,0]
        Ec=Ener[:,1]
        exp_phikt=np.exp(1j*_phik(ktx,kty,a))
        K2M=-c1*((Ec+Ev)*(CM+K1M/2)+(Ec-Ev)*exp_phikt*(CP+K1P/2))
        K2P=-c1*((Ec+Ev)*(CP+K1P/2)+(Ec-Ev)*np.conj(exp_phikt)*(CM+K1M/2))
                
        K3M=-c1*((Ec+Ev)*(CM+K2M/2)+(Ec-Ev)*exp_phikt*(CP+K2P/2))
        K3P=-c1*((Ec+Ev)*(CP+K2P/2)+(Ec-Ev)*np.conj(exp_phikt)*(CM+K2M/2))

        ktx=kx-norm_Atdt
        kty=ky
        Ener=_Ek(ktx,kty,a,gamma0)
        Ev=Ener[:,0]
        Ec=Ener[:,1]            
        exp_phikt=np.exp(1j*_phik(ktx,kty,a))
        K4M=-c1*((Ec+Ev)*(CM+K3M)+(Ec-Ev)*exp_phikt*(CP+K3P))
        K4P=-c1*((Ec+Ev)*(CP+K3P)+(Ec-Ev)*np.conj(exp_phikt)*(CM+K3M))

        CM+=K1M/6+K2M/3+K3M/3+K4M/6
        CP+=K1P/6+K2P/3+K3P/3+K4P/6

    return CM, CP

class TB_graphene_Oscar:

    def __init__(self, crystal:cr.crystal, Field:Field):

        if not crystal.grid.cartesian:
            raise ValueError("crystal.grid must be cartesian")

        self.crystal = crystal

        self.Field=Field

        self.k=self.crystal.grid.x[self.crystal.grid.inGrid] # note that k are points from the filtered grid
        self.dV=self.crystal.grid.dV[self.crystal.grid.inGrid]

        self.gamma0=crystal.parameters['orbitals'][0]['neighbors'][0]['hopping']*eV

        kx=self.k[:,0]
        ky=self.k[:,1]
        self.CM   =-np.ones(kx.shape[0],dtype=complex)
        self.CM_px=-np.ones(kx.shape[0],dtype=complex)
        self.CM_mx=-np.ones(kx.shape[0],dtype=complex)
        self.CM_py=-np.ones(kx.shape[0],dtype=complex)
        self.CM_my=-np.ones(kx.shape[0],dtype=complex)

        self.CP   =np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx,ky))    
        self.CP_px=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx+epsk,ky))
        self.CP_mx=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx-epsk,ky))
        self.CP_py=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx, ky+epsk))
        self.CP_my=np.ones(kx.shape[0],dtype=complex)*np.exp(-1j*self.phik(kx, ky-epsk))

    # def fk(self,k):
    #     a=self.crystal.direct_vectors
    #     fk=np.exp(-1j*2*np.pi*k.dot(a[0]+a[1])/3)*(1+np.exp(1j*2*np.pi*k.dot(a[0]))+np.exp(1j*2*np.pi*k.dot(a[1])))
    #     return fk

    def fk(self,kx,ky):
        return _fk(kx,ky,self.crystal.direct_vectors)

    # def Ek(self,k):
    #     en=self.gamma0*np.abs(self.fk(k))
    #     if k.ndim==1:
    #         return np.array([en,-en])
    #     else:
    #         return np.column_stack((en,-en))# energy[0] valence, energy[1] conduction

    def Ek(self,kx,ky):
        return _Ek(kx,ky,self.crystal.direct_vectors,self.gamma0)

    # def phik(self,k):
    #     phik=np.angle(self.fk(k))
    #     return phik

    # def phik(self,k):
    #     phik=np.angle(self.fk(k))
    #     return phik

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
        
    def rk_evolveCMCP(self,CM, CP, k, from_it:int, to_it:int):
        if to_it>len(self.Field.A)-1:
            to_it=len(self.Field.A)-1
        kx=k[:,0]
        ky=k[:,1]
        CM, CP=_rk_evolveCMCP(CM, CP, kx,ky, self.Field.A/self.crystal.reciprocal_lattice_unit, self.Field.dt, self.crystal.direct_vectors, 
                              self.gamma0, from_it, to_it)
        return CM, CP