import numpy as np
import grid as gr
import TBcrystal as cr
from Field import Field
from scipy.constants import hbar, elementary_charge, eV

from numba import njit


qe=-elementary_charge

epsk=1e-5 # small interval to compute the derivative
epskx=np.array([epsk,0])
epsky=np.array([0,epsk])

@njit
def _fk(k,a):
    fk=np.exp(-1j*2*np.pi*k.dot(a[0]+a[1])/3)*(1+np.exp(1j*2*np.pi*k.dot(a[0]))+np.exp(1j*2*np.pi*k.dot(a[1])))
    return fk

@njit
def _Ek(k,a,gamma0):
    en=gamma0*np.abs(_fk(k,a))
    if k.ndim==1:
        return np.array([en,-en])
    else:
        return np.column_stack((en,-en))# energy[0] valence, energy[1] conduction
    
@njit
def _phik(k,a):
    return np.angle(_fk(k,a))
    
@njit
def _rk_evolveCMCP(CM, CP, k, A, dt, a, gamma0, from_it:int, to_it:int):
    itmax=len(A)
    to_it=min(to_it, itmax)

    for it in range(from_it,to_it):
        norm_At=qe*A[it]/hbar
        if it==itmax:
            norm_Atdt=qe*(2*A[itmax]-A[itmax-1])/hbar # extrapolation
        else:
            norm_Atdt=qe*A[it+1]/hbar
            
        norm_Atdt2=(norm_Atdt+norm_At)/2

        kt=k-norm_At*np.array([1,0])
        Ener=_Ek(kt,a,gamma0)
        Ev=Ener[:,0]
        Ec=Ener[:,1]

        exp_phikt=np.exp(1j*_phik(kt,a))
        K1M=-dt*1j/2/hbar*((Ec+Ev)*CM+(Ec-Ev)*exp_phikt*CP)
        K1P=-dt*1j/2/hbar*((Ec+Ev)*CP+(Ec-Ev)*np.conj(exp_phikt)*CM)

        kt=k-norm_Atdt2*np.array([1,0])
        Ener=_Ek(kt,a,gamma0)
        Ev=Ener[:,0]
        Ec=Ener[:,1]
        exp_phikt=np.exp(1j*_phik(kt,a))
        K2M=-dt*1j/2/hbar*((Ec+Ev)*(CM+K1M/2)+(Ec-Ev)*exp_phikt*(CP+K1P/2))
        K2P=-dt*1j/2/hbar*((Ec+Ev)*(CP+K1P/2)+(Ec-Ev)*np.conj(exp_phikt)*(CM+K1M/2))

        kt=k-norm_Atdt2*np.array([1,0])
        Ener=_Ek(kt,a,gamma0)
        Ev=Ener[:,0]
        Ec=Ener[:,1]                
        exp_phikt=np.exp(1j*_phik(kt,a))
        K3M=-dt*1j/2/hbar*((Ec+Ev)*(CM+K2M/2)+(Ec-Ev)*exp_phikt*(CP+K2P/2))
        K3P=-dt*1j/2/hbar*((Ec+Ev)*(CP+K2P/2)+(Ec-Ev)*np.conj(exp_phikt)*(CM+K2M/2))

        kt=k-norm_Atdt*np.array([1,0])
        Ener=_Ek(kt,a,gamma0)
        Ev=Ener[:,0]
        Ec=Ener[:,1]            
        exp_phikt=np.exp(1j*_phik(kt,a))
        K4M=-dt*1j/2/hbar*((Ec+Ev)*(CM+K3M)+(Ec-Ev)*exp_phikt*(CP+K3P))
        K4P=-dt*1j/2/hbar*((Ec+Ev)*(CP+K3P)+(Ec-Ev)*np.conj(exp_phikt)*(CM+K3M))

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

        self.CM   =-np.ones(self.k.shape[0],dtype=complex)
        self.CM_px=-np.ones(self.k.shape[0],dtype=complex)
        self.CM_mx=-np.ones(self.k.shape[0],dtype=complex)
        self.CM_py=-np.ones(self.k.shape[0],dtype=complex)
        self.CM_my=-np.ones(self.k.shape[0],dtype=complex)

        self.CP   =np.ones(self.k.shape[0],dtype=complex)*np.exp(-1j*self.phik(self.k))    
        self.CP_px=np.ones(self.k.shape[0],dtype=complex)*np.exp(-1j*self.phik(self.k+epskx))
        self.CP_mx=np.ones(self.k.shape[0],dtype=complex)*np.exp(-1j*self.phik(self.k-epskx))
        self.CP_py=np.ones(self.k.shape[0],dtype=complex)*np.exp(-1j*self.phik(self.k+epsky))
        self.CP_my=np.ones(self.k.shape[0],dtype=complex)*np.exp(-1j*self.phik(self.k-epsky))

    # def fk(self,k):
    #     a=self.crystal.direct_vectors
    #     fk=np.exp(-1j*2*np.pi*k.dot(a[0]+a[1])/3)*(1+np.exp(1j*2*np.pi*k.dot(a[0]))+np.exp(1j*2*np.pi*k.dot(a[1])))
    #     return fk

    def fk(self,k):
        return _fk(k,self.crystal.direct_vectors)

    # def Ek(self,k):
    #     en=self.gamma0*np.abs(self.fk(k))
    #     if k.ndim==1:
    #         return np.array([en,-en])
    #     else:
    #         return np.column_stack((en,-en))# energy[0] valence, energy[1] conduction

    def Ek(self,k):
        return _Ek(k,self.crystal.direct_vectors,self.gamma0)

    # def phik(self,k):
    #     phik=np.angle(self.fk(k))
    #     return phik

    # def phik(self,k):
    #     phik=np.angle(self.fk(k))
    #     return phik

    def phik(self,k):
        return _phik(k,self.crystal.direct_vectors)
    
    def grad_phik_x(self,k):
        kp=k+np.array([epsk,0])
        km=k-np.array([epsk,0])
        phikp=self.phik(kp)
        phikm=self.phik(km)
        gradphik=(phikp-phikm)/(2*epsk)/self.crystal.reciprocal_lattice_unit
        return gradphik

    def grad_phik_y(self,k):
        kp=k+np.array([0,epsk])
        km=k-np.array([0,epsk])
        phikp=self.phik(kp)
        phikm=self.phik(km)
        gradphik=(phikp-phikm)/(2*epsk)/self.crystal.reciprocal_lattice_unit
        return gradphik
    
    def Dk(self,k):
        gradphik_x=self.grad_phik_x(k)
        gradphik_y=self.grad_phik_y(k)
        if k.ndim==1:
            return qe/2*np.array([gradphik_x,gradphik_y])
        else:
            return qe/2*np.column_stack((gradphik_x,gradphik_y))
        
    def Cv(self):
        exp_phik=np.exp(1j*self.phik(self.k))
        return (self.CP*exp_phik-self.CM)/2
    
    def Cc(self):
        exp_phik=np.exp(1j*self.phik(self.k))
        return (self.CP*exp_phik+self.CM)/2
        
    def rk_evolveCMCP(self,CM, CP, k, from_it:int, to_it:int):
        if to_it>len(self.Field.A)-1:
            to_it=len(self.Field.A)-1
        CM, CP=_rk_evolveCMCP(CM, CP, k, self.Field.A/self.crystal.reciprocal_lattice_unit, self.Field.dt, self.crystal.direct_vectors, 
                              self.gamma0, from_it, to_it)
        return CM, CP