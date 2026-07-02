import numpy as np
import grid as gr
import TBcrystal as cr
from Field import PolarizedHarmonicField
from scipy.constants import hbar, elementary_charge, eV
import logging
import time
from datetime import timedelta
from numba import njit, prange

# logging.basicConfig(
#     filename="graphene_TightBinding.log",
#     filemode="w", 
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s: %(message)s",
# )

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

@njit(parallel=True)
def _t_nm(kx,ky,kz,h_Rnm, deltas, R):
    n_orb=h_Rnm.shape[1]
    n_k=len(kx)
    n_R = R.shape[0]

    t=np.empty((n_orb,n_orb,n_k), dtype=np.complex128)
    for n in range(n_orb):
        for m in range(n_orb):
            tnm_vector=h_Rnm[:,n,m]
            arg_exp = np.zeros((n_R, n_k), dtype=np.complex128)
            
            for iR in range(n_R):
                #R_delta=R[iR]+deltas[m]-deltas[n]
                R_delta=R[iR]
                arg_exp[iR, :] = (
                    1j * 2 * np.pi * kx * R_delta[0] +
                    1j * 2 * np.pi * ky * R_delta[1] +
                    1j * 2 * np.pi * kz * R_delta[2]
                )

            exp=np.exp(arg_exp)
            t[n,m]=np.dot(exp.T, tnm_vector)
    return t

def _r_nm(kx,ky,kz,r_Rnm, R):
    n_orb=r_Rnm.shape[1]
    n_k=len(kx)
    n_R = R.shape[0]

    r=np.empty((n_orb,n_orb,n_k,3), dtype=np.complex128)
    for n in range(n_orb):
        for m in range(n_orb):
            rnm_x_vector=r_Rnm[:,n,m,0]
            rnm_y_vector=r_Rnm[:,n,m,1]
            rnm_z_vector=r_Rnm[:,n,m,2]
            arg_exp= np.zeros((n_R, n_k), dtype=np.complex128)
            
            for iR in range(n_R):
                R_delta=R[iR]
                arg_exp[iR, :] = (
                    1j * 2 * np.pi * kx * R_delta[0] +
                    1j * 2 * np.pi * ky * R_delta[1] +
                    1j * 2 * np.pi * kz * R_delta[2]
                )

            exp=np.exp(arg_exp)
            r[n,m,:,0]=np.dot(exp.T, rnm_x_vector)
            r[n,m,:,1]=np.dot(exp.T, rnm_y_vector)
            r[n,m,:,2]=np.dot(exp.T, rnm_z_vector)
    return r

def _eigen_hermitian(t_nm):
    t_nm = np.moveaxis(t_nm, -1, 0)          # (N_k, N_orb, N_orb)
    energies, eigvecs = np.linalg.eigh(t_nm)
    # energies: (N_k, N_orb), eigvecs: (N_k, N_orb, N_orb)
    energies = np.moveaxis(energies, 0, -1)      # (N_orb, N_k)
    eigvecs = np.moveaxis(eigvecs, 0, -1)        # (N_orb, N_orb, N_k)
    return energies, eigvecs


@njit(parallel=True)
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


#@njit(parallel=True)
def _rk_evolveCB(CB, kx,ky, kz, Ax, Ay, Az, Ex, Ey, Ez, dt,  h_Rnm, r_Rnm, deltas, R, from_it:int, to_it:int):

    itmax=len(Ax)

    to_it=min(to_it, itmax)

    c_qe__hbar=qe/hbar
    c1=-dt*1j/2/hbar

    n_orb=h_Rnm.shape[1]
    n_k=len(kx)
    tnm=np.empty((n_orb,n_orb,n_k), dtype=np.complex128)
    tnm_dt=np.empty((n_orb,n_orb,n_k), dtype=np.complex128)

    for it in range(from_it,to_it):
        norm_At_x=c_qe__hbar*Ax[it]
        norm_At_y=c_qe__hbar*Ay[it]
        norm_At_z=c_qe__hbar*Az[it]
        qFt_x=qe*Ex[it]
        qFt_y=qe*Ey[it]
        qFt_z=qe*Ez[it]
        if it==itmax:
            norm_Atdt_x=c_qe__hbar*(2*Ax[itmax]-Ax[itmax-1]) # extrapolation
            norm_Atdt_y=c_qe__hbar*(2*Ay[itmax]-Ay[itmax-1]) # extrapolation
            norm_Atdt_z=c_qe__hbar*(2*Az[itmax]-Az[itmax-1]) # extrapolation
            qFtdt_x=qe*(2*Ex[itmax]-Ex[itmax-1]) # extrapolation
            qFtdt_y=qe*(2*Ey[itmax]-Ey[itmax-1]) # extrapolation
            qFtdt_z=qe*(2*Ez[itmax]-Ez[itmax-1]) # extrapolation
        else:
            norm_Atdt_x=c_qe__hbar*Ax[it+1]
            norm_Atdt_y=c_qe__hbar*Ay[it+1]
            norm_Atdt_z=c_qe__hbar*Az[it+1]
            qFtdt_x=qe*Ex[it+1]
            qFtdt_y=qe*Ey[it+1]
            qFtdt_z=qe*Ez[it+1]
            
        norm_Atdt2_x=(norm_Atdt_x+norm_At_x)/2
        norm_Atdt2_y=(norm_Atdt_y+norm_At_y)/2
        norm_Atdt2_z=(norm_Atdt_z+norm_At_z)/2

        qFtdt2_x=(qFtdt_x+qFt_x)/2
        qFtdt2_y=(qFtdt_y+qFt_y)/2
        qFtdt2_z=(qFtdt_z+qFt_z)/2

        if it==from_it:
            ktx=kx-norm_At_x
            kty=ky-norm_At_y
            ktz=kz-norm_At_z
            tnm=_t_nm(ktx,kty,ktz, h_Rnm, deltas, R)
            rnm=_r_nm(ktx,kty,ktz, r_Rnm, R)
            ktx=kx-norm_Atdt_x
            kty=ky-norm_Atdt_y
            ktz=kz-norm_Atdt_z
            tnm_dt=_t_nm(ktx,kty,ktz, h_Rnm, deltas, R)
            rnm_dt=_r_nm(ktx,kty,ktz, r_Rnm, R)
            tnm_dt2=(tnm_dt+tnm)/2
            rnm_dt2=(rnm_dt+rnm)/2
        else:
            tnm=tnm_dt
            rnm=rnm_dt
            ktx=kx-norm_Atdt_x
            kty=ky-norm_Atdt_y
            ktz=kz-norm_Atdt_z
            tnm_dt=_t_nm(ktx,kty,ktz, h_Rnm, deltas, R)
            rnm_dt=_r_nm(ktx,kty,ktz, r_Rnm, R)
            tnm_dt2=(tnm_dt+tnm)/2        
            rnm_dt2=(rnm_dt+rnm)/2

        Mnm=tnm-qFt_x*rnm[...,0]-qFt_y*rnm[...,1]-qFt_z*rnm[...,2]
        print(f"{rnm.shape=}, {tnm.shape=}, {Mnm.shape=}")
        K1=c1*(np.einsum('nmi,mi->ni', Mnm, CB, optimize=True))

        Mnm=tnm_dt2-qFtdt2_x*rnm_dt2[...,0]-qFtdt2_y*rnm_dt2[...,1]-qFtdt2_z*rnm_dt2[...,2]
        K2=c1*(np.einsum('nmi,mi->ni', Mnm, CB+K1/2, optimize=True))
        K3=c1*(np.einsum('nmi,mi->ni', Mnm, CB+K2/2, optimize=True))
  
        Mnm=tnm_dt-qFtdt_x*rnm_dt[...,0]-qFtdt_y*rnm_dt[...,1]-qFtdt_z*rnm_dt[...,2]
        K4=c1*(np.einsum('nmi,mi->ni', Mnm, CB+K3, optimize=True))
  
        CB+=K1/6+K2/3+K3/3+K4/6

    return CB

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

        self.R=cr.vector_in_cart(self.crystal.R_vectors, self.crystal.direct_vectors) # coordinate of the WZ cell in cartesian

        self.deltas=self.crystal.deltas
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
    
    def tnm(self,kx,ky,kz):
        return _t_nm(kx, ky, kz, self.h_Rnm, self.deltas, self.R)
    
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

        CM, CP=_rk_evolveCMCP(CM, CP, kx,ky,kz, Ax, Ay, Az, self.Field.dt, self.h_Rnm, self.deltas, self.R, from_it, to_it)
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

class TBevolution_Bloch:

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
        self.r_Rnm=self.crystal.r_Rnm

        self.R=cr.vector_in_cart(self.crystal.R_vectors, self.crystal.direct_vectors) # coordinate of the WZ cell in cartesian

        self.deltas=self.crystal.deltas
        self.num_wann=self.crystal.num_wann
        self.nrpts=self.crystal.nrpts

        kx=self.k[:,0]
        ky=self.k[:,1]
        kz=self.k[:,2]

        # set the initial amplitudes of the Bloch states. 
        # As all the population is in the lower band, the Bloch state aplitudes correspond to the 
        # lower band amplitudes. CB has the coeficients for the orbitals in each column

        _,CB=self.bands(kx,ky,kz)
        self.CB=CB

    def __repr__(self):
        info=f"# {self.__class__.__name__}:  id= {id(self):x} \n"
        info+=f"# \n"
        return info
    
    def bands(self,kx,ky,kz):
        return _eigen_hermitian(self.tnm(kx,ky,kz))
    
    def tnm(self,kx,ky,kz):
        return _t_nm(kx, ky, kz, self.h_Rnm, self.deltas, self.R)
    
    def rnm(self,kx,ky,kz):
        return _r_nm(kx, ky, kz, self.r_Rnm, self.R)
    
    def rk_evolve(self,CB, kx, ky, kz, from_it:int, to_it:int):
        if to_it>len(self.Field.A[:,0])-1:
            to_it=len(self.Field.A[:,0])-1

        # Assume that the parallel direction is in the plane x,z and perp in the direction y
        Ax=self.Field.A[:,0]/self.crystal.reciprocal_lattice_unit*self.Field.s_direction[2]
        Ex=self.Field.E[:,0]/self.crystal.reciprocal_lattice_unit*self.Field.s_direction[2]
        sinus=np.sqrt(1-self.Field.s_direction[2]**2)
        Az=self.Field.A[:,0]/self.crystal.reciprocal_lattice_unit*sinus
        Ez=self.Field.E[:,0]/self.crystal.reciprocal_lattice_unit*sinus

        ## add the axial part
        Ax+=self.Field.A[:,2]/self.crystal.reciprocal_lattice_unit*sinus
        Ex+=self.Field.E[:,2]/self.crystal.reciprocal_lattice_unit*sinus
        Az+=self.Field.A[:,2]/self.crystal.reciprocal_lattice_unit*self.Field.s_direction[2]
        Ez+=self.Field.E[:,2]/self.crystal.reciprocal_lattice_unit*self.Field.s_direction[2]
    
        Ay=self.Field.A[:,1]/self.crystal.reciprocal_lattice_unit
        Ey=self.Field.E[:,1]/self.crystal.reciprocal_lattice_unit

        CB=_rk_evolveCB(CB, kx,ky,kz, Ax, Ay, Az, Ex, Ey, Ez, self.Field.dt, self.h_Rnm, self.r_Rnm, self.deltas, self.R, from_it, to_it)

        return CB

    def rk_dipole_velocity(self, npt:int):
        
        start_time = time.time()

        imax=len(self.Field.t)+1
        istep=imax//npt

        Cvw=self.Cv.copy()
        Ccw=self.Cc.copy()
        Cv_px=self.Cv_px.copy()
        Cc_px=self.Cc_px.copy()
        Cv_mx=self.Cv_mx.copy()
        Cc_mx=self.Cc_mx.copy()
        Cv_py=self.Cv_py.copy()
        Cc_py=self.Cc_py.copy()
        Cv_my=self.Cv_my.copy()
        Cc_my=self.Cc_my.copy()
        dk=epsk*self.crystal.reciprocal_lattice_unit

        time_dip=np.zeros(npt,dtype=np.float64)
        dipole_x=np.zeros(npt,dtype=np.complex128)
        dipole_y=np.zeros(npt,dtype=np.complex128)


        gradCv_x=(Cv_px-Cv_mx)/(2*dk)
        gradCc_x=(Cc_px-Cc_mx)/(2*dk)
        gradCv_y=(Cv_py-Cv_my)/(2*dk)
        gradCc_y=(Cc_py-Cc_my)/(2*dk)

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