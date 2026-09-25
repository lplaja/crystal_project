import numpy as np
import grid as gr
import TBcrystal as cr
from Field import PulsedField
from scipy.constants import hbar, elementary_charge, eV
import logging
import time
from datetime import timedelta
from numba import njit, prange
import warnings

# logging.basicConfig(
#     filename="graphene_TightBinding.log",
#     filemode="w",
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s: %(message)s",
# )

logger = logging.getLogger(__name__)

qe=-elementary_charge

epsk=1e-13 # small interval to compute the derivative

def polarization_frame(s_direction):
    """Base ortonormal dextrogira (p, q, s) del campo.
    s = direccion de propagacion (normalizada).
    p = eje x proyectado perpendicular a s (normalizado)  -> componente 'paralela'.
    q = s x p                                              -> componente 'perpendicular'.
    Las columnas de Field.E y Field.A son (paralela, perpendicular, axial) = (p, q, s)."""
    s = np.asarray(s_direction, dtype=float)
    s = s/np.linalg.norm(s)
    p = np.array([1.0, 0.0, 0.0]) - s[0]*s
    norm_p = np.linalg.norm(p)
    if norm_p < 1e-12:
        raise ValueError("s_direction paralela a x: no se puede definir la direccion paralela p")
    p = p/norm_p
    q = np.cross(s, p)
    return p, q, s

# This function is necessary only with the purpose of testing fuction _t_nm
def _fk(kx,ky,a):
    kdota0a1=2*np.pi*(kx*(a[0][0]+a[1][0])+ky*(a[0][1]+a[1][1]))
    kdota0=2*np.pi*(kx*a[0][0]+ky*a[0][1])
    kdota1=2*np.pi*(kx*a[1][0]+ky*a[1][1])
    fk=np.exp(-1j*kdota0a1/3)*(1+np.exp(1j*kdota0)+np.exp(1j*kdota1))
    return fk

#@njit(parallel=False)
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

@njit(parallel=False)
def _grad_t_nm(dim,kx,ky,kz,h_Rnm, deltas, R):  # coputes the gradient in the direction dim
    n_orb=h_Rnm.shape[1]
    n_k=len(kx)
    n_R = R.shape[0]

    grad_t=np.empty((n_orb,n_orb,n_k), dtype=np.complex128)
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
            grad_t[n,m]=1j*2*np.pi*np.dot(exp.T, R[:,dim]*tnm_vector)
    return grad_t

@njit(parallel=False)
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

#@njit(parallel=True)
def _eigen_hermitian(t_nm):

    t_nm = np.moveaxis(t_nm, -1, 0)          # (N_k, N_orb, N_orb)
    energies, eigvecs = np.linalg.eigh(t_nm)
    # energies: (N_k, N_orb), eigvecs: (N_k, N_orb, N_orb)
    energies = np.moveaxis(energies, 0, -1)      # (N_orb, N_k)
    eigvecs = np.moveaxis(eigvecs, 0, -1)        # (N_orb, N_orb, N_k)
    return energies, eigvecs


#@njit(parallel=True)
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


#@njit(parallel=False)
def _rk_evolveCB(CB, kx,ky, kz, Ax, Ay, Az, Ex, Ey, Ez, dt,  h_Rnm, r_Rnm, deltas, R, from_it:int, to_it:int):

    itmax=len(Ax)

    to_it=min(to_it, itmax)

    c_qe__hbar=qe/hbar
    c1=-dt*1j/hbar

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
        K1=c1*(np.einsum('nmi,mi->ni', Mnm, CB, optimize=True))

        Mnm=tnm_dt2-qFtdt2_x*rnm_dt2[...,0]-qFtdt2_y*rnm_dt2[...,1]-qFtdt2_z*rnm_dt2[...,2]
        K2=c1*(np.einsum('nmi,mi->ni', Mnm, CB+K1/2, optimize=True))
        K3=c1*(np.einsum('nmi,mi->ni', Mnm, CB+K2/2, optimize=True))

        Mnm=tnm_dt-qFtdt_x*rnm_dt[...,0]-qFtdt_y*rnm_dt[...,1]-qFtdt_z*rnm_dt[...,2]
        K4=c1*(np.einsum('nmi,mi->ni', Mnm, CB+K3, optimize=True))

        CB+=K1/6+K2/3+K3/3+K4/6

    return CB

class TBevolution_CMCP:

    def __init__(self, crystal:cr.crystal, Field:PulsedField):

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

            time_dip[min(it//istep+1, len(dipole_x)-1)]=self.Field.t[it+istep]
            dipole_x[min(it//istep+1, len(dipole_x)-1)]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_x+np.conj(CPw)*gradCP_x)*self.dV)
            dipole_y[min(it//istep+1, len(dipole_x)-1)]=1j*qe/2*np.sum((np.conj(CMw)*gradCM_y+np.conj(CPw)*gradCP_y)*self.dV)

        return time_dip, dipole_x, dipole_y

class TBevolution_Bloch:

    def __init__(self, crystal:cr.crystal, Field:PulsedField, spin_degeneracy:int=2):

        if not crystal.grid.cartesian:
            raise ValueError("crystal.grid must be cartesian")

        self.crystal = crystal

        self.Field=Field

        self.k=self.crystal.grid.x[self.crystal.grid.inGrid] # note that k are points from the filtered grid
        self.dV=self.crystal.grid.dV[self.crystal.grid.inGrid]
        self.V=np.sum(self.dV)

        # Size of the unit cell (length, area or volume, in A^dim) spanned by the first dim direct vectors:
        # sqrt of the Gram determinant, valid for dim = 1, 2, 3 (|a1|, |a1 x a2|, |a1.(a2 x a3)|).
        # A k grid covering exactly one Brillouin zone has sum_k dV = 1/cell_size (k without 2 pi).
        self.spin_degeneracy = spin_degeneracy
        dim = self.crystal.grid.ndim
        a = np.asarray(self.crystal.direct_vectors, dtype=float)[:dim]
        self.cell_size = np.sqrt(np.linalg.det(a @ a.T))
        n_BZ = self.V*self.cell_size
        if abs(n_BZ - 1) > 0.1:
            warnings.warn(f"The k grid covers {n_BZ:.2f} Brillouin zones (sum dV = {self.V:.4g} 1/A^{dim}, "
                          f"unit cell = {self.cell_size:.4g} A^{dim}): the dipole velocity is multiplied by that factor.")

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
        self.r_Rnm=self.crystal.r_Rnm* self.crystal.direct_lattice_unit               # longitud: Å → m

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
        self.CB=CB[:,0,:]  # banda de menor energia (eigh ordena ascendente): toda la poblacion en la banda de valencia

    def __repr__(self):
        info=f"# {self.__class__.__name__}:  id= {id(self):x} \n"
        info+=f"# \n"
        return info

    def bands(self,kx,ky,kz):
        return _eigen_hermitian(self.tnm(kx,ky,kz))

    def tnm(self,kx,ky,kz):
        return _t_nm(kx, ky, kz, self.h_Rnm, self.deltas, self.R)

    def grad_tnm(self,dim,kx,ky,kz):
        return _grad_t_nm(dim,kx,ky,kz,self.h_Rnm, self.deltas, self.R)# coputes the gradient in the direction dim

    def rnm(self,kx,ky,kz):
        return _r_nm(kx, ky, kz, self.r_Rnm, self.R)

    def _field_components(self):
        """Componentes cartesianas (x,y,z) del potencial vector A [dividido por
        reciprocal_lattice_unit] y del campo electrico E [SI], en todos los instantes."""
        if self.Field.A.ndim != 2 or self.Field.A.shape[1] != 3:
            raise ValueError("Field.A y Field.E deben tener 3 columnas (paralela, perp., axial)")
        p, q, s = polarization_frame(self.Field.s_direction)

        def to_cartesian(F):                   # F: (n_t, 3) = (paralela, perp., axial)
            return F[:, 0, None]*p + F[:, 1, None]*q + F[:, 2, None]*s

        A = to_cartesian(self.Field.A)/self.crystal.reciprocal_lattice_unit
        E = to_cartesian(self.Field.E)
        return A[:, 0], A[:, 1], A[:, 2], E[:, 0], E[:, 1], E[:, 2]

    def rk_evolve(self,CB, kx, ky, kz, from_it:int, to_it:int):
        if to_it>len(self.Field.A[:,0])-1:
            to_it=len(self.Field.A[:,0])-1

        Ax, Ay, Az, Ex, Ey, Ez = self._field_components()

        CB=_rk_evolveCB(CB, kx,ky,kz, Ax, Ay, Az, Ex, Ey, Ez, self.Field.dt, self.h_Rnm, self.r_Rnm, self.deltas, self.R, from_it, to_it)

        return CB

    def _kappa(self, it):
        """kappa(t) = k - (qe/hbar) A(t), en el indice de tiempo 'it'."""
        kx, ky, kz = self.k[:, 0], self.k[:, 1], self.k[:, 2]
        Ax, Ay, Az, _, _, _ = self._field_components()
        c_qe__hbar = qe/hbar
        return kx - c_qe__hbar*Ax[it], ky - c_qe__hbar*Ay[it], kz - c_qe__hbar*Az[it]

    def _velocity_k(self, CB, kxt, kyt, kzt):
        """Velocity of the electron in the state of every k point, shape (n_k, 3), in m/s:

            v_k = < C_k | (1/hbar) dH/dK + (i/hbar) [H, r] | C_k >        evaluated at kappa

        H(kappa) = sum_R h_R exp(2 pi i kappa.R)   (orbital basis, J)
        r(kappa) = sum_R r_R exp(2 pi i kappa.R)   (Wannier position matrix, m)
        kappa = (kxt, kyt, kzt) = k - (q/hbar) A(t), given by the caller, in 1/A WITHOUT 2 pi;
        K = 2 pi kappa is the wavevector in 1/m: dH/dK = grad_tnm / reciprocal_lattice_unit (J m).
        The two terms come from the position operator in the Bloch representation, x = i d/dK + r(K):
        group velocity (intraband) and commutator with the Wannier position matrix (interband).
        C_k = CB[:, k] are the orbital amplitudes, normalized to 1: velocity of ONE electron.
        No dV weight, no spin, no charge."""
        h = self.tnm(kxt, kyt, kzt)
        r = self.rnm(kxt, kyt, kzt)
        gradh = np.stack([self.grad_tnm(dim, kxt, kyt, kzt) for dim in range(3)], axis=-1)
        gradh = gradh / self.crystal.reciprocal_lattice_unit
        commutator = np.einsum('mlk,lnka->mnka', h, r) - np.einsum('mlka,lnk->mnka', r, h)
        bracket = commutator - 1j*gradh
        per_k = np.einsum('mk,mnka,nk->ka', np.conj(CB), bracket, CB)
        return 1j/hbar * per_k

    def _velocity(self, CB, kxt, kyt, kzt):
        """Brillouin-zone sum of the electron velocity, shape (3,), in A^-dim m/s:

            V = sum_k dV_k v_k          (v_k from _velocity_k, in m/s)

        dV_k is the k-grid cell in 1/A^dim, with k WITHOUT 2 pi (k = K/2pi), so that
        sum_k dV_k ~ integral d^dK/(2 pi)^dim (number of states per A^dim, per spin) and, for a grid
        covering one Brillouin zone, sum_k dV_k = 1/cell_size. V is the particle current density of the
        band per spin. No spin, no charge."""
        v_k = self._velocity_k(CB, kxt, kyt, kzt)
        return np.einsum('ka,k->a', v_k, self.dV)

    def rk_dipole_velocity(self, npt: int):
        """Evolve the Bloch amplitudes (RK4) and sample the dipole velocity per unit cell at npt times:

            vd(t) = g_s q cell_size sum_k dV_k v_k(t)       in C m/s

        v_k(t): velocity of the electron in state k at kappa(t) (see _velocity_k), in m/s
        dV_k:   k-grid cell in 1/A^dim (k without 2 pi); cell_size: unit cell in A^dim, so that
                cell_size * sum_k dV_k = 1 when the grid covers one Brillouin zone (then vd is the charge
                times the summed velocity of the g_s electrons of the band in one unit cell)
        q = -e, g_s = spin_degeneracy (2 without spin-orbit)
        Current density: j = vd / (cell_size * 1e-10**dim)  in A (1D), A/m (2D), A/m^2 (3D).

        The dipole velocity is sampled every istep = len(t)//npt time steps: use len(t) multiple of npt.
        Returns t (s) and the components vdx, vdy, vdz (complex arrays; the imaginary part is numerical noise).
        """
        imax = len(self.Field.t)
        istep = imax//npt

        CBw = self.CB.copy()
        kx, ky, kz = self.k[:, 0], self.k[:, 1], self.k[:, 2]

        time_dip = np.zeros(npt, dtype=np.float64)
        v = np.zeros((npt, 3), dtype=np.complex128)  # v[:,0]=vx, v[:,1]=vy, v[:,2]=vz

        time_dip[0] = self.Field.t[0]
        v[0] = self._velocity(CBw, *self._kappa(0))

        for it in range(0, imax, istep):
            if it+istep >= imax:
                break

            CBw = self.rk_evolve(CBw, kx, ky, kz, it, it+istep)

            idx = min(it//istep+1, npt-1)
            time_dip[idx] = self.Field.t[it+istep]
            v[idx] = self._velocity(CBw, *self._kappa(it+istep))

        vd = self.spin_degeneracy*qe*self.cell_size*v   # dipole velocity per unit cell, C m/s
        return time_dip, vd[:, 0], vd[:, 1], vd[:, 2]
