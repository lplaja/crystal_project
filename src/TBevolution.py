"""
TBevolution.py -- time evolution of a tight-binding crystal (Wannier90 model) driven by a laser field.

TBevolution_Bloch(crystal, Field) evolves the orbital amplitudes of the valence band with RK4;
rk_dipole_velocity(npt) returns the dipole velocity per unit cell vd(t), the source of the HHG spectrum.

Units
-----
Two systems with a single frontier; every quantity exists in ONE unit inside the evolver.
  lattice units   k, κ = k - a(t) and a(t): Å⁻¹ WITHOUT 2π;  R: Å.
                  They only enter the phase exp(2πi κ·R) of the lattice Fourier sums.
                  K = RLU·k is the wavevector in m⁻¹, RLU = crystal.reciprocal_lattice_unit = 2π·1e10.
  SI              everything else: H in J, r in m, E in V/m, t in s, v in m/s, vd in C·m/s.
The conversions are done once:
  __init__            h_Rnm: eV -> J;  r_Rnm: Å -> m;  k weights w_k = cell_size·dV_k (dimensionless)
  _field_components   a = (q/ħ)·A/RLU  (A in V·s/m)  and  E in Cartesian axes (V/m)
"""
import numpy as np
import TBcrystal as cr
from Field import PulsedField
from scipy.constants import hbar, elementary_charge, eV
import logging
import time
from datetime import timedelta

logger = logging.getLogger(__name__)

qe=-elementary_charge

def polarization_frame(s_direction):
    """Right-handed orthonormal frame (p, q, s) of the field, in Cartesian coordinates.
    s = propagation direction (normalized).
    p = x axis projected perpendicular to s (normalized)   -> 'parallel' component.
    q = s x p                                              -> 'perpendicular' component.
    The columns of Field.E and Field.A are (parallel, perpendicular, axial) = (p, q, s)."""
    s = np.asarray(s_direction, dtype=float)
    s = s/np.linalg.norm(s)
    p = np.array([1.0, 0.0, 0.0]) - s[0]*s
    norm_p = np.linalg.norm(p)
    if norm_p < 1e-12:
        raise ValueError("s_direction parallel to x: the parallel direction p cannot be defined")
    p = p/norm_p
    q = np.cross(s, p)
    return p, q, s

def _phases(kx, ky, kz, R):
    """exp(2πi k·R) for every k point and lattice vector R, shape (n_k, n_R).
    k in Å⁻¹ WITHOUT 2π, R in Å (Cartesian)."""
    return np.exp(2j*np.pi*(np.outer(kx, R[:, 0]) + np.outer(ky, R[:, 1]) + np.outer(kz, R[:, 2])))


def _fourier(P, R, X_R, a=None):
    """Lattice Fourier sum  X(k - a) = Σ_R exp(2πi (k - a)·R) X_R[R, ...]  for every k.

    P = _phases(k) (n_k, n_R);  X_R: (n_R, n_orb, n_orb) or (n_R, n_orb, n_orb, 3);  a: shift (3,) in Å⁻¹.
    exp(2πi (k - a)·R) = exp(2πi k·R) exp(-2πi a·R): the shift only multiplies X_R by n_R phases,
    so P can be computed once and reused for every a. The sum over R is a single matrix product.
    Returns shape (n_orb, n_orb, n_k) or (n_orb, n_orb, n_k, 3)."""
    n_k, n_R = P.shape
    X = X_R.reshape(n_R, -1)                                   # (n_R, n_orb*n_orb[*3])
    if a is not None:
        X = np.exp(-2j*np.pi*(R @ a))[:, None]*X
    out = (P @ X).reshape((n_k,) + X_R.shape[1:])              # (n_k, n_orb, n_orb[, 3])
    return np.moveaxis(out, 0, 2)                              # (n_orb, n_orb, n_k[, 3])


def _t_nm(kx,ky,kz,h_Rnm, deltas, R):
    """H(k) = Σ_R h_R exp(2πi k·R), shape (n_orb, n_orb, n_k). (deltas unused: Wannier gauge)"""
    return _fourier(_phases(kx, ky, kz, R), R, h_Rnm)


def _grad_t_nm(dim,kx,ky,kz,h_Rnm, deltas, R):
    """∂H/∂k_dim = Σ_R 2πi R_dim h_R exp(2πi k·R), shape (n_orb, n_orb, n_k); k without 2π."""
    return _fourier(_phases(kx, ky, kz, R), R, 2j*np.pi*R[:, dim, None, None]*h_Rnm)


def _r_nm(kx,ky,kz,r_Rnm, R):
    """r(k) = Σ_R r_R exp(2πi k·R), shape (n_orb, n_orb, n_k, 3)."""
    return _fourier(_phases(kx, ky, kz, R), R, r_Rnm)


def _eigen_hermitian(t_nm):
    """Eigenvalues (ascending) and eigenvectors of H(k) at every k: shapes (n_orb, n_k) and (n_orb, n_orb, n_k)."""
    t_nm = np.moveaxis(t_nm, -1, 0)          # (N_k, N_orb, N_orb)
    energies, eigvecs = np.linalg.eigh(t_nm)
    # energies: (N_k, N_orb), eigvecs: (N_k, N_orb, N_orb)
    energies = np.moveaxis(energies, 0, -1)      # (N_orb, N_k)
    eigvecs = np.moveaxis(eigvecs, 0, -1)        # (N_orb, N_orb, N_k)
    return energies, eigvecs


def _rk_evolveCB(CB, P, a, E, dt, h_Rnm, r_Rnm, R, from_it:int, to_it:int):
    """RK4 evolution of the orbital amplitudes CB from time index from_it to to_it.

    P = _phases(k) of the unshifted k grid.  a, E: (n_t, 3) from _field_components (a in Å⁻¹ without
    2π, E in V/m).  H and r at κ = k - a(t) are obtained with _fourier(P, R, ..., a), which only
    rotates the n_R coefficients (no exponential of size n_k per step).
    Matrix used in every RK stage: M = H(κ) - q E·r(κ), in J."""
    to_it = min(to_it, len(a) - 1)                   # the step it -> it+1 needs the field at it+1
    qE = qe*E                                        # (n_t, 3), J/m
    c1 = -dt*1j/hbar

    for it in range(from_it, to_it):
        if it == from_it:                            # H, r at t (later steps reuse those at t+dt)
            tnm = _fourier(P, R, h_Rnm, a[it])
            rnm = _fourier(P, R, r_Rnm, a[it])
        else:
            tnm, rnm = tnm_dt, rnm_dt
        tnm_dt = _fourier(P, R, h_Rnm, a[it+1])      # H, r at t+dt
        rnm_dt = _fourier(P, R, r_Rnm, a[it+1])
        tnm_dt2 = (tnm_dt + tnm)/2                   # H, r and qE at t+dt/2: average of t and t+dt
        rnm_dt2 = (rnm_dt + rnm)/2
        qE_dt2 = (qE[it] + qE[it+1])/2

        Mnm = tnm - rnm @ qE[it]                     # rnm (n, n, n_k, 3) @ (3,) -> (n, n, n_k)
        K1 = c1*np.einsum('nmi,mi->ni', Mnm, CB, optimize=True)

        Mnm = tnm_dt2 - rnm_dt2 @ qE_dt2
        K2 = c1*np.einsum('nmi,mi->ni', Mnm, CB + K1/2, optimize=True)
        K3 = c1*np.einsum('nmi,mi->ni', Mnm, CB + K2/2, optimize=True)

        Mnm = tnm_dt - rnm_dt @ qE[it+1]
        K4 = c1*np.einsum('nmi,mi->ni', Mnm, CB + K3, optimize=True)

        CB += K1/6 + K2/3 + K3/3 + K4/6

    return CB

class TBevolution_Bloch:

    def __init__(self, crystal:cr.crystal, Field:PulsedField, spin_degeneracy:int=2):

        if not crystal.grid.cartesian:
            raise ValueError("crystal.grid must be cartesian")

        self.crystal = crystal

        self.Field=Field

        self.k=self.crystal.grid.x[self.crystal.grid.inGrid] # note that k are points from the filtered grid
        self.dV=self.crystal.grid.dV[self.crystal.grid.inGrid]
        self.V=np.sum(self.dV)

        # Size of the unit cell (length, area or volume, in Åᵈⁱᵐ) spanned by the first dim direct vectors:
        # sqrt of the Gram determinant, valid for dim = 1, 2, 3 (|a1|, |a1 x a2|, |a1·(a2 x a3)|).
        # A k grid covering exactly one Brillouin zone has Σ_k dV = 1/cell_size (k without 2π).
        self.spin_degeneracy = spin_degeneracy
        dim = self.crystal.grid.ndim
        a = np.asarray(self.crystal.direct_vectors, dtype=float)[:dim]
        self.cell_size = np.sqrt(np.linalg.det(a @ a.T))
        self.w = self.dV*self.cell_size                 # dimensionless k weights: sum(w) = 1 for one Brillouin zone
        n_BZ = self.w.sum()                             # number of Brillouin zones covered by the k grid
        if abs(n_BZ - 1) > 0.1:                         # 10 %: discretization of the zone border is ~1-3 %
            raise ValueError(f"the k grid covers {n_BZ:.2f} Brillouin zones (sum dV = {self.V:.4g} 1/A^{dim}, "
                             f"unit cell = {self.cell_size:.4g} A^{dim}): the dipole velocity would be multiplied "
                             f"by that factor. Check the BZ limits and the filter radius against the lattice "
                             f"constant of the crystal file.")

        if self.crystal.grid.ndim == 2:
            if Field.s_direction[0]!= 0 or Field.s_direction[1]!= 0:
                raise ValueError("s_direction must be [0,0,1], orthogonal to the xy plane")
            # add a column of zeros for kz
            kz = np.zeros((self.k.shape[0], 1), dtype=np.float64)
            self.k = np.column_stack([self.k, kz])
        elif self.crystal.grid.ndim == 1:
            if Field.s_direction[0]!= 0:
                raise ValueError("s_direction must be orthogonal to x")
            # add columns of zeros for ky and kz
            kz = np.zeros((self.k.shape[0], 1), dtype=np.float64)
            ky = np.zeros((self.k.shape[0], 1), dtype=np.float64)
            self.k = np.column_stack([self.k, ky, kz])

        self.h_Rnm = self.crystal.h_Rnm * eV / self.crystal.deg_weights[:, None, None] # eV -> J, divided by the Wigner-Seitz degeneracy weights
        self.r_Rnm=self.crystal.r_Rnm* self.crystal.direct_lattice_unit               # Å -> m

        self.R=cr.vector_in_cart(self.crystal.R_vectors, self.crystal.direct_vectors) # lattice vectors R in Cartesian coordinates, Å

        self.deltas=self.crystal.deltas
        self.num_wann=self.crystal.num_wann
        self.nrpts=self.crystal.nrpts

        self.a, self.E_xyz = self._field_components()  # k shift (Å⁻¹ without 2π) and E (V/m), (n_t, 3)

        kx=self.k[:,0]
        ky=self.k[:,1]
        kz=self.k[:,2]

        # initial amplitudes: all the population in the lowest band (valence band).
        # CB[:, k] are the orbital coefficients of the state at k (one column per k point)

        _,CB=self.bands(kx,ky,kz)
        self.CB=CB[:,0,:]  # lowest band: eigh returns the eigenvalues in ascending order

    def __repr__(self):
        info=f"# {self.__class__.__name__}:  id= {id(self):x} \n"
        info+=f"# \n"
        return info

    def bands(self,kx,ky,kz):
        return _eigen_hermitian(self.tnm(kx,ky,kz))

    def tnm(self,kx,ky,kz):
        return _t_nm(kx, ky, kz, self.h_Rnm, self.deltas, self.R)

    def grad_tnm(self,dim,kx,ky,kz):
        return _grad_t_nm(dim,kx,ky,kz,self.h_Rnm, self.deltas, self.R)# ∂H/∂k along direction dim

    def rnm(self,kx,ky,kz):
        return _r_nm(kx, ky, kz, self.r_Rnm, self.R)

    def _field_components(self):
        """Field in Cartesian axes (x, y, z) at every time step, as used by the evolution.

        Returns
            a : (n_t, 3) shift of the crystal momentum, κ(t) = k - a(t), in Å⁻¹ WITHOUT 2π
                (the units of k):  a = (q/ħ) A / RLU,  A in V·s/m, RLU = reciprocal_lattice_unit
                (2π·1e10 m⁻¹ per Å⁻¹: K = RLU·k is the wavevector in m⁻¹)
            E : (n_t, 3) electric field, V/m
        Field.A and Field.E have columns (parallel, perpendicular, axial), along the vectors (p, q, s)
        of polarization_frame(s_direction)."""
        if self.Field.A.ndim != 2 or self.Field.A.shape[1] != 3:
            raise ValueError("Field.A and Field.E must have 3 columns (parallel, perpendicular, axial)")
        frame = np.array(polarization_frame(self.Field.s_direction))    # rows p, q, s
        A = self.Field.A @ frame                                         # A_par p + A_perp q + A_ax s
        E = self.Field.E @ frame
        a = qe/hbar*A/self.crystal.reciprocal_lattice_unit
        return a, E

    def rk_evolve(self, CB, kx, ky, kz, from_it:int, to_it:int):
        P = _phases(kx, ky, kz, self.R)                  # once per call, reused in every time step
        return _rk_evolveCB(CB, P, self.a, self.E_xyz, self.Field.dt, self.h_Rnm, self.r_Rnm, self.R, from_it, to_it)

    def _kappa(self, it):
        """κ(t) = k - a(t) at time index it, in Å⁻¹ without 2π (components kx, ky, kz)."""
        kappa = self.k - self.a[it]
        return kappa[:, 0], kappa[:, 1], kappa[:, 2]

    def _velocity_k(self, CB, kxt, kyt, kzt):
        """Velocity of the electron in the state of every k point, shape (n_k, 3), in m/s:

            v_k = ⟨C_k| (1/ħ) ∂H/∂K + (i/ħ) [H, r] |C_k⟩        evaluated at κ

        H(κ) = Σ_R h_R exp(2πi κ·R)   (orbital basis, J)
        r(κ) = Σ_R r_R exp(2πi κ·R)   (Wannier position matrix, m)
        κ = (kxt, kyt, kzt) = k - a(t), given by the caller (see _kappa), in Å⁻¹ WITHOUT 2π;
        K = RLU·κ is the wavevector in m⁻¹: ∂H/∂K = grad_tnm / RLU (J·m), RLU = reciprocal_lattice_unit.
        The two terms come from the position operator in the Bloch representation, x = i ∂/∂K + r(K):
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
        """Brillouin-zone sum of the electron velocity with dimensionless weights, shape (3,), in m/s:

            V = Σ_k w_k v_k,     w_k = cell_size·dV_k     (v_k from _velocity_k, in m/s)

        dV_k is the k-grid cell in Å⁻ᵈⁱᵐ (k WITHOUT 2π) and cell_size the unit cell in Åᵈⁱᵐ, so that
        Σ_k w_k = 1 when the grid covers one Brillouin zone: V is the summed velocity of the electrons of
        the band in one unit cell, per spin. No spin, no charge."""
        v_k = self._velocity_k(CB, kxt, kyt, kzt)
        return np.einsum('ka,k->a', v_k, self.w)

    def rk_dipole_velocity(self, npt: int):
        """Evolve the Bloch amplitudes (RK4) and sample the dipole velocity per unit cell at npt times:

            vd(t) = g_s q Σ_k w_k v_k(t)       in C·m/s

        v_k(t): velocity of the electron in state k at κ(t) (see _velocity_k), in m/s
        w_k:    dimensionless k weights, cell_size·dV_k, with Σ_k w_k = 1 for one Brillouin zone
                (see _velocity): vd is the charge times the summed velocity of the g_s electrons of the
                band in one unit cell
        q = -e, g_s = spin_degeneracy (2 without spin-orbit)
        Current density: j = vd / (cell_size·1e-10**dim)  in A (1D), A/m (2D), A/m² (3D).

        The dipole velocity is sampled every istep = len(t)/npt time steps: len(t) must be a multiple of npt.
        Returns t (s) and the components vdx, vdy, vdz (complex arrays; the imaginary part is numerical noise).
        """
        imax = len(self.Field.t)
        if imax % npt != 0:
            # otherwise the loop produces more samples than npt and the extra ones overwrite the last slot
            raise ValueError(f"the number of time steps ({imax}) must be a multiple of npt ({npt})")
        istep = imax//npt
        start = time.time()                                  
        log_every = max(1, npt//20)                 # progress in the log every ~5 %

        CBw = self.CB.copy()
        kx, ky, kz = self.k[:, 0], self.k[:, 1], self.k[:, 2]

        time_dip = np.zeros(npt, dtype=np.float64)
        v = np.zeros((npt, 3), dtype=np.complex128)  # v[:,0]=vx, v[:,1]=vy, v[:,2]=vz

        time_dip[0] = self.Field.t[0]
        v[0] = self._velocity(CBw, *self._kappa(0))
        logger.info(f"step 0/{imax} (0 %)   evolution started: {npt} samples, every {istep} time steps, "
                    f"{len(self.k)} k points")

        for it in range(0, imax, istep):
            if it+istep >= imax:
                break
            CBw = self.rk_evolve(CBw, kx, ky, kz, it, it+istep)

            idx = it//istep + 1                           # 1 ... npt-1
            time_dip[idx] = self.Field.t[it+istep]
            v[idx] = self._velocity(CBw, *self._kappa(it+istep))

            if idx % log_every == 0:
                elapsed = time.time() - start
                remaining = elapsed*(imax - it - istep)/(it + istep)
                logger.info(f"step {it+istep}/{imax} ({100*(it+istep)/imax:.0f} %)   "
                                f"elapsed {timedelta(seconds=round(elapsed))}   remaining {timedelta(seconds=round(remaining))}")

        vd = self.spin_degeneracy*qe*v                  # dipole velocity per unit cell, C m/s
        return time_dip, vd[:, 0], vd[:, 1], vd[:, 2]
