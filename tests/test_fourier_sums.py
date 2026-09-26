"""
test_fourier_sums.py -- lattice Fourier sums of TBevolution.py (_phases, _fourier, _t_nm, _r_nm, _grad_t_nm).

Reference: the definitions written directly as a sum over R (einsum over an explicit exp(2 pi i k.R)),
    H(k)      = sum_R h_R exp(2 pi i k.R)
    r(k)      = sum_R r_R exp(2 pi i k.R)
    dH/dk_d   = sum_R 2 pi i R_d h_R exp(2 pi i k.R)
    X(k - a)  = sum_R X_R exp(2 pi i (k - a).R)      (_fourier with a shift: phases P reused)
k in 1/A without 2 pi, R in A.

Run from the repository root:   pytest -q tests/test_fourier_sums.py
"""
import numpy as np
import pytest
import TBevolution as TBe

rng = np.random.default_rng(0)
AVEC = np.array([[2.1304225, 1.23, 0.0], [2.1304225, -1.23, 0.0], [0.0, 0.0, 20.0]])   # graphene, A
RINT = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, -1, 0], [-1, 0, 0], [0, -1, 0], [-1, 1, 0], [0, 0, 1]])
R = RINT @ AVEC
N_ORB, N_K = 3, 500
H_R = rng.normal(size=(len(R), N_ORB, N_ORB)) + 1j*rng.normal(size=(len(R), N_ORB, N_ORB))
R_R = rng.normal(size=(len(R), N_ORB, N_ORB, 3)) + 1j*rng.normal(size=(len(R), N_ORB, N_ORB, 3))
KX, KY, KZ = rng.uniform(-0.5, 0.5, (3, N_K))


def reference(X_R, kx, ky, kz):
    """sum_R exp(2 pi i k.R) X_R, moved to shape (n_orb, n_orb, n_k[, 3])."""
    e = np.exp(2j*np.pi*(np.outer(kx, R[:, 0]) + np.outer(ky, R[:, 1]) + np.outer(kz, R[:, 2])))   # (n_k, n_R)
    return np.moveaxis(np.einsum('kr,r...->k...', e, X_R), 0, 2)


def close(A, B):
    return A.shape == B.shape and np.max(np.abs(A - B)) <= 1e-12*np.max(np.abs(B))


def test_t_nm():
    assert close(TBe._t_nm(KX, KY, KZ, H_R, None, R), reference(H_R, KX, KY, KZ))


def test_r_nm():
    assert close(TBe._r_nm(KX, KY, KZ, R_R, R), reference(R_R, KX, KY, KZ))


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_grad_t_nm(dim):
    X = 2j*np.pi*R[:, dim, None, None]*H_R
    assert close(TBe._grad_t_nm(dim, KX, KY, KZ, H_R, None, R), reference(X, KX, KY, KZ))


@pytest.mark.parametrize("X_R", [H_R, R_R], ids=["h", "r"])
def test_shift_reuses_phases(X_R):
    """_fourier(P(k), a) must equal the sum evaluated directly at k - a."""
    P = TBe._phases(KX, KY, KZ, R)
    for a in (np.array([0.013, -0.021, 0.004]), np.array([0.3, 0.1, -0.2])):
        assert close(TBe._fourier(P, R, X_R, a), reference(X_R, KX - a[0], KY - a[1], KZ - a[2]))


def test_hermitian_model_gives_hermitian_H():
    """With h_{-R} = h_R^dagger, H(k) must be hermitian at every k (checks the sign of the phase)."""
    Rint = RINT[:7]                                   # (0,0,0), three R and their opposites
    Rc = Rint @ AVEC
    h = np.zeros((7, N_ORB, N_ORB), complex)
    h[0] = H_R[0] + H_R[0].conj().T
    for i in (1, 2, 3):
        h[i] = H_R[i]
        h[i + 3] = H_R[i].conj().T
    H = TBe._t_nm(KX, KY, KZ, h, None, Rc)
    assert np.allclose(H, np.conj(np.swapaxes(H, 0, 1)), atol=1e-12)
