"""
test_field_components.py

Tests de polarization_frame y TBevolution_Bloch._field_components / _kappa
(rotacion de las componentes (paralela, perpendicular, axial) del campo a cartesianas).
No necesitan cristal ni malla: se llama al metodo con un "self" simulado.

Ejecutar:  pytest -s test_field_components.py     (o python3 test_field_components.py)
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")

from types import SimpleNamespace, MethodType
import numpy as np
from scipy.constants import hbar, elementary_charge
import TBevolution as TBe

RLU = 2*np.pi*1e10
qe = -elementary_charge


def fake_self(A, E, s, k=None):
    """Objeto minimo con lo que usan _field_components y _kappa (sin cristal ni malla)."""
    if k is None:
        k = np.zeros((3, 3))
    fs = SimpleNamespace(Field=SimpleNamespace(A=A, E=E, s_direction=s),
                         crystal=SimpleNamespace(reciprocal_lattice_unit=RLU), k=k)
    fs._field_components = MethodType(TBe.TBevolution_Bloch._field_components, fs)
    return fs


def s_aleatorias(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=3) for _ in range(n)]


def test_base_ortonormal_dextrogira():
    for s in s_aleatorias():
        if abs(s[1]**2 + s[2]**2) < 1e-3*np.linalg.norm(s)**2:
            continue
        p, q, sn = TBe.polarization_frame(s)
        M = np.array([p, q, sn])
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-12), "base no ortonormal"
        assert np.allclose(np.cross(p, q), sn, atol=1e-12), "base no dextrogira"
        assert np.allclose(sn, s/np.linalg.norm(s))


def test_base_caso_defecto_z():
    p, q, s = TBe.polarization_frame([0, 0, 1])
    assert np.allclose(p, [1, 0, 0]) and np.allclose(q, [0, 1, 0]) and np.allclose(s, [0, 0, 1])


def test_base_plano_xz_es_rotacion_alrededor_de_y():
    for alpha in (0.3, -0.7, 1.2):
        p, q, s = TBe.polarization_frame([np.sin(alpha), 0, np.cos(alpha)])
        assert np.allclose(p, [np.cos(alpha), 0, -np.sin(alpha)], atol=1e-12)
        assert np.allclose(q, [0, 1, 0], atol=1e-12)


def test_base_1D_s_perpendicular_a_x():
    beta = 0.4
    p, q, s = TBe.polarization_frame([0, np.sin(beta), np.cos(beta)])
    assert np.allclose(p, [1, 0, 0])                 # paralela a lo largo de la cadena


def test_base_s_paralela_a_x_da_error():
    try:
        TBe.polarization_frame([1, 0, 0])
    except ValueError:
        return
    raise AssertionError("debia dar ValueError")


def test_field_components_caso_defecto_reproduce_lo_anterior():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(50, 3)); E = rng.normal(size=(50, 3))
    Ax, Ay, Az, Ex, Ey, Ez = fake_self(A, E, [0, 0, 1])._field_components()
    assert np.allclose(Ax, A[:, 0]/RLU) and np.allclose(Ay, A[:, 1]/RLU) and np.allclose(Az, A[:, 2]/RLU)
    assert np.allclose(Ex, E[:, 0]) and np.allclose(Ey, E[:, 1]) and np.allclose(Ez, E[:, 2])


def test_field_components_conserva_norma_y_no_modifica_el_campo():
    rng = np.random.default_rng(2)
    for s in s_aleatorias(10, seed=3):
        A = rng.normal(size=(40, 3)); E = rng.normal(size=(40, 3))
        A0, E0 = A.copy(), E.copy()
        Ax, Ay, Az, Ex, Ey, Ez = fake_self(A, E, s)._field_components()
        assert np.allclose(Ax**2 + Ay**2 + Az**2, np.sum(A0**2, axis=1)/RLU**2)   # rotacion: |A| igual
        assert np.allclose(Ex**2 + Ey**2 + Ez**2, np.sum(E0**2, axis=1))
        assert np.array_equal(A, A0) and np.array_equal(E, E0), "se ha modificado Field.A o Field.E"


def test_field_components_dos_columnas_da_error():
    A = np.zeros((10, 2)); E = np.zeros((10, 2))
    try:
        fake_self(A, E, [0, 0, 1])._field_components()
    except ValueError:
        return
    raise AssertionError("debia dar ValueError")


def test_kappa_usa_las_mismas_componentes():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(30, 3)); E = rng.normal(size=(30, 3))
    s = [0.3, 0.2, 0.9]
    k = rng.normal(size=(5, 3))
    fs = fake_self(A, E, s, k=k)
    kx, ky, kz = TBe.TBevolution_Bloch._kappa(fs, 7)
    Ax, Ay, Az, *_ = fs._field_components()
    assert np.allclose(kx, k[:, 0] - qe/hbar*Ax[7])
    assert np.allclose(ky, k[:, 1] - qe/hbar*Ay[7])
    assert np.allclose(kz, k[:, 2] - qe/hbar*Az[7])


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("OK ", nombre)
