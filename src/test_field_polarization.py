"""
test_field_polarization.py

Tests de las clases de campo polarizado de Field.py:
    polarizedHarmonicPotentialVectorField   (A dado, E = -dA/dt)
    polarizedHarmonicElectricField          (E dado, A = -int E dt)

Comprueban:
  - A y E tienen 3 columnas (paralela, perpendicular, axial), como exige rk_evolve.
  - theta_rad reparte la amplitud: transversal ~ cos(theta), axial ~ sin(theta).
  - La elipticidad funciona: con ellip=1 (circular) el modulo del campo transversal
    es constante; con ellip=0 (lineal) el campo transversal pasa por cero.

Ejecutar:  pytest -s test_field_polarization.py     (o python3 test_field_polarization.py)
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")

import numpy as np
import grid as gr
import Field

LAMBDA_NM = 800.0
T0 = Field.lambda2T(LAMBDA_NM*1e-9)
NPT = 4000                                   # 4 periodos, 1000 puntos por periodo


def tgrid():
    return gr.UniformCartesianGrid(1, limits=[0, 4*T0], nptx=NPT)


def campo_A(**kw):
    return Field.polarizedHarmonicPotentialVectorField(tgrid(), I_W__cm2=1e13, lambda0_nm=LAMBDA_NM, **kw)


def campo_E(**kw):
    return Field.polarizedHarmonicElectricField(tgrid(), I_W__cm2=1e13, lambda0_nm=LAMBDA_NM, **kw)


def cociente_min_max(X):
    """min/max del modulo del campo transversal (columnas 0 y 1)."""
    mod = np.hypot(X[:, 0], X[:, 1])
    return mod.min()/mod.max()


# ------------------------------------------------------------ campo A
def test_A_tres_columnas():
    f = campo_A()
    assert f.A.shape == (NPT, 3) and f.E.shape == (NPT, 3)


def test_A_theta_reparte_amplitud():
    theta = 0.4
    f = campo_A(theta_rad=theta)
    A0 = abs(f.A0)
    assert np.isclose(np.max(np.abs(f.A[:, 2])), A0*np.sin(theta), rtol=1e-4)
    assert np.isclose(np.max(np.hypot(f.A[:, 0], f.A[:, 1])), A0*np.cos(theta), rtol=1e-4)
    assert np.isclose(np.max(np.linalg.norm(f.A, axis=1)), A0, rtol=1e-4)


def test_A_theta_cero_sin_componente_axial():
    f = campo_A(theta_rad=0.0)
    assert np.max(np.abs(f.A[:, 2])) == 0.0


def test_A_circular_modulo_transversal_constante():
    assert cociente_min_max(campo_A(chi_rad=0.0, ellip=1.0).A) > 0.999


def test_A_lineal_transversal_pasa_por_cero():
    assert cociente_min_max(campo_A(chi_rad=0.0, ellip=0.0).A) < 1e-2


# ------------------------------------------------------------ campo E
def test_E_tres_columnas():
    f = campo_E()
    assert f.E.shape == (NPT, 3) and f.A.shape == (NPT, 3)


def test_E_theta_reparte_amplitud():
    theta = 0.4
    f = campo_E(theta_rad=theta)
    assert np.isclose(np.max(np.abs(f.E[:, 2])), f.E0*np.sin(theta), rtol=1e-4)
    assert np.isclose(np.max(np.hypot(f.E[:, 0], f.E[:, 1])), f.E0*np.cos(theta), rtol=1e-4)


def test_E_circular_modulo_transversal_constante():
    assert cociente_min_max(campo_E(chi_rad=0.0, ellip=1.0).E) > 0.999, \
        "polarizedHarmonicElectricField con ellip=1 no da polarizacion circular"


def test_E_lineal_transversal_pasa_por_cero():
    assert cociente_min_max(campo_E(chi_rad=0.0, ellip=0.0).E) < 1e-2


if __name__ == "__main__":
    fallos = 0
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            try:
                f()
                print("OK   ", nombre)
            except AssertionError as e:
                fallos += 1
                print("FALLA", nombre, "->", e)
    print(f"\n{fallos} fallo(s)")