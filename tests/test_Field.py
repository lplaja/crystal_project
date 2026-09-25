"""
test_Field.py -- tests de Field.py (PulsedField, ellipse_to_jones, jones_to_ellipse).

Convenio (transparencias de clase):
    E = A e_sigma e^{-i(w t + varphi)},  e_sigma = cos(phi) e_par + e^{-i delta_varphi} sin(phi) e_perp
    chi = inclinación del eje mayor respecto a e_par, en (-pi/2, pi/2]
    eps = tan(psi) = ±b/a, con eps > 0 dextrógira: el campo gira de e_par hacia -e_perp
Columnas de E y A: (paralela, perpendicular, axial).

La elipse se MIDE sobre la trayectoria E(t), sin usar las fórmulas de Field.py:
  - eje mayor e inclinación: autovectores de la matriz de covarianza <E_i E_j> en ciclos enteros
  - semiejes: a^2 = 2 lambda_max, b^2 = 2 lambda_min
  - sentido de giro: signo de <E_par dE_perp/dt - E_perp dE_par/dt>

Ejecutar desde la raíz del repo:   pytest -q tests/test_Field.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import grid as gr
import Field

LAMBDA_NM = 800.0
I_W_CM2 = 1e13
N_PER = 4                      # periodos enteros
PTS_PER = 1000                 # puntos por periodo
T0 = Field.lambda_to_T(LAMBDA_NM*1e-9)
E0 = Field.I_to_E(I_W_CM2*1e4)


def tgrid(n_per=N_PER):
    return gr.UniformCartesianGrid(1, limits=[0, n_per*T0], nptx=n_per*PTS_PER)


def campo(**kw):
    kw.setdefault('I_W__cm2', I_W_CM2)
    kw.setdefault('lambda0_nm', LAMBDA_NM)
    return Field.PulsedField(tgrid(), **kw)


def elipse_medida(E):
    """(chi, eps, a, b) de la trayectoria transversal E[:, :2], sin usar Field.py."""
    X = E[:, :2]
    C = X.T @ X/len(X)                                    # <E_i E_j>, exacto en ciclos enteros
    lam, vec = np.linalg.eigh(C)                          # orden ascendente
    a, b = np.sqrt(2*lam[1]), np.sqrt(2*max(lam[0], 0.0))
    v = vec[:, 1]                                         # eje mayor
    chi = np.arctan2(v[1], v[0])
    chi = np.pi/2 - np.mod(np.pi/2 - chi, np.pi)          # (-pi/2, pi/2]
    dX = np.gradient(X, axis=0)
    L = np.mean(X[:, 0]*dX[:, 1] - X[:, 1]*dX[:, 0])      # > 0: gira de e_par a +e_perp
    eps = -np.sign(L)*b/a if b/a > 1e-7 else 0.0          # dextrógira (eps>0) <=> L < 0; lineal: b ~ sqrt(redondeo)
    return chi, eps, a, b


def mismo_chi(c1, c2, tol):
    """Compara inclinaciones módulo pi."""
    d = np.mod(c1 - c2 + np.pi/2, np.pi) - np.pi/2
    return abs(d) < tol


# ------------------------------------------------------------------ forma y tipos
@pytest.mark.parametrize("ft", ['E', 'A'])
def test_forma_y_tipo(ft):
    f = campo(chi_rad=0.3, ellip=0.4, theta_rad=0.2, field_type=ft)
    n = N_PER*PTS_PER
    assert f.E.shape == (n, 3) and f.A.shape == (n, 3)
    assert f.E.dtype == np.float64 and f.A.dtype == np.float64
    assert f.t.shape == (n,)


# ------------------------------------------------------------------ comprobaciones de entrada
def test_field_type_invalido():
    with pytest.raises(ValueError):
        campo(field_type='e')


@pytest.mark.parametrize("ellip", [1.0001, -1.5])
def test_ellip_fuera_de_rango(ellip):
    with pytest.raises(ValueError):
        campo(ellip=ellip)


def test_intensidad_negativa_o_lambda_no_positiva():
    with pytest.raises(ValueError):
        campo(I_W__cm2=-1.0)
    with pytest.raises(ValueError):
        campo(lambda0_nm=0.0)


def test_intensidad_cero_da_campo_nulo():
    f = campo(I_W__cm2=0.0, chi_rad=0.5, ellip=0.3)
    assert np.all(f.E == 0) and np.all(f.A == 0)


# ------------------------------------------------------------------ ellipse_to_jones / jones_to_ellipse
def test_ida_y_vuelta_elipse_jones():
    for chi in np.linspace(-np.pi/2, np.pi/2, 91)[1:]:        # (-pi/2, pi/2]
        for eps in np.linspace(-1, 1, 21):
            c, e = Field.jones_to_ellipse(*Field.ellipse_to_jones(chi, eps))
            assert abs(e - eps) < 1e-12
            if abs(abs(eps) - 1) > 1e-12:                     # circular: chi no definido
                assert abs(c - chi) < 1e-12


def test_ellipse_to_jones_periodo_pi_en_chi():
    for chi in [-1.2, 0.0, 0.4, np.pi/2]:
        for eps in [-0.6, 0.0, 0.5]:
            p1, d1 = Field.ellipse_to_jones(chi, eps)
            p2, d2 = Field.ellipse_to_jones(chi + np.pi, eps)
            p3, d3 = Field.ellipse_to_jones(chi - 3*np.pi, eps)
            assert np.allclose([p1, d1], [p2, d2], atol=1e-12)
            assert np.allclose([p1, d1], [p3, d3], atol=1e-12)


def test_casos_conocidos_ellipse_to_jones():
    # lineal según e_par, lineal según e_perp, circular dextrógira y levógira
    assert np.allclose(Field.ellipse_to_jones(0.0, 0.0), (0.0, 0.0))
    assert np.allclose(Field.ellipse_to_jones(np.pi/2, 0.0), (np.pi/2, 0.0))
    assert np.allclose(Field.ellipse_to_jones(0.0, 1.0), (np.pi/4, np.pi/2))
    assert np.allclose(Field.ellipse_to_jones(0.0, -1.0), (np.pi/4, -np.pi/2))


# ------------------------------------------------------------------ convenio medido sobre E(t)
CASOS = [(chi, eps) for chi in [-np.pi/2 + 1e-3, -1.0, -np.pi/4, -0.3, 0.0, 0.3, np.pi/4, 1.2, np.pi/2]
         for eps in [-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0]]
CASOS += [(2.5, 0.4), (-4.0, -0.7), (np.pi, 0.3)]           # chi fuera de rango


@pytest.mark.parametrize("chi, eps", CASOS)
def test_convenio_elipse_trazada(chi, eps):
    f = campo(chi_rad=chi, ellip=eps, varphi_rad=0.37)
    chi_m, eps_m, a, b = elipse_medida(f.E)
    assert abs(eps_m - eps) < 1e-7, f"eps pedido {eps}, trazado {eps_m}"
    assert np.isclose(a, E0/np.sqrt(1 + eps**2), rtol=1e-9)
    assert np.isclose(b, E0*abs(eps)/np.sqrt(1 + eps**2), rtol=1e-9, atol=1e-6*E0)
    if abs(abs(eps) - 1) > 1e-9:                                # circular: chi no definido
        assert mismo_chi(chi_m, chi, 1e-9), f"chi pedido {chi}, trazado {chi_m}"
    assert mismo_chi(f.chi, chi, 1e-12) and -np.pi/2 < f.chi <= np.pi/2


def test_dextrogira_gira_de_par_a_menos_perp():
    # circular dextrógira: en t=0 (varphi=0) el campo está en +e_par y a continuación E_perp < 0
    f = campo(chi_rad=0.0, ellip=1.0)
    assert f.E[0, 0] > 0 and abs(f.E[0, 1]) < 1e-12*E0
    assert f.E[1, 1] < 0


# ------------------------------------------------------------------ amplitud, intensidad y theta
@pytest.mark.parametrize("eps", [-1.0, -0.3, 0.0, 0.7])
def test_intensidad_no_depende_de_la_elipse(eps):
    f = campo(chi_rad=0.8, ellip=eps)
    assert np.isclose(np.mean(np.sum(f.E**2, axis=1)), E0**2/2, rtol=1e-9)


def test_theta_reparte_amplitud():
    th = 0.4
    f = campo(theta_rad=th)                                     # lineal según e_par
    assert np.isclose(np.max(np.abs(f.E[:, 2])), E0*np.sin(th), rtol=1e-6)
    assert np.isclose(np.max(np.hypot(f.E[:, 0], f.E[:, 1])), E0*np.cos(th), rtol=1e-6)
    assert np.max(np.abs(campo(theta_rad=0.0).E[:, 2])) == 0.0


def test_varphi_es_la_fase_de_E():
    # sin envolvente, lineal según e_par:  E_par = E0 cos(w t + varphi)
    for vp in [0.0, 0.7, -np.pi/2]:
        f = campo(varphi_rad=vp)
        assert np.allclose(f.E[:, 0], E0*np.cos(f.w0*f.t + vp), atol=1e-9*E0)


# ------------------------------------------------------------------ relación entre E y A
ENV = dict(env=Field.env_sin2, env_parameters={'start': 0, 'end': 1, 'ton': 0.5, 'toff': 0.5})


@pytest.mark.parametrize("ft", ['E', 'A'])
def test_E_es_menos_derivada_de_A(ft):
    f = campo(chi_rad=0.5, ellip=0.4, theta_rad=0.2, field_type=ft, **ENV)
    dA = np.gradient(f.A, f.dt, axis=0)
    err = np.max(np.abs(-dA - f.E))/np.max(np.abs(f.E))
    assert err < 1e-4, err


def test_campo_E_y_campo_A_coinciden_sin_envolvente():
    # sin envolvente ni transitorio, A = Re[A0 e_sigma e^{-i(wt+varphi)}] con A0 = -E0/w
    # da E = -dA/dt = Re[-i E0 e_sigma e^{-i(...)}]: el campo de tipo E con varphi + pi/2.
    fE = campo(chi_rad=0.5, ellip=0.4, varphi_rad=0.3 + np.pi/2, field_type='E')
    fA = campo(chi_rad=0.5, ellip=0.4, varphi_rad=0.3, field_type='A')
    interior = slice(1, -1)                                    # np.gradient es de 1er orden en los bordes
    err = np.max(np.abs(fA.E[interior] - fE.E[interior]))/E0
    assert err < 1e-4, err


def test_repr():
    s = repr(campo(chi_rad=0.3, ellip=0.5, field_type='A', **ENV))
    assert 'field_type=A' in s and 'env_sin2' in s
    assert all(line.startswith('#') for line in s.splitlines())
