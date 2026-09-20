"""
test_conductividad_lineal.py

Prueba 3 de la validacion del grafeno con campo: RESPUESTA LINEAL.

Cadena de comprobaciones (cada eslabon fija el siguiente):

  1. test_sigma_universal_referencia
       La referencia de respuesta lineal (formula de Kubo interbanda del grafeno NN, escrita
       aqui de forma independiente de TBevolution) da la conductividad universal
       sigma_0 = e^2/(4 hbar) a bajas frecuencias (lambda = 4 um, hbar*omega ~ 0.3 eV).
       Integracion exacta: paralelogramo (b1, b2) de una celda del espacio k.
  2. test_malla_hexagonal_normalizacion
       La malla hexagonal de la simulacion (gr.UniformCartesianGrid + polygonfilter, radio |K|,
       nodos simetricos) con la normalizacion  j = g_s q sum_k dV v  da la misma sigma que la
       referencia del paso 1 (pulso de 800 nm). Comprueba dV, el area de la zona y el factor 1e20.
  3. test_evolucion_vs_respuesta_lineal_x / _y
       rk_dipole_velocity con un pulso debil reproduce la respuesta lineal EXACTA sobre los
       MISMOS puntos k (sin error de cuadratura, que en mallas pequenas es de decenas de %).
       Valida evolucion RK4 + acoplo qE.r (con las posiciones tau de los orbitales) + velocidad
       + signo de la corriente, en las dos orientaciones (x = Gamma-M, y = Gamma-K).

Metodo: sigma_eff = int j(t) E(t) dt / int E(t)^2 dt  =  media de Re sigma(omega) ponderada
con |E(omega)|^2. La corriente intrabanda (v_vv(kappa(t))) no contribuye: es par en el tiempo
respecto al centro del pulso y E es impar (pulso sin(omega t) x sin^2, con int E dt = 0).
Se resta j(0) porque, k a k, la velocidad de la banda llena no es nula.

Requiere que A en Field.polarizedHarmonicElectricField sea la integral TRAPEZOIDAL de E
(A = -cumulative_trapezoid(E, dx=dt, initial=0)). Con A = -cumsum(E)*dt hay un desfase de
medio paso entre A y E y la sigma sale ~1 % baja (error O(omega*dt/2)).

Ejecutar:  python3 test_conductividad_lineal.py      (o pytest -s)     Tiempo: ~1.5 min.
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")

from functools import lru_cache
import numpy as np
from scipy.constants import hbar, eV, elementary_charge as e
import TBcrystal as cr
import grid as gr
import Field
import TBevolution

W90_FILE = '/home/lplaja/crystal_project/calculations/Wannier90 data/gr1NN_tb.dat'
SIGMA_0 = e**2/(4*hbar)                 # conductividad universal del grafeno [S]
G_S = 2                                 # espin (los dos valles ya estan en la zona de Brillouin)
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# ---------------------------------------------------------------- montaje
@lru_cache(maxsize=None)
def crystal_referencia():
    """Cristal con una malla minima: solo se usan h_Rnm, R_vectors, direct_vectors y deltas."""
    g = gr.UniformCartesianGrid(2, [[-1, 1], [-1, 1]], [2, 2])
    return cr.crystal.from_W90_TB_file(filename=W90_FILE, grid=g, species_name='graphene', threshold_hopping=0)


def radio_BZ(crystal):
    b1, b2 = np.array(crystal.reciprocal_vectors[0])[:2], np.array(crystal.reciprocal_vectors[1])[:2]
    return np.linalg.norm(2*b1 + b2)/3, abs(b1[0]*b2[1] - b1[1]*b2[0])      # |K| [1/A], A_BZ [1/A^2]


def malla_hexagonal(n):
    """n x n con filtro hexagonal de radio |K|; nodos simetricos respecto a k = 0."""
    R, _ = radio_BZ(crystal_referencia())
    L = 1.1*R
    dx = 2*L/n
    return gr.UniformCartesianGrid(2, [[-L, L], [-L, L]], [n, n], origin=(dx/2, dx/2),
                                   filter=gr.polygonfilter, filter_args={'nsides': 6, 'radius': R})


def campo(direccion, lambda_nm, n_ciclos, nptx, intensidad=1e8):
    """Pulso debil, lineal, E = E0 sin(omega t) sin^2(pi t/T): int E dt = 0.
    direccion 'x' (Gamma-M): chi=0;  'y' (Gamma-K): chi=pi/2."""
    T0 = Field.lambda2T(lambda_nm*1e-9)
    tt = gr.UniformCartesianGrid(1, limits=[0, n_ciclos*T0], nptx=nptx)
    return Field.polarizedHarmonicElectricField(
        tt, I_W__cm2=intensidad, lambda0_nm=lambda_nm, phi_rad=-np.pi/2,
        chi_rad=0.0 if direccion == 'x' else np.pi/2, ellip=0.0,
        env=Field.env_sin2, env_parameters={'start': 0, 'end': 1, 'ton': 0.5, 'toff': 0.5})


# ------------------------------------------- referencia de respuesta lineal (Kubo)
def sigma_lineal(crystal, k, w, E, dt, direccion):
    """sigma_eff [S] = int j E dt / int E^2 dt segun la respuesta lineal interbanda del grafeno NN,
    sobre los puntos k [1/A, sin 2 pi] con pesos w [1/A^2] (sum w = A_BZ).

    Hamiltoniano fisico  h(k) = [[0, F], [F*, 0]],  F(k) = sum_R t_R exp(i k . delta_R),
    delta_R = tau_B - tau_A + R.a  (posiciones del fichero de Wannier90).
    Elemento de velocidad interbanda: |<c|dh/dk|v>| = |Im(dF F*)|/|F|;  E_cv = 2|F|.
        Re sigma(omega) = pi e^2 g_s/omega  sum_k w |v_cv|^2 delta(hbar omega - E_cv)
    y con delta y Parseval:  int j E dt = (e^2 g_s/hbar) sum_k w |v_cv|^2 |E(omega_k)|^2 / omega_k.
    """
    kt = 2*np.pi*k                                                   # 1/A verdadero
    Rc = np.asarray(crystal.R_vectors) @ np.asarray(crystal.direct_vectors)
    delta = (crystal.deltas[1] - crystal.deltas[0]) + Rc              # (nR, 3) [A]
    t_R = crystal.h_Rnm[:, 0, 1].real/crystal.deg_weights             # [eV]
    F = 0
    dF = 0
    for tR, d in zip(t_R, delta):
        if tR == 0:
            continue
        fase = np.exp(1j*(kt[:, 0]*d[0] + kt[:, 1]*d[1]))
        F = F + tR*fase
        dF = dF + tR*1j*d[direccion]*fase                             # dF/dk_direccion [eV A]
    absF = np.abs(F)
    M = np.imag(dF*np.conj(F))/absF                                   # [eV A]
    v2 = (M*eV*1e-10)**2/hbar**2                                      # [(m/s)^2]
    omega_k = 2*absF*eV/hbar                                          # [rad/s]

    Npad = 1 << 18
    E_w = np.fft.rfft(E, Npad)*dt
    omega = 2*np.pi*np.fft.rfftfreq(Npad, dt)
    P = np.interp(omega_k, omega, np.abs(E_w)**2)
    return e**2*G_S/hbar*np.sum(w*1e20*v2*P/omega_k)/(np.sum(E**2)*dt)


def malla_romboidal(crystal, N):
    """Una celda del espacio k (paralelogramo b1, b2), N x N puntos: cuadratura exacta de funciones periodicas."""
    a = np.asarray(crystal.direct_vectors)[:2, :2]
    B = np.linalg.inv(a).T                                            # filas b1, b2 [1/A, sin 2 pi]
    s = (np.arange(N) + 0.5)/N
    S1, S2 = np.meshgrid(s, s, indexing='ij')
    k = S1.ravel()[:, None]*B[0] + S2.ravel()[:, None]*B[1]
    return k, np.full(len(k), abs(np.linalg.det(B))/N**2)


# ---------------------------------------------------------------- tests
def test_sigma_universal_referencia():
    c = crystal_referencia()
    k, w = malla_romboidal(c, 1000)
    f = campo('x', 4000.0, 2, 4000)
    print(f"\n[1] referencia de Kubo, lambda = 4 um (hbar*omega ~ 0.3 eV)")
    s = {d: sigma_lineal(c, k, w, f.E[:, 0], f.dt, i)/SIGMA_0 for i, d in enumerate('xy')}
    print(f"    sigma_xx/sigma_0 = {s['x']:.5f}    sigma_yy/sigma_0 = {s['y']:.5f}   (TB: 1 + O((hbar omega/t)^2))")
    assert abs(s['x'] - 1) < 5e-3 and abs(s['y'] - 1) < 5e-3, "la referencia no da sigma_0 = e^2/(4 hbar)"
    assert abs(s['x'] - s['y']) < 1e-3, "sigma_xx != sigma_yy: falta isotropia"


def test_malla_hexagonal_normalizacion():
    c = crystal_referencia()
    _, A_BZ = radio_BZ(c)
    f = campo('x', 800.0, 4, 4000)
    k_ref, w_ref = malla_romboidal(c, 1000)
    ref = sigma_lineal(c, k_ref, w_ref, f.E[:, 0], f.dt, 0)
    g = malla_hexagonal(800)
    k, dV = g.x[g.inGrid], g.dV[g.inGrid]
    s = sigma_lineal(c, k, dV, f.E[:, 0], f.dt, 0)
    print(f"\n[2] malla hexagonal 800x800 (Nk = {len(k)}):  sum(dV)/A_BZ = {dV.sum()/A_BZ:.4f}   sigma_hex/sigma_referencia = {s/ref:.4f}")
    print(f"    (referencia a 800 nm: sigma/sigma_0 = {ref/SIGMA_0:.4f}, > 1 por la desviacion de la banda TB)")
    assert abs(dV.sum()/A_BZ - 1) < 1e-2, "la malla no cubre una zona de Brillouin"
    assert abs(s/ref - 1) < 1e-2, "la malla hexagonal no reproduce la referencia: revisar dV / area / factor 1e20"


N_K = 40                     # malla pequena (868 puntos): el objetivo es comparar con la referencia sobre los MISMOS k
NPTX = 1200                  # dt = 8.9e-18 s
NPT = 400                    # NPTX = NPT * 3: rk_dipole_velocity da muestras uniformes


def _evolucion_vs_lineal(direccion):
    c = cr.crystal.from_W90_TB_file(filename=W90_FILE, grid=malla_hexagonal(N_K), species_name='graphene',
                                    threshold_hopping=0)
    f = campo(direccion, 800.0, 4, NPTX)
    TBev = TBevolution.TBevolution_Bloch(c, f)
    t, vx, vy, vz = TBev.rk_dipole_velocity(NPT)
    assert np.allclose(np.diff(t), np.diff(t)[0]), "muestras temporales no uniformes (NPTX debe ser multiplo de NPT)"
    v = (vx if direccion == 'x' else vy).real
    j = G_S*(-e)*(v - v[0])*1e20                                      # [A/m]: j = g_s q sum_k dV v, dV en 1/A^2
    col = 0 if direccion == 'x' else 1
    E = f.E[np.rint(t/f.dt).astype(int), col]
    sigma_sim = _trapz(j*E, t)/_trapz(E*E, t)
    sigma_ref = sigma_lineal(c, TBev.k[:, :2], TBev.dV, f.E[:, col], f.dt, col)
    print(f"    campo segun {direccion}: sigma_sim/sigma_0 = {sigma_sim/SIGMA_0:.5f}   respuesta lineal/sigma_0 = {sigma_ref/SIGMA_0:.5f}"
          f"   cociente = {sigma_sim/sigma_ref:.5f}   (Nk = {len(TBev.k)})")
    assert sigma_sim > 0, "signo de la corriente: sigma negativa"
    assert abs(sigma_sim/sigma_ref - 1) < 3e-3, \
        "la evolucion no reproduce la respuesta lineal (si sale ~0.99: A = -cumsum(E)*dt en Field.py; usar cumulative_trapezoid)"


def test_evolucion_vs_respuesta_lineal_x():
    print("\n[3] rk_dipole_velocity vs respuesta lineal exacta sobre los mismos k (pulso debil, 800 nm)")
    _evolucion_vs_lineal('x')


def test_evolucion_vs_respuesta_lineal_y():
    _evolucion_vs_lineal('y')


if __name__ == "__main__":
    test_sigma_universal_referencia()
    test_malla_hexagonal_normalizacion()
    test_evolucion_vs_respuesta_lineal_x()
    test_evolucion_vs_respuesta_lineal_y()
    print("\nTodos los tests pasan.")
