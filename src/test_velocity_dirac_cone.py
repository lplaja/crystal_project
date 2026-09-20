"""
test_velocity_dirac_cone.py

Verificacion de la velocidad (banda de valencia) cerca de los conos de Dirac K y K'
del grafeno real (gr1NN_tb.dat, primeros vecinos).

Que se comprueba (cada bloque es una funcion test_*):

  1. test_bandas_puntos_simetria
       E(Gamma)=+-3t, E(M)=+-t, E(K)=E(K')=0.  K, K' y M se construyen a partir de
       los vectores reciprocos del cristal (no de "a" a mano):
           K = (2 b1 + b2)/3      K' = (b1 + 2 b2)/3      M = (b1 + b2)/2
  2. test_barrido_lineal_cono
       E(K+delta) crece linealmente con delta; la pendiente debe ser
       2*pi*hbar*v_F*1e10 (k en 1/A sin 2*pi), independiente de reciprocal_lattice_unit.
       Esto comprueba tambien la constante reciprocal_lattice_unit.
  3. test_velocidad_cono   (el test principal)
       Nivel 1: v = grad_k E / hbar por diferencia finita de las energias de `bands`.
                |v| = v_F (con desviacion trigonal ~ 4.5*delta), v radial hacia K,
                mismo |v| en K y K'.
       Nivel 2: _velocity_k (conmutador + grad h) coincide con el Nivel 1, punto a punto.
  4. test_evolucion_campo_nulo
       Con campo nulo, rk_evolve sobre autoestados de banda solo cambia la fase:
       norma conservada y _velocity_k por k sin cambios.

Ejecutar:   python3 test_velocity_dirac_cone.py        (imprime los numeros)
            pytest -s test_velocity_dirac_cone.py      (tambien vale)

Convenciones: k en 1/A sin 2*pi (como los reciprocal_vectors del cristal);
energias en julios; velocidades en m/s.
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")

from functools import lru_cache

import numpy as np
from scipy.constants import hbar, eV
import TBcrystal as cr
import grid as gr
import Field
import TBevolution

W90_FILE = '/home/lplaja/crystal_project/calculations/Wannier90 data/gr1NN_tb.dat'

# ---------------------------------------------------------------- tolerancias
TOL_E_EV = 1e-5          # energias en puntos de simetria [eV]
DK_FD = 1e-7             # paso de la diferencia finita [1/A]; debe ser << delta
DELTAS = (1e-3, 1e-4)    # distancias a K [1/A]
N_THETA = 12             # direcciones alrededor de K (incluye theta y theta+pi)
COEF_TRIGONAL = 6.0      # |v|/v_F-1 < COEF_TRIGONAL*delta (esperado ~4.5*delta)
TOL_FORMULA = 1e-5       # _velocity_k frente al gradiente de E (error relativo)
TOL_EVOL = 1e-8          # evolucion campo nulo (error relativo)


# ---------------------------------------------------------------- montaje
@lru_cache(maxsize=None)
def get_system():
    """Cristal real de grafeno + TBevolution con campo nulo (se construye una vez)."""
    filter_ = gr.polygonfilter
    filter_args = {'nsides': 6, 'radius': 2/3}
    ggr = gr.UniformCartesianGrid(2, [[-0.75, 0.75], [-0.75, 0.75]], [20, 20],
                                  origin=(0.0, 0), filter=filter_, filter_args=filter_args)
    TBcr = cr.crystal.from_W90_TB_file(filename=W90_FILE, grid=ggr,
                                       species_name='graphene', threshold_hopping=0)

    # 40 pasos de 5e-17 s; campo identicamente nulo
    tt = gr.UniformCartesianGrid(1, limits=[0, 2e-15], nptx=41)
    Efield = Field.polarizedHarmonicElectricField(
        tt, I_W__cm2=0.0, lambda0_nm=800.0, phi_rad=0.0,
        chi_rad=0.0, ellip=0.0, theta_rad=0.0,
    )
    Efield.A[:] = 0.0
    Efield.E[:] = 0.0

    TBev = TBevolution.TBevolution_Bloch(TBcr, Efield)
    return TBcr, TBev


def referencias(TBcr):
    """Puntos de simetria y constantes fisicas de referencia, sacados del propio cristal."""
    b1 = np.array(TBcr.reciprocal_vectors[0])[:2]
    b2 = np.array(TBcr.reciprocal_vectors[1])[:2]
    pts = {
        'Gamma': np.array([0.0, 0.0]),
        'M':     (b1 + b2)/2,
        'K':     (2*b1 + b2)/3,
        "K'":    (b1 + 2*b2)/3,
    }
    a_cc = 2/(3*np.linalg.norm(b1))            # [A]: |b| = 2/(3 a_cc) sin 2*pi
    R = np.array(TBcr.R_vectors)
    i0 = np.where(np.all(R == 0, axis=1))[0][0]
    t_hop = abs(TBcr.h_Rnm[i0][0, 1])          # [eV]
    v_F = 3*(t_hop*eV)*(a_cc*1e-10)/(2*hbar)   # [m/s]
    slope = 2*np.pi*hbar*v_F*1e10              # [J por 1/A]: E = hbar v_F |q|, q = 2 pi delta
    return pts, a_cc, t_hop, v_F, slope


def E_valencia(TBev, kx, ky):
    """Energia de la banda de valencia [J] en los puntos (kx,ky)."""
    en, _ = TBev.bands(kx, ky, np.zeros_like(kx))
    return en[0]


def puntos_alrededor(K0, delta):
    theta = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
    kx = K0[0] + delta*np.cos(theta)
    ky = K0[1] + delta*np.sin(theta)
    return theta, kx, ky, np.zeros_like(kx)


# ---------------------------------------------------------------- tests
def test_bandas_puntos_simetria():
    TBcr, TBev = get_system()
    pts, a_cc, t_hop, v_F, slope = referencias(TBcr)
    print(f"\n[1] a_cc = {a_cc:.5f} A,  t = {t_hop:.7f} eV,  v_F = {v_F:.6e} m/s")

    esperado = {'Gamma': 3*t_hop, 'M': t_hop, 'K': 0.0, "K'": 0.0}   # |E| en eV
    for nombre, k0 in pts.items():
        en, _ = TBev.bands(np.array([k0[0]]), np.array([k0[1]]), np.array([0.0]))
        Ev, Ec = en[0, 0]/eV, en[1, 0]/eV
        print(f"    {nombre:6s}: E_val = {Ev:+.7f} eV   E_cond = {Ec:+.7f} eV   (|E| esperado {esperado[nombre]:.7f})")
        assert abs(Ev + esperado[nombre]) < TOL_E_EV, f"{nombre}: E_val incorrecta"
        assert abs(Ec - esperado[nombre]) < TOL_E_EV, f"{nombre}: E_cond incorrecta"


def test_barrido_lineal_cono():
    TBcr, TBev = get_system()
    pts, a_cc, t_hop, v_F, slope = referencias(TBcr)
    print(f"\n[2] pendiente esperada = {slope/eV:.4f} eV por (1/A)")
    for nombre in ('K', "K'"):
        K0 = pts[nombre]
        for delta in (1e-5, 1e-4, 1e-3):
            E = -E_valencia(TBev, np.array([K0[0] + delta]), np.array([K0[1]]))[0]   # > 0
            rel = E/delta/slope - 1
            print(f"    {nombre:2s} delta={delta:.0e}:  E/delta = {E/delta/eV:.4f} eV   E/delta / esperado - 1 = {rel:+.2e}")
            assert abs(rel) < 10*delta + 1e-6, f"E no es lineal con la pendiente esperada en {nombre}, delta={delta}"


def test_velocidad_cono():
    TBcr, TBev = get_system()
    pts, a_cc, t_hop, v_F, slope = referencias(TBcr)
    rlu = TBcr.reciprocal_lattice_unit
    print(f"\n[3] v_F = {v_F:.6e} m/s")
    print("    punto  delta   max|v/vF-1|   max(cos v.delta)   err(formula vs FD)")

    vmod_por_punto = {}
    for nombre in ('K', "K'"):
        for delta in DELTAS:
            theta, kx, ky, kz = puntos_alrededor(pts[nombre], delta)

            # Nivel 1: gradiente de las energias (diferencia finita centrada), en m/s
            vx_fd = (E_valencia(TBev, kx + DK_FD, ky) - E_valencia(TBev, kx - DK_FD, ky))/(2*DK_FD)/rlu/hbar
            vy_fd = (E_valencia(TBev, kx, ky + DK_FD) - E_valencia(TBev, kx, ky - DK_FD))/(2*DK_FD)/rlu/hbar
            vmod = np.hypot(vx_fd, vy_fd)
            dev_vF = np.max(np.abs(vmod/v_F - 1))
            cos_rad = (vx_fd*np.cos(theta) + vy_fd*np.sin(theta))/vmod    # -1 si v apunta hacia K

            # Nivel 2: _velocity_k (conmutador + grad h) con el autoestado de valencia
            _, eig = TBev.bands(kx, ky, kz)
            CB = eig[:, 0, :]
            vk = TBev._velocity_k(CB, kx, ky, kz)                          # (n_k, 3)
            err_form = max(np.max(np.abs(vk[:, 0].real - vx_fd)),
                           np.max(np.abs(vk[:, 1].real - vy_fd)))/np.max(vmod)
            im_rel = np.max(np.abs(vk.imag))/np.max(vmod)

            print(f"    {nombre:2s}   {delta:.0e}   {dev_vF:.3e}       {cos_rad.max():+.6f}        {err_form:.2e}")
            assert dev_vF < COEF_TRIGONAL*delta, f"{nombre}, delta={delta}: |v| se desvia de v_F mas de lo esperado"
            assert cos_rad.max() < -0.9999, f"{nombre}, delta={delta}: v no es radial hacia K"
            assert err_form < TOL_FORMULA, f"{nombre}, delta={delta}: _velocity_k no coincide con grad E"
            assert im_rel < 1e-8, f"{nombre}, delta={delta}: v tiene parte imaginaria"
            vmod_por_punto[(nombre, delta)] = np.sort(vmod)

    # K' = -K (mod G): mismo conjunto de |v| (las direcciones theta y theta+pi estan en la lista)
    for delta in DELTAS:
        assert np.allclose(vmod_por_punto[('K', delta)], vmod_por_punto[("K'", delta)], rtol=1e-5), \
            f"|v| distinta en K y K' (delta={delta})"


def test_evolucion_campo_nulo():
    TBcr, TBev = get_system()
    pts, a_cc, t_hop, v_F, slope = referencias(TBcr)
    print("\n[4] evolucion con campo nulo sobre autoestados de valencia cerca de K y K'")
    nsteps = 30
    for nombre in ('K', "K'"):
        _, kx, ky, kz = puntos_alrededor(pts[nombre], 1e-3)
        _, eig = TBev.bands(kx, ky, kz)
        CB0 = eig[:, 0, :].copy()
        CB1 = TBev.rk_evolve(CB0.copy(), kx, ky, kz, 0, nsteps)

        norma = np.sum(np.abs(CB1)**2, axis=0)
        err_norma = np.max(np.abs(norma - 1))
        v0 = TBev._velocity_k(CB0, kx, ky, kz)
        v1 = TBev._velocity_k(CB1, kx, ky, kz)
        err_v = np.max(np.abs(v1 - v0))/np.max(np.abs(v0))
        print(f"    {nombre:2s}: |norma-1| = {err_norma:.2e}   variacion relativa de v_k = {err_v:.2e}")
        assert err_norma < TOL_EVOL, f"{nombre}: la norma no se conserva"
        assert err_v < TOL_EVOL, f"{nombre}: v_k cambia con campo nulo"


if __name__ == "__main__":
    test_bandas_puntos_simetria()
    test_barrido_lineal_cono()
    test_velocidad_cono()
    test_evolucion_campo_nulo()
    print("\nTodos los tests pasan.")