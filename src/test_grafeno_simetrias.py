"""
test_grafeno_simetrias.py

Pruebas de SIMETRIA sobre grafeno real (gr1NN_tb.dat) con campo activo. No necesitan
solucion de referencia: son propiedades exactas de la dinamica.

  0. test_malla_es_una_zona_de_brillouin
       La malla hexagonal cubre UNA zona de Brillouin: sum(dV) = |b1 x b2| y es simetrica
       (media de k = 0). Ojo: el hexagono debe tener radio |K| = 2/(3a) = 0.271 1/A
       (en 1/A, no en unidades de 1/a). Con radius=2/3 la malla cubria ~6 zonas.
  1. test_sin_corriente_perpendicular
       Campo lineal segun x (Gamma-M) o segun y (Gamma-K): <v_perp>(t) = 0 por simetria
       de espejo. Exige malla simetrica bajo el espejo (origen desplazado dx/2).
  2. test_inversion_campo_y   (Gamma-K)
       Inversion espacial: E -> -E implica <v>(t) -> -<v>(t) para TODO pulso (es la
       version exacta de "no hay armonicos pares"). Con E segun y es exacto (~1e-12).
  3. test_inversion_campo_x   (Gamma-M)
       Igual, pero aqui el termino -qE.r contiene un desplazamiento escalar qE.tau
       (tau_B = 1.42 A en x) que la solucion exacta ignora (fase global) y RK4 no:
       la violacion es pura truncacion de RK4, y debe bajar como dt^4-5 (se comprueba).

Ejecutar:  python3 test_grafeno_simetrias.py     (o pytest -s test_grafeno_simetrias.py)
Tiempo: ~1 min.
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")

from functools import lru_cache
import numpy as np
import TBcrystal as cr
import grid as gr
import Field
import TBevolution

W90_FILE = '/home/lplaja/crystal_project/calculations/Wannier90 data/gr1NN_tb.dat'

N_K = 24                    # malla n x n (con hexagono filtrado ~300 puntos k): solo mecanica
LAMBDA_NM = 3000.0
INTENSIDAD = 5e10           # W/cm^2  (E0 ~ 6e8 V/m)
N_CICLOS = 4
NPT_TIEMPO = 3200           # dt = 1.25e-17 s
N_MUESTRAS = 200


# ---------------------------------------------------------------- montaje
@lru_cache(maxsize=None)
def radio_BZ():
    """Distancia Gamma-K [1/A] = |2 b1 + b2|/3, sacada de los vectores reciprocos del cristal."""
    g = gr.UniformCartesianGrid(2, [[-1, 1], [-1, 1]], [2, 2])
    c = cr.crystal.from_W90_TB_file(filename=W90_FILE, grid=g, species_name='graphene', threshold_hopping=0)
    b1, b2 = np.array(c.reciprocal_vectors[0])[:2], np.array(c.reciprocal_vectors[1])[:2]
    return np.linalg.norm(2*b1 + b2)/3, abs(b1[0]*b2[1] - b1[1]*b2[0])


def malla(n=N_K):
    R, _ = radio_BZ()
    L = 1.1*R
    dx = 2*L/n
    # origin = dx/2 -> nodos simetricos respecto a k=0 (sin nodos en el borde del hexagono)
    return gr.UniformCartesianGrid(2, [[-L, L], [-L, L]], [n, n], origin=(dx/2, dx/2),
                                   filter=gr.polygonfilter, filter_args={'nsides': 6, 'radius': R})


def evolucion(direccion, signo, nptx=NPT_TIEMPO, n=N_K):
    """Velocidad media <v>(t) (por celda, sin espin) con campo lineal segun 'x' o 'y'.
    signo=+1/-1 invierte el campo (fase de la portadora + pi)."""
    chi = 0.0 if direccion == 'x' else np.pi/2
    phi = -np.pi/2 if signo > 0 else np.pi/2
    g = malla(n)
    crystal = cr.crystal.from_W90_TB_file(filename=W90_FILE, grid=g, species_name='graphene', threshold_hopping=0)
    T0 = Field.lambda2T(LAMBDA_NM*1e-9)
    tt = gr.UniformCartesianGrid(1, limits=[0, N_CICLOS*T0], nptx=nptx)
    campo = Field.polarizedHarmonicElectricField(
        tt, I_W__cm2=INTENSIDAD, lambda0_nm=LAMBDA_NM, phi_rad=phi, chi_rad=chi, ellip=0.0,
        env=Field.env_sin2, env_parameters={'start': 0, 'end': 1, 'ton': 0.5, 'toff': 0.5})
    TBev = TBevolution.TBevolution_Bloch(crystal, campo)
    _, vx, vy, vz = TBev.rk_dipole_velocity(N_MUESTRAS)
    return np.array([vx, vy, vz])                       # (3, n_muestras), complejo


@lru_cache(maxsize=None)
def cache(direccion, signo, nptx):
    return evolucion(direccion, signo, nptx)


def par_perp(v, direccion):
    return (v[0], v[1]) if direccion == 'x' else (v[1], v[0])


# ---------------------------------------------------------------- tests
def test_malla_es_una_zona_de_brillouin():
    R, A_BZ = radio_BZ()
    g = malla()
    area = np.sum(g.dV[g.inGrid])
    k = g.x[g.inGrid]
    print(f"\n[0] |K| = {R:.5f} 1/A   A_BZ = {A_BZ:.6f} 1/A^2   sum(dV)/A_BZ = {area/A_BZ:.4f}   |<k>| = {np.abs(k.mean(axis=0)).max():.1e}")
    assert abs(area/A_BZ - 1) < 0.05, "la malla no cubre una zona de Brillouin"
    assert np.abs(k.mean(axis=0)).max() < 1e-12, "la malla no es simetrica respecto a k=0"


def test_sin_corriente_perpendicular():
    print("\n[1] corriente perpendicular al campo (simetria de espejo)")
    for d in ('x', 'y'):
        par, perp = par_perp(cache(d, +1, NPT_TIEMPO), d)
        rel = np.abs(perp).max()/np.abs(par).max()
        print(f"    campo segun {d}: max|v_par| = {np.abs(par).max():.3e} m/s   |v_perp|/|v_par| = {rel:.1e}")
        assert np.abs(par).max() > 1.0, "el test seria vacuo: no hay corriente paralela"
        assert rel < 1e-10, f"campo segun {d}: hay corriente perpendicular"


def test_inversion_campo_y():
    print("\n[2] inversion E -> -E, campo segun y (Gamma-K)")
    par, _ = par_perp(cache('y', +1, NPT_TIEMPO), 'y')
    par_m, _ = par_perp(cache('y', -1, NPT_TIEMPO), 'y')
    err = np.abs(par + par_m).max()/np.abs(par).max()
    print(f"    max|v(E)+v(-E)|/max|v| = {err:.1e}   max|Im v|/max|Re v| = {np.abs(par.imag).max()/np.abs(par.real).max():.1e}")
    assert err < 1e-10, "v(-E) != -v(E): la dinamica rompe la simetria de inversion"
    assert np.abs(par.imag).max()/np.abs(par.real).max() < 1e-10


def test_inversion_campo_x():
    print("\n[3] inversion E -> -E, campo segun x (Gamma-M): la violacion debe ser truncacion de RK4")
    errs = []
    for nptx in (NPT_TIEMPO//2, NPT_TIEMPO):
        par, _ = par_perp(cache('x', +1, nptx), 'x')
        par_m, _ = par_perp(cache('x', -1, nptx), 'x')
        errs.append(np.abs(par + par_m).max()/np.abs(par).max())
        print(f"    nptx = {nptx}:  max|v(E)+v(-E)|/max|v| = {errs[-1]:.2e}")
    print(f"    cociente al doblar la resolucion temporal = {errs[0]/errs[1]:.1f}  (RK4: >= 16)")
    assert errs[1] < 1e-5, "v(-E) != -v(E) mas alla de la truncacion de RK4"
    assert errs[0]/errs[1] > 12, "el error no baja como RK4 al reducir dt: no es truncacion"


if __name__ == "__main__":
    test_malla_es_una_zona_de_brillouin()
    test_sin_corriente_perpendicular()
    test_inversion_campo_y()
    test_inversion_campo_x()
    print("\nTodos los tests pasan.")
