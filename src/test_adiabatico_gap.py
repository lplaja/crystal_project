"""
test_adiabatico_gap.py

Prueba 2 de la validacion del grafeno con campo: LIMITE ADIABATICO (parte INTRABANDA).

Idea: con un campo lento y debil, el estado sigue a la banda de valencia y la corriente es
      j(t) = g_s q  sum_k dV  v_vv(kappa(t)),   v_vv = (1/hbar) dE_v/dk,   kappa = k - (q/hbar) A(t)
calculada SOLO con la estructura de bandas, sin resolver la dinamica. La simulacion
(rk_dipole_velocity: RK4 + acoplo qE.r + velocidad con el conmutador) debe reproducirla.

Por que grafeno CON GAP: el grafeno sin dopar no tiene gap, el teorema adiabatico falla cerca
de K y la parte interbanda (sigma_0, ver test_conductividad_lineal.py) contamina la
comparacion. Se anade un desplazamiento de energia +-Delta en los sitios A/B (Delta = 1 eV,
E_cv >= 2 eV >> hbar*omega = 0.1 eV): la transicion Landau-Zener es despreciable.

Correccion de polarizacion: aun adiabaticamente, el campo mezcla un poco de banda de conduccion
(estado de valencia dressed) y aparece la corriente de polarizacion
      j_pol = d(P)/dt,   P = alpha(t) E(t),   alpha = 2 g_s q^2 sum_k dV |<c|dh/dk|v>|^2 / E_cv^3
Se calcula analiticamente (2x2) y se resta. Es ~0.5 % de la corriente intrabanda.

Resultado esperado (n=24, 12 um, 1e10 W/cm2, 3 ciclos, dt = T/4000):
   error max / max|j_ref|:   x -> 1.6e-3      y -> 1.1e-2
(en y, j_ref es 14 veces menor que en x -por simetria- y el residuo absoluto es similar,
 por eso el error relativo es mayor; tolerancias 4e-3 y 2.5e-2.)

Ejecutar:  python3 test_adiabatico_gap.py      (o pytest -s)      Tiempo: ~1 min.
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")
sys.path.append(".")

import numpy as np
from scipy.constants import hbar, eV, elementary_charge as e
import TBcrystal as cr
import Field
import TBevolution
import grid as gr
from test_conductividad_lineal import W90_FILE, G_S, radio_BZ, crystal_referencia, malla_hexagonal

DELTA_EV = 1.0
LAMBDA_NM = 12000.
INTENSIDAD = 1e10        # W/cm2  ->  E0 ~ 2.6e8 V/m
N_MALLA = 24
NPTX, NPT, N_CICLOS = 12000, 400, 3
TOL = {'x': 4e-3, 'y': 2.5e-2}


def F_dF(crystal, k_true, direccion):
    """F(k) = sum_R t_R exp(i k.delta_R) (gauge fisico) y dF/dk_direccion. k_true en 1/A con 2 pi."""
    Rc = np.asarray(crystal.R_vectors) @ np.asarray(crystal.direct_vectors)
    delta = (crystal.deltas[1] - crystal.deltas[0]) + Rc
    t = crystal.h_Rnm[:, 0, 1].real/crystal.deg_weights
    F = 0; dF = 0
    for tR, d in zip(t, delta):
        if tR == 0: continue
        ph = np.exp(1j*(k_true[:, 0]*d[0] + k_true[:, 1]*d[1]))
        F = F + tR*ph; dF = dF + tR*1j*d[direccion]*ph
    return F, dF


def simula_y_referencia(direccion):
    col = 0 if direccion == 'x' else 1
    c = cr.crystal.from_W90_TB_file(W90_FILE, malla_hexagonal(N_MALLA), 'graphene', 0)
    c.h_Rnm[c.ir0, 0, 0] = +DELTA_EV          # A: +Delta ; B: -Delta   (gap 2 Delta en K y K')
    c.h_Rnm[c.ir0, 1, 1] = -DELTA_EV
    T0 = Field.lambda2T(LAMBDA_NM*1e-9)
    tt = gr.UniformCartesianGrid(1, limits=[0, N_CICLOS*T0], nptx=NPTX)
    f = Field.polarizedHarmonicElectricField(
        tt, I_W__cm2=INTENSIDAD, lambda0_nm=LAMBDA_NM, env=Field.env_sin2,
        env_parameters={'start': 0, 'end': 1, 'ton': 0.5, 'toff': 0.5},
        phi_rad=-np.pi/2, chi_rad=0.0 if direccion == 'x' else np.pi/2, ellip=0.0)
    tb = TBevolution.TBevolution_Bloch(c, f)

    # ---- simulacion
    t, vx, vy, vz = tb.rk_dipole_velocity(NPT)
    v = (vx if direccion == 'x' else vy).real
    j_sim = G_S*(-e)*v*1e20                                     # A/m
    assert np.allclose(np.diff(t), t[1]-t[0], rtol=1e-6), "muestreo no uniforme (usa nptx multiplo de npt)"

    # ---- referencia de bandas + polarizacion (analitica 2x2)
    idx = np.rint(t/f.dt).astype(int)
    A = f.A[idx, :2]                                            # V s/m
    E = f.E[idx, col]
    kt = 2*np.pi*tb.k[:, None, :2] - (-e/hbar)*A[None, :, :]*1e-10      # kappa(t) verdadero, 1/A
    j_ref = np.zeros(len(t)); alpha = np.zeros(len(t))
    for it in range(len(t)):
        F, dF = F_dF(c, kt[:, it, :], col)
        Ec = np.sqrt(DELTA_EV**2 + np.abs(F)**2)                # eV
        vk = -np.real(np.conj(F)*dF)/Ec*eV*1e-10/hbar           # (1/hbar) dE_v/dk  [m/s]
        j_ref[it] = G_S*(-e)*np.sum(tb.dV*vk)*1e20
        # elemento de matriz <c|dh/dk|v> (2x2 analitico, eigh)
        h = np.zeros((len(F), 2, 2), complex)
        h[:, 0, 0] = DELTA_EV; h[:, 1, 1] = -DELTA_EV; h[:, 0, 1] = F; h[:, 1, 0] = np.conj(F)
        dh = np.zeros_like(h); dh[:, 0, 1] = dF; dh[:, 1, 0] = np.conj(dF)
        _, U = np.linalg.eigh(h)
        M = np.einsum('ki,kij,kj->k', np.conj(U[:, :, 1]), dh, U[:, :, 0])   # eV A
        Ecv = 2*Ec
        alpha[it] = 2*G_S*e**2*np.sum(tb.dV*np.abs(M)**2/Ecv**2/(Ecv*eV))   # C m / (V/m) x ...
    j_pol = np.gradient(alpha*E, t)
    return t, j_sim, j_ref, j_pol


def _comprueba(direccion):
    t, j_sim, j_ref, j_pol = simula_y_referencia(direccion)
    escala = np.abs(j_ref).max()
    err_sin = np.abs(j_sim - j_ref).max()/escala
    err_con = np.abs(j_sim - j_ref - j_pol).max()/escala
    print(f"  {direccion}: max|j_ref|={escala:.3e} A/m  max|j_pol|={np.abs(j_pol).max():.2e} A/m  "
          f"err sin j_pol={err_sin:.2e}  con j_pol={err_con:.2e}  (tol {TOL[direccion]:.1e})")
    assert err_con < TOL[direccion]
    assert err_con < err_sin/3, "la correccion de polarizacion deberia reducir claramente el error"


def test_adiabatico_x():
    _comprueba('x')


def test_adiabatico_y():
    _comprueba('y')


if __name__ == "__main__":
    test_adiabatico_x()
    test_adiabatico_y()
    print("OK")
