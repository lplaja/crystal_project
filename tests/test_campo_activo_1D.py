"""
test_campo_activo_1D.py

Tests con CAMPO ACTIVO y solucion analitica exacta, sobre cristales 1D sinteticos
(sin fichero Wannier90): se construye un "cristal falso" con las magnitudes que usa
TBevolution_Bloch y se le pasa un campo constante.

  A) test_rabi_*                    Rabi de dos niveles (2 orbitales, sin hopping, dipolo r_Rnm).
                                    P_up(t) = V^2/(V^2+a^2) sin^2( sqrt(V^2+a^2) t/hbar ),
                                    a = eps/2, V = |q| F_eff d.
     -> valida: acoplo campo-dipolo (qF.r), unidades de F [V/m] y de r_Rnm [A->m],
        rk_evolve (RK4) y la ROTACION del campo a cartesianas (columnas paralela/perp/axial
        con s_direction inclinada), incluido el signo relativo de q y s.
  B) test_oscilaciones_de_Bloch     Cadena de 1 orbital, E(k) = -2 t cos(2 pi k a), un solo k.
                                    v(t) = (2 t a/hbar) sin(2 pi k0 a - e F a t/hbar)
     -> valida: kappa(t) = k - (q/hbar) A(t) (signo, 2*pi, reciprocal_lattice_unit),
        _velocity_k y toda la fontaneria de rk_dipole_velocity (muestreo temporal).

Ejecutar:  python3 test_campo_activo_1D.py      (o pytest -s test_campo_activo_1D.py)
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")

from types import SimpleNamespace
import numpy as np
from scipy.constants import hbar, eV, elementary_charge
import grid as gr
import Field
import TBevolution as TBe

A_LAT = 3.0                 # A, constante de red de la cadena (solo cuenta en el test B)
DL = 1e-10                  # direct_lattice_unit: A -> m
RLU = 2*np.pi*1e10          # reciprocal_lattice_unit (validado en test_velocity_dirac_cone.py)


# ------------------------------------------------------------------ montaje
def crystal_falso(h_Rnm_eV, r_Rnm_A, R_vectors, k):
    """Objeto con lo que TBevolution_Bloch necesita de un cristal 1D. k en 1/A (sin 2 pi)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n_orb = h_Rnm_eV.shape[1]
    grid = SimpleNamespace(cartesian=True, ndim=1, x=k[:, None],
                           inGrid=np.ones(len(k), dtype=bool), dV=np.ones(len(k)))
    # numba (@njit en _r_nm y _grad_t_nm) exige mismo dtype en np.dot: el cristal real trae
    # h_Rnm y r_Rnm complejos (Wannier90), asi que aqui tambien.
    return SimpleNamespace(
        grid=grid, h_Rnm=h_Rnm_eV.astype(np.complex128), deg_weights=np.ones(len(R_vectors)),
        r_Rnm=r_Rnm_A.astype(np.complex128),
        R_vectors=np.array(R_vectors), direct_vectors=A_LAT*np.eye(3),
        direct_lattice_unit=DL, reciprocal_lattice_unit=RLU,
        deltas=np.zeros((n_orb, 3)), num_wann=n_orb, nrpts=len(R_vectors))


def campo_constante(amplitudes, s_direction, N, dt):
    """Campo constante en el tiempo: amplitudes = {columna: F [V/m]} con columnas
    0=paralela, 1=perpendicular, 2=axial. A = -F t exacto (sin el desfase de cumsum)."""
    tt = gr.UniformCartesianGrid(1, limits=[0, N*dt], nptx=N)
    f = Field.PulsedField(tt, I_W__cm2=0.0, lambda0_nm=800.0,
                                             s_direction_cartesian=np.array(s_direction, dtype=float))
    t = f.t
    f.E[:] = 0.0
    f.A[:] = 0.0
    for col, F in amplitudes.items():
        f.E[:, col] = F
        f.A[:, col] = -F*t
    return f


# ------------------------------------------------------------------ A) Rabi
EPS_EV = 1.0                # separacion entre niveles [eV]
D_A = 1.0                   # modulo del dipolo [A]
F0 = 2.5e9                  # V/m  ->  V = e F d = 0.25 eV
N_T, DT = 801, 1e-17


def frame_explicito(beta):
    """(p, q, s) escritos a mano para s_direction = (0, sin b, cos b): NO usa polarization_frame."""
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, np.cos(beta), -np.sin(beta)])
    s = np.array([0.0, np.sin(beta), np.cos(beta)])
    return p, q, s


def rabi_P(t, F_eff):
    a = 0.5*EPS_EV*eV
    V = elementary_charge*abs(F_eff)*D_A*DL
    E_R = np.hypot(a, V)
    return (V/E_R)**2*np.sin(E_R*t/hbar)**2


# nombre, beta, direccion del dipolo, {columna: amplitud relativa}
CASOS_RABI = [
    ("par_x",                    0.0, (1, 0, 0), {0: 1.0}),
    ("perp_no_acopla_con_dx",    0.0, (1, 0, 0), {1: 1.0}),
    ("par_x_con_s_inclinada",    0.5, (1, 0, 0), {0: 1.0}),
    ("perp_dy_inclinada",        0.5, (0, 1, 0), {1: 1.0}),
    ("perp_dz_inclinada",        0.5, (0, 0, 1), {1: 1.0}),
    ("axial_dy_inclinada",       0.5, (0, 1, 0), {2: 1.0}),
    ("perp+axial_dy (signos)",   0.5, (0, 1, 0), {1: 1.0, 2: 1.0}),
    ("perp+axial_dz (signos)",   0.5, (0, 0, 1), {1: 1.0, 2: 1.0}),
]


def _rabi_caso(nombre, beta, n_dip, columnas, pasos=(100, 250, 400, 700)):
    n_dip = np.array(n_dip, dtype=float)
    p, q, s = frame_explicito(beta)
    ejes = {0: p, 1: q, 2: s}
    F_eff = F0*sum(amp*np.dot(n_dip, ejes[c]) for c, amp in columnas.items())   # F . n

    h = np.array([[[-0.5*EPS_EV, 0.0], [0.0, 0.5*EPS_EV]]])                    # (nR=1, 2, 2) eV
    r = np.zeros((1, 2, 2, 3)); r[0, 0, 1] = r[0, 1, 0] = D_A*n_dip              # A
    crystal = crystal_falso(h, r, [[0, 0, 0]], k=[0.0])
    campo = campo_constante({c: F0*amp for c, amp in columnas.items()},
                            (0, np.sin(beta), np.cos(beta)), N_T, DT)
    TBev = TBe.TBevolution_Bloch(crystal, campo)
    kx, ky, kz = TBev.k[:, 0], TBev.k[:, 1], TBev.k[:, 2]

    print(f"    {nombre:26s} F_eff/F0 = {F_eff/F0:+.4f}")
    for n in pasos:
        CB = TBev.rk_evolve(TBev.CB.copy(), kx, ky, kz, 0, n)
        P_num = abs(CB[1, 0])**2
        P_ana = rabi_P(n*campo.dt, F_eff)
        assert abs(P_num - P_ana) < 1e-6, f"{nombre}: paso {n}: P_num={P_num:.8f}  P_analitica={P_ana:.8f}"


def test_rabi_todos_los_casos():
    print("\n[A] Rabi de dos niveles con campo constante")
    for caso in CASOS_RABI:
        _rabi_caso(*caso)


def test_rabi_hay_dinamica_apreciable():
    """Control: en el caso acoplado la poblacion llega a P ~ V^2/E_R^2 = 0.2; si no, el test seria vacuo."""
    P = rabi_P(np.linspace(0, N_T*DT, 400), F0)
    assert np.max(P) > 0.15


# ------------------------------------------------------------------ B) Bloch
def test_oscilaciones_de_Bloch():
    print("\n[B] Oscilaciones de Bloch, cadena de 1 orbital, un solo k")
    t_hop = 1.0                                         # eV
    a_m = A_LAT*DL
    T_B = 1e-14                                         # periodo de Bloch [s]
    F = hbar*(2*np.pi/T_B)/(elementary_charge*a_m)      # V/m
    fase0 = 0.7                                         # 2 pi k0 a [rad]
    k0 = fase0/(2*np.pi*A_LAT)                          # 1/A

    h = np.zeros((3, 1, 1)); h[0] = -t_hop; h[2] = -t_hop     # R = -1, 0, +1
    r = np.zeros((3, 1, 1, 3))
    crystal = crystal_falso(h, r, [[-1, 0, 0], [0, 0, 0], [1, 0, 0]], k=[k0])
    campo = campo_constante({0: F}, (0, 0, 1), 2001, 1e-17)          # 2 periodos de Bloch
    TBev = TBe.TBevolution_Bloch(crystal, campo)

    time_dip, vx, vy, vz = TBev.rk_dipole_velocity(npt=40)
    q = -elementary_charge
    v0 = 2*t_hop*eV*a_m/hbar
    v_ana = v0*np.sin(fase0 + q*F*a_m*time_dip/hbar)

    err = np.max(np.abs(vx.real - v_ana))/v0
    print(f"    F = {F:.3e} V/m   v0 = {v0:.3e} m/s   error relativo max = {err:.2e}")
    print(f"    max|Im vx|/v0 = {np.max(np.abs(vx.imag))/v0:.1e}   max|vy|,|vz|/v0 = {max(np.max(np.abs(vy)), np.max(np.abs(vz)))/v0:.1e}")
    assert time_dip[-1] > 1.5*T_B, "el muestreo no cubre 1.5 periodos de Bloch"
    # El error residual (~7e-9) es la no-unitariedad de RK4: |C|^2 decae ~ z^6/72 por paso
    # (z = E dt/hbar ~ 0.03) y v es proporcional a |C|^2. No es un error del codigo.
    assert err < 1e-7, "v(t) no sigue la oscilacion de Bloch analitica"
    assert np.max(np.abs(vx.imag))/v0 < 1e-8
    assert max(np.max(np.abs(vy)), np.max(np.abs(vz)))/v0 < 1e-8


if __name__ == "__main__":
    test_rabi_todos_los_casos()
    test_rabi_hay_dinamica_apreciable()
    test_oscilaciones_de_Bloch()
    print("\nTodos los tests pasan.")