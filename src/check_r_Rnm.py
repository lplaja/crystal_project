"""
check_r_Rnm.py
Inspecciona la matriz de posicion r_Rnm que lee TBcrystal del fichero Wannier90 del grafeno 1NN.
Objetivo: saber si los elementos fuera de la diagonal (dipolo interorbital) son ~0
(acoplo esencialmente Peierls) o no. Unidades: Angstrom (tal como sale del _tb.dat).
"""
import sys
sys.path.append("/home/lplaja/crystal_project/src")

import numpy as np
import TBcrystal as cr
import grid as gr

W90_FILE = '/home/lplaja/crystal_project/calculations/Wannier90 data/gr1NN_tb.dat'

ggr = gr.UniformCartesianGrid(2, [[-0.75, 0.75], [-0.75, 0.75]], [20, 20],
                              origin=(0.0, 0), filter=gr.polygonfilter,
                              filter_args={'nsides': 6, 'radius': 2/3})
TBcr = cr.crystal.from_W90_TB_file(filename=W90_FILE, grid=ggr,
                                   species_name='graphene', threshold_hopping=0)

r_c = TBcr.r_Rnm                     # (nrpts, nw, nw, 3) complejo, en Angstrom
r = np.abs(r_c)
i0 = TBcr.ir0
nR, nw = r.shape[0], r.shape[1]
R = np.array(TBcr.R_vectors)

print(f"nrpts = {nR},  num_wann = {nw},  ir0 = {i0}")
print("max |r| total [A]:", r.max())
print("posiciones de orbital (diagonal, R=0) [A]:")
print(np.real(r_c[i0, np.arange(nw), np.arange(nw)]))

# Mascara de elementos fuera de la diagonal orbital (m != n), para todos los R
offdiag = ~np.eye(nw, dtype=bool)
r_off = r[:, offdiag, :]             # (nrpts, nw*(nw-1), 3)
print("\nmax |r| fuera de la diagonal (todos los R) [A]:", r_off.max())
print("max |r| fuera de la diagonal en R=0 [A]:", r[i0][offdiag].max())

# Diagonal con R != 0 (deberia ser ~0 si las posiciones son solo las del R=0)
mask_R = np.arange(nR) != i0
diag_R = np.array([r[mask_R, n, n, :].max() for n in range(nw)])
print("max |r| diagonal con R != 0 [A], por orbital:", diag_R)

# Los 5 elementos fuera de la diagonal mas grandes: donde estan
print("\nElementos m != n mas grandes (R, m, n, componente, |r|):")
lista = []
for ir in range(nR):
    for m in range(nw):
        for n in range(nw):
            if m != n:
                for c in range(3):
                    lista.append((r[ir, m, n, c], tuple(R[ir]), m, n, c))
lista.sort(reverse=True)
for val, Rv, m, n, c in lista[:5]:
    print(f"   R={Rv}  m={m} n={n}  comp={'xyz'[c]}  |r|={val:.3e} A")

# Parte imaginaria de la diagonal (deberia ser ~0: es una posicion real)
print("\nmax |Im r| diagonal [A]:", np.max(np.abs(np.imag(r_c[:, np.arange(nw), np.arange(nw), :]))))
