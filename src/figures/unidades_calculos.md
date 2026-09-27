# Unidades: de la simulación al análisis de armónicos

Crystal Project, cadena TBevolution → executeTB → `tilt_oscar_view.ipynb`.
Donde la dimensión importa se indica con dim (grafeno: dim = 2).

## 0. Fichero de entrada `config.xtoml`

| Clave | Qué es | Unidades |
|---|---|---|
| `crystal.W90filename` | fichero `*_tb.dat` de Wannier90 (h en eV, r y vectores de red en Å) | — |
| `crystal.threshold_hopping` | se conservan las celdas R con \|h_R\| > umbral·máx\|h_R\|; 0 = todas | fracción (adimensional) |
| `crystal.a` | constante de red; solo informativa (executeTB no la lee) | Å |
| `crystal.dim` | dimensión de la malla de k | entero |
| `BZ.nkx, nky, nkz` | puntos de la malla en cada dirección | entero |
| `BZ.kx_min … kz_max` | límites de la malla | Å⁻¹ (sin 2π) |
| `BZ.filter.args.radius` | radio del filtro poligonal (\|ΓK\| = 2/(3a)) | Å⁻¹ (sin 2π) |
| `BZ.filter.args.nsides` | lados del polígono | entero |
| `Field.I_W__cm2` | irradiancia de pico | W/cm² |
| `Field.lambda` | longitud de onda | nm |
| `Field.varphi` | fase de la portadora (CEP) | rad |
| `Field.chi` | inclinación de la elipse, desde e∥ | rad |
| `Field.ellip` | elipticidad ε = ±b/a | adimensional |
| `Field.theta` (opcional) | ángulo de la componente axial | rad |
| `Field.s_direction` (opcional) | dirección de propagación (cartesiana) | vector adimensional |
| `Field.field_type` | amplitud definida sobre `'E'` o `'A'` | — |
| `Field.env.args` (start, end, ton, toff) | envolvente: inicio, fin y rampas | fracción de la duración total |
| `time.tini`, `time.tfin` | inicio y fin de la simulación | periodos ópticos T₀ |
| `time.nt` | pasos temporales del Runge–Kutta | entero |
| `evolver.calculation.args.nt` | muestras guardadas de vd (`time.nt`/nt entero) | entero |

## 1. Entrada: cristal y malla de k

| Magnitud | Qué es | Unidades |
|---|---|---|
| `R` | vectores de red (cartesianos) | Å |
| `k`, `kx, ky` | vector de onda **sin 2π**: fase exp(2πi k·R) | Å⁻¹ |
| `dV` | peso de cada punto k; Σ dV = 1/cell_size | Å⁻ᵈⁱᵐ |
| `cell_size` | área (volumen) de la celda, √det(a aᵀ) | Åᵈⁱᵐ (5.24084 Å²) |
| `crystal.h_Rnm` | hamiltoniano leído de Wannier90 | eV |
| `crystal.r_Rnm` | elementos de posición leídos de Wannier90 | Å |
| `reciprocal_lattice_unit` (RLU) | factor de k_código a k_SI: k_SI = RLU·k | 2π·10¹⁰ m⁻¹ |

## 2. Entrada: campo del driver (Field.py)

| Magnitud | Qué es | Unidades |
|---|---|---|
| `I_W__cm2` (toml) → `Field.I` | irradiancia de pico | W/cm² → W/m² |
| `lambda` (toml) → `Field.lambda0` | longitud de onda | nm → m |
| `chi`, `ellip`, `varphi` | inclinación, elipticidad, CEP | rad, adimensional, rad |
| `Field.E` | campo eléctrico, E₀ = √(2I/cε₀) | V/m |
| `Field.A` | potencial vector, E = −∂A/∂t | V·s/m |
| `Field.dt`, `t` | paso y malla temporal | s |

## 3. Dentro de la evolución (TBevolution)

| Magnitud | Qué es | Unidades |
|---|---|---|
| `self.h_Rnm` | `crystal.h_Rnm · eV` | J |
| `self.r_Rnm` | `crystal.r_Rnm · 10⁻¹⁰` | m |
| `Ax, Ay, Az` (`_field_components`) | A proyectado sobre x, y, z y dividido por RLU | V·s (intermedia) |
| a(t) = (q/ħ)·Ax | desplazamiento κ = k − a en `_fourier` | Å⁻¹ (sin 2π) |
| −qE·r | acoplamiento dipolar en M_nm | J |
| −i dt·H/ħ | paso del Runge–Kutta | adimensional |

Frontera entre sistemas: k, R y a están en unidades de red (solo aparecen en la fase exp(2πi (k−a)·R));
todo lo demás está en SI. Se convierte en dos sitios: `__init__` (h: eV → J, r: Å → m) y
`_field_components` (A / RLU).

## 4. Salida: velocidades y corriente

| Magnitud | Qué es | Unidades |
|---|---|---|
| `_velocity_k` → v_k | velocidad en cada punto k | m/s |
| `_velocity` → Σ_k dV v_k | suma en la zona de Brillouin | Å⁻ᵈⁱᵐ·m/s |
| `rk_dipole_velocity` → vd | g_s·q·cell_size·Σ dV v_k: velocidad dipolar **por celda** | C·m/s |
| j = vd/(cell_size·10⁻²⁰) | densidad de corriente; superficial en 2D (¹) | A/m (A/m² en 3D) |
| fichero `*_dipole_velocity.txt` | columnas t, vd_x, vd_y | s, C·m/s |

(¹) `cell_size` está en Å² y 1 Å² = (10⁻¹⁰ m)² = 10⁻²⁰ m². En general el factor es 10^(−10·dim): 10⁻³⁰ en 3D.

## 5. Notebook: espectro

| Magnitud | Qué es | Unidades |
|---|---|---|
| `dipole_vel_x`, `vd_x` | vd leída (y multiplicada por la máscara) | C·m/s |
| `fft(vd)*dt` | FT de vd ≈ ∫ vd e^{iωt} dt | C·m |
| `FT_accel_x` = iω·FT(vd), conjugada | FT de la aceleración dipolar d̈ | C·m/s |
| `omega` | frecuencia angular de cada bin | rad/s |
| `T0` = λ/c | periodo del driver | s |
| `w` = ω T₀/2π | orden armónico (variable continua) | adimensional |
| dq = `w[1]−w[0]` = T₀/duración | anchura de bin (1/8 con 8 periodos) | adimensional |
| `spectrum` = \|d̈_x\|² + \|d̈_y\|² | densidad espectral | C²·m²·s⁻² |

## 6. Notebook: tabla `harm`

| Magnitud | Qué es | Unidades |
|---|---|---|
| S0…S3 por bin | Stokes de cada frecuencia | C²·m²·s⁻² |
| `I` = Σ S0·dq = ∫ \|d̈\|² dw, de q−Δ/2 a q+Δ/2 | área del pico del armónico q | C²·m²·s⁻² |
| `S1, S2, S3` | Stokes normalizados (S0 = 1), promedio ponderado con \|FT\|² | adimensional |
| `P`, `eps` | grado de polarización, elipticidad | adimensional |
| `chi`, `phi`, `delta_varphi` | ángulos de la elipse y de Jones | rad (en las gráficas, grados) |
| `ok` | I_q > I_REL_MIN · máx de su cálculo | booleano |
