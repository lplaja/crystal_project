"""
Field.py -- polarized laser pulses for the tight-binding time evolution (TBevolution).

Convention (lecture slides "La elipse de polarización")
-------------------------------------------------------
    E(t) = Re[ A env(t) e_sigma e^{-i(w0 t + varphi)} ]
    e_sigma = cos(phi) e_par + e^{-i delta_varphi} sin(phi) e_perp

    varphi        absolute phase of the carrier (CEP)
    phi           Jones-vector angle  (A_par = A cos phi, A_perp = A sin phi)
    delta_varphi  phase delay of the perpendicular component (varphi_perp = varphi + delta_varphi)
    chi           tilt of the ellipse major axis with respect to e_par, in (-pi/2, pi/2]
    eps           ellipticity eps = tan(psi) = ±b/a, with sin(2 psi) = sin(2 phi) sin(delta_varphi)
                  eps > 0 right-handed (the field rotates from e_par towards -e_perp), eps < 0
                  left-handed, eps = 0 linear, |eps| = 1 circular

Output: E and A are real arrays (n_t, 3) with columns (parallel, perpendicular, axial).
TBevolution converts them to Cartesian components with polarization_frame(s_direction).

Units: SI internally (W/m^2, m, rad/s, s, V/m, V·s/m). Constructor inputs carry the unit
in their name (I_W__cm2 in W/cm^2, lambda0_nm in nm, angles in rad).
"""
import numpy as np
from grid import UniformCartesianGrid, FonGrid
from scipy.constants import epsilon_0, mu_0, nano, femto, micro, centi
from scipy.constants import c as c_light
from scipy.integrate import cumulative_trapezoid

### HELPERS
def lambda_to_w(lambda0):
    """Angular frequency w = 2 pi c / lambda0.  lambda0 in m -> w in rad/s."""
    return 2*np.pi*c_light/lambda0

def w_to_lambda(w):
    """Wavelength lambda = 2 pi c / w.  w in rad/s -> lambda in m."""
    return 2*np.pi*c_light/w

def lambda_to_T(lambda0):
    """Optical period T = lambda0 / c.  lambda0 in m -> T in s."""

    return lambda0/c_light

def T_to_lambda(T):
    """Wavelength lambda = c T.  T in s -> lambda in m."""
    return c_light*T

def I_to_E(I):
    """Field amplitude E0 from the mean intensity, I = c eps0 E0^2 / 2.

    I in W/m^2 -> E0 in V/m. Since |e_sigma| = 1, E0 is the amplitude for any polarization:
    for linear polarization it is the peak value; for circular each component is E0/sqrt(2).
    """
    return np.sqrt(I/(c_light*epsilon_0/2))

def E_to_I(E):
    """Mean intensity I = c eps0 E^2 / 2 (inverse of I_to_E).  E in V/m -> I in W/m^2."""
    return E**2*(c_light*epsilon_0/2)

def ellipse_to_jones(chi, eps):
    """Jones-vector parameters from the polarization ellipse.

    Lecture slide "Parámetros del vector de polarización a partir de la elipse".

    Parameters
    ----------
    chi : float or array
        Tilt of the major axis with respect to e_par (rad). Any value: it is reduced to
        (-pi/2, pi/2], since chi and chi + n pi describe the same ellipse.
    eps : float or array
        Ellipticity eps = tan(psi) = ±b/a in [-1, 1]; eps > 0 right-handed.

    Returns
    -------
    phi : Jones angle in [-pi/2, pi/2], with the sign of chi (sign(0) = +1).
    delta_varphi : phase delay in [-pi/2, pi/2].

    Notes
    -----
    phi = sign(chi)/2 arccos(cos 2psi cos 2chi),
    delta_varphi = atan2(sign(chi) sin 2psi, cos 2psi |sin 2chi|).
    Branch-free form, no singularity at chi = ±pi/4. Inverse: jones_to_ellipse.
    """
    chi = np.pi/2 - np.mod(np.pi/2 - chi, np.pi)          # chi in (-pi/2, pi/2]
    sgn = np.where(chi >= 0, 1.0, -1.0)                   # sign(0) = +1
    cos2psi = (1 - eps**2)/(1 + eps**2)
    sin2psi = 2*eps/(1 + eps**2)
    # clip: round-off could leave the argument at 1+1e-16 and arccos would return NaN
    phi = 0.5*sgn*np.arccos(np.clip(cos2psi*np.cos(2*chi), -1, 1))
    delta_varphi = np.arctan2(sgn*sin2psi, cos2psi*np.abs(np.sin(2*chi)))
    return phi, delta_varphi

def jones_to_ellipse(phi, delta_varphi):
    """Polarization-ellipse parameters from the Jones vector.

    Lecture slide "Parámetros de la elipse" (inverse of ellipse_to_jones).

    Parameters
    ----------
    phi, delta_varphi : float or array
        e_sigma = cos(phi) e_par + exp(-i delta_varphi) sin(phi) e_perp. Any value.

    Returns
    -------
    chi : tilt of the major axis in (-pi/2, pi/2]. For circular light (|eps| = 1) it is
        undefined and the returned value is arbitrary.
    eps : ellipticity eps = tan(psi) = ±b/a in [-1, 1]; eps > 0 right-handed.

    Notes
    -----
    chi = 1/2 atan2(sin 2phi cos delta_varphi, cos 2phi),  sin 2psi = sin 2phi sin delta_varphi.
    """
    chi = 0.5*np.arctan2(np.sin(2*phi)*np.cos(delta_varphi), np.cos(2*phi))
    eps = np.tan(0.5*np.arcsin(np.clip(np.sin(2*phi)*np.sin(delta_varphi), -1, 1)))
    return chi, eps

#### ENVELOPES
# Envelope parameters are FRACTIONS of the time-grid length tmax = t[-1] + dt
# (the grid does not include its end point):
#   start, end : pulse start and end;   ton, toff : duration of the rise and fall ramps.
# With the default values the pulse spans the whole grid.

def env_sin2(t:np.array, parameters: dict={'start':0, 'end':1, 'ton':0.5, 'toff':0.5}):
    """Envelope with sin^2 rise and cos^2 fall ramps, 1 in between and 0 outside.

    Parameters
    ----------
    t : array (n_t,)
        Uniform time grid (s).
    parameters : dict
        {'start', 'end', 'ton', 'toff'} as fractions of tmax (see the ENVELOPES comment).
        With ton = toff = 0.5, start = 0, end = 1 it gives sin^2(pi t / tmax).

    Returns
    -------
    env : array (n_t,) with values in [0, 1].
    """
    dt=t[1]-t[0]
    tmax=t[-1]+dt

    t0=parameters['start']*tmax
    t1=parameters['end']*tmax
    ton=parameters['ton']*tmax
    toff=parameters['toff']*tmax

    env=np.ones(t.shape[0])
    env[t<t0]=0
    env[t>t1]=0
    mask = (t >= t0) & (t < t0 + ton)                                # rise ramp
    env[mask]=np.sin((t[mask]-t0)/ton*np.pi/2)**2
    mask = (t < t1) & (t >= t1-toff)                                 # fall ramp
    env[mask]=np.cos((t[mask]-(t1-toff))/toff*np.pi/2)**2

    return env

def env_lin(t:np.array, parameters: dict={'start':0, 'end':1, 'ton':0.5, 'toff':0.5}):
    """Trapezoidal envelope: linear rise and fall ramps, 1 in between and 0 outside.

    Same arguments and units as env_sin2.
    """
    dt=t[1]-t[0]
    tmax=t[-1]+dt

    t0=parameters['start']*tmax
    t1=parameters['end']*tmax
    ton=parameters['ton']*tmax
    toff=parameters['toff']*tmax

    env=np.ones(t.shape[0])
    env[t<t0]=0
    env[t>t1]=0
    mask = (t >= t0) & (t < t0 + ton)                                # rise ramp
    env[mask]=(t[mask]-t0)/ton
    mask = (t < t1) & (t >= t1-toff)                                 # fall ramp
    env[mask]=-(t[mask]-(t1-toff))/toff+1

    return env

class PulsedField():
    """Polarized laser pulse: electric field E(t) and vector potential A(t), with E = -dA/dt.

    Parameters
    ----------
    time_s : UniformCartesianGrid
        1D time grid (s).
    I_W__cm2 : float
        Mean intensity of the pulse in W/cm^2 (>= 0; I = 0 gives a null field).
    lambda0_nm : float
        Central wavelength in nm (> 0).
    varphi_rad : float
        Carrier phase (CEP), rad. Refers to E if field_type='E' and to A if field_type='A'.
    env : callable, optional
        Envelope env(t) or env(t, env_parameters), e.g. env_sin2. None: envelope equal to 1.
    env_parameters : dict, optional
        Envelope parameters (see ENVELOPES).
    chi_rad : float
        Tilt of the ellipse with respect to e_par, rad. Stored reduced to (-pi/2, pi/2].
    ellip : float
        Ellipticity eps = tan(psi) = ±b/a in [-1, 1]; > 0 right-handed, < 0 left-handed.
    theta_rad : float
        Split between the transverse plane and the axial component (column 2): the
        (par, perp) components are multiplied by cos(theta) and the axial one is sin(theta).
    s_direction_cartesian : array (3,)
        Direction s in Cartesian coordinates. PulsedField only stores it; TBevolution uses it
        in polarization_frame to map (par, perp, axial) to (x, y, z).
    field_type : {'E', 'A'}
        Quantity the envelope is applied to (the "given" one):
        'E': E = env Re[E0 ...] and A = -int E dt  (trapezoidal rule);
        'A': A = env Re[A0 ...], A0 = -E0/w0, and E = -dA/dt  (np.gradient).
        With an envelope both cases are NOT the same pulse: with 'A', int E dt = 0 holds.

    Attributes
    ----------
    E, A : real arrays (n_t, 3), columns (parallel, perpendicular, axial). V/m and V·s/m.
    t : array (n_t,) in s.   dt : time step (s).   nt : number of points.
    I (W/m^2), lambda0 (m), w0 (rad/s), T0 (s), varphi, chi, ellip, theta, s_direction.
    phi, delta_varphi : Jones parameters computed with ellipse_to_jones.
    envelope : array (n_t,).   complex_carrier : e^{-i(w0 t + varphi)}, array (n_t,).
    amplitude_parallel, amplitude_perp, amplitude_axial : complex Jones amplitudes
        (in V/m if field_type='E', in V·s/m if 'A').
    """
    def __init__(self,time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float,
                 varphi_rad:float=0, env:callable=None, env_parameters:dict=None, chi_rad:float=0,
                 ellip:float=0, theta_rad:float=0, s_direction_cartesian=np.array([0,0,1], dtype=float),field_type:str='E'):

        # --- input checks: these cases raise no numerical error but give a meaningless field
        if field_type !='E' and  field_type !='A':
            raise ValueError("field_type must be 'E' for electric field or 'A' for the magnetic vector potential")
        if not -1 <= ellip <= 1:
            # |eps| > 1 would silently be reinterpreted as the ellipse with its axes swapped
            raise ValueError(f"ellip = b/a must be in [-1, 1], got {ellip}")
        if I_W__cm2 < 0 or lambda0_nm <= 0:
            raise ValueError("I_W__cm2 must be >= 0 and lambda0_nm > 0")


        self.field_type=field_type

        # --- time grid and pulse parameters (SI)
        self.t=time_s.x[:,0]
        self.dt=time_s.dx[0]
        self.nt=time_s.npts
        self.I=I_W__cm2/centi**2                                       # W/cm^2 -> W/m^2
        self.lambda0=lambda0_nm*nano                                   # nm -> m
        self.env=env
        self.env_parameters=env_parameters

        self.w0=lambda_to_w(self.lambda0)
        self.T0=lambda_to_T(self.lambda0)
        self.varphi=varphi_rad

        # carrier with the lecture-slide convention: e^{-i(w0 t + varphi)}
        self.complex_carrier=np.exp(-1j*self.w0*self.t-1j*self.varphi)
        if env is not None:
            if env_parameters is None:
                self.envelope=self.env(self.t)
            else:
                self.envelope=self.env(self.t, self.env_parameters)
        else:
            self.envelope=np.ones(time_s.npts)

        # --- polarization
        self.chi = np.pi/2 - np.mod(np.pi/2 - chi_rad, np.pi)      # (-pi/2, pi/2]
        self.ellip=ellip
        self.theta=theta_rad
        self.s_direction=s_direction_cartesian

        self.phi, self.delta_varphi = ellipse_to_jones(self.chi, self.ellip)

        # amplitude of the given quantity; with A0 = -E0/w0, E = -dA/dt has amplitude E0 (slow envelope)
        if field_type=='E':
            amplitude= I_to_E(self.I)
        else:
            amplitude= -I_to_E(self.I)/self.w0

        # complex Jones amplitudes (par, perp) and axial component
        self.amplitude_parallel=amplitude*np.cos(self.phi)*np.cos(self.theta)
        self.amplitude_perp=amplitude*np.sin(self.phi)*np.exp(-1j*self.delta_varphi)*np.cos(self.theta)
        self.amplitude_axial=amplitude*np.sin(self.theta)

        field=self.envelope[:,np.newaxis]* \
              np.column_stack((self.amplitude_parallel*self.complex_carrier,
                               self.amplitude_perp*self.complex_carrier,
                               self.amplitude_axial*self.complex_carrier))
        # real part taken ONCE, before integrating or differentiating: E and A must be real
        # (TBevolution uses them in -qE·r and in kappa = k - qA/hbar)
        field=np.real(field)

        if field_type=='E':
            self.E=field # electric field
            self.A=-cumulative_trapezoid(field, dx=self.dt, axis=0, initial=0)   # A(t0) = 0
        else:
            self.A=field
            self.E=-np.gradient(field, self.dt, axis=0)    # centred; first order at the end points

    def __repr__(self):
        """Parameter summary; every line starts with '#' so it can be used in file headers."""
        env_name = self.env.__name__ if self.env is not None else None
        info  = f"# {self.__class__.__name__}: field_type={self.field_type} \n"
        info += f"# \t I={self.I:.3e} W/m^2 \t lambda0={self.lambda0:.4e} m \t varphi={self.varphi/np.pi:.4e} π rad \n"
        info += f"# \t env={env_name} \t env_parameters={self.env_parameters} \n"
        info += f"# \t chi={self.chi/np.pi:.4e} π rad \t ellip={self.ellip:.4e} \n"
        info += f"# \t phi={self.phi/np.pi:.4e} π rad \t delta_varphi={self.delta_varphi/np.pi:.4e} π rad \n"
        info += f"# \t theta={self.theta/np.pi:.4e} π rad \t s_direction={self.s_direction} \n"
        info += f"# \n"
        return info
