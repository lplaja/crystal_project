import numpy as np
from grid import UniformCartesianGrid, FonGrid
from scipy.constants import epsilon_0, mu_0, nano, femto, micro, centi
from scipy.constants import c as c_light
from scipy.integrate import cumulative_trapezoid

### HELPERS
def lambda_to_w(lambda0):
    return 2*np.pi*c_light/lambda0

def w_to_lambda(w):
    return 2*np.pi*c_light/w
    
def lambda_to_T(lambda0):
    
    return lambda0/c_light

def T_to_lambda(T):
     return c_light*T

def I_to_E(I):
    return np.sqrt(I/(c_light*epsilon_0/2))

def E_to_I(E):
    return E**2*(c_light*epsilon_0/2)

def ellipse_to_jones(chi, eps):
    # Transparencia "Parámetros del vector de polarización a partir de la elipse"
    # chi: inclinación (rad); eps = tan(psi) = ±b/a (eps>0 dextrógira)
    chi = np.pi/2 - np.mod(np.pi/2 - chi, np.pi)          # chi en (-pi/2, pi/2]
    sgn = np.where(chi >= 0, 1.0, -1.0)                   # sign(0) = +1
    cos2psi = (1 - eps**2)/(1 + eps**2)
    sin2psi = 2*eps/(1 + eps**2)
    phi = 0.5*sgn*np.arccos(np.clip(cos2psi*np.cos(2*chi), -1, 1))
    delta_varphi = np.arctan2(sgn*sin2psi, cos2psi*np.abs(np.sin(2*chi)))
    return phi, delta_varphi

#### ENVELOPES

def env_sin2(t:np.array, parameters: dict={'start':0, 'end':1, 'ton':0.5, 'toff':0.5}):
    dt=t[1]-t[0]
    tmax=t[-1]+dt

    t0=parameters['start']*tmax
    t1=parameters['end']*tmax
    ton=parameters['ton']*tmax
    toff=parameters['toff']*tmax

    env=np.ones(t.shape[0])
    env[t<t0]=0
    env[t>t1]=0
    mask = (t >= t0) & (t < t0 + ton)
    env[mask]=np.sin((t[mask]-t0)/ton*np.pi/2)**2
    mask = (t < t1) & (t >= t1-toff)
    env[mask]=np.cos((t[mask]-(t1-toff))/toff*np.pi/2)**2

    return env

def env_lin(t:np.array, parameters: dict={'start':0, 'end':1, 'ton':0.5, 'toff':0.5}):
    dt=t[1]-t[0]
    tmax=t[-1]+dt

    t0=parameters['start']*tmax
    t1=parameters['end']*tmax
    ton=parameters['ton']*tmax
    toff=parameters['toff']*tmax

    env=np.ones(t.shape[0])
    env[t<t0]=0
    env[t>t1]=0
    mask = (t >= t0) & (t < t0 + ton)
    env[mask]=(t[mask]-t0)/ton
    mask = (t < t1) & (t >= t1-toff)
    env[mask]=-(t[mask]-(t1-toff))/toff+1

    return env

class PulsedField():
    def __init__(self,time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float, 
                 varphi_rad:float=0, env:callable=None, env_parameters:dict=None, chi_rad:float=0, 
                 ellip:float=0, theta_rad:float=0, s_direction_cartesian=np.array([0,0,1], dtype=float),field_type:str='E'):

        if field_type !='E' and  field_type !='A':
            raise ValueError("field_type must be 'E' for electric field or 'A' for the magnetic vector potential")
        if not -1 <= ellip <= 1:
            raise ValueError(f"ellip = b/a must be in [-1, 1], got {ellip}")
        if I_W__cm2 < 0 or lambda0_nm <= 0:
            raise ValueError("I_W__cm2 must be >= 0 and lambda0_nm > 0")
 
        
        self.field_type=field_type

        self.t=time_s.x[:,0]
        self.dt=time_s.dx[0]
        self.nt=time_s.npts
        self.I=I_W__cm2/centi**2
        self.lambda0=lambda0_nm*nano
        self.env=env
        self.env_parameters=env_parameters

        self.w0=lambda_to_w(self.lambda0)
        self.T0=lambda_to_T(self.lambda0)
        self.varphi=varphi_rad

        self.complex_carrier=np.exp(-1j*self.w0*self.t-1j*self.varphi)
        if env is not None:
            if env_parameters is None:
                self.envelope=self.env(self.t)
            else:
                self.envelope=self.env(self.t, self.env_parameters)
        else:
            self.envelope=np.ones(time_s.npts)

        self.chi = np.pi/2 - np.mod(np.pi/2 - chi_rad, np.pi)      # (-pi/2, pi/2]
        self.ellip=ellip
        self.theta=theta_rad
        self.s_direction=s_direction_cartesian

        self.phi, self.delta_varphi = ellipse_to_jones(self.chi, self.ellip)

        if field_type=='E':
            amplitude= I_to_E(self.I)
        else:
            amplitude= -I_to_E(self.I)/self.w0

        self.amplitude_parallel=amplitude*np.cos(self.phi)*np.cos(self.theta)
        self.amplitude_perp=amplitude*np.sin(self.phi)*np.exp(-1j*self.delta_varphi)*np.cos(self.theta)
        self.amplitude_axial=amplitude*np.sin(self.theta)
        
        field=self.envelope[:,np.newaxis]* \
              np.column_stack((self.amplitude_parallel*self.complex_carrier, 
                               self.amplitude_perp*self.complex_carrier, 
                               self.amplitude_axial*self.complex_carrier))
        field=np.real(field)

        if field_type=='E':
            self.E=field # electric field
            self.A=-cumulative_trapezoid(field, dx=self.dt, axis=0, initial=0)
        else:
            self.A=field
            self.E=-np.gradient(field, self.dt, axis=0)
    
    def __repr__(self):
        env_name = self.env.__name__ if self.env is not None else None
        info  = f"# {self.__class__.__name__}: field_type={self.field_type} \n"
        info += f"# \t I={self.I:.3e} W/m^2 \t lambda0={self.lambda0:.4e} m \t varphi={self.varphi/np.pi:.4e} π rad \n"
        info += f"# \t env={env_name} \t env_parameters={self.env_parameters} \n"
        info += f"# \t chi={self.chi/np.pi:.4e} π rad \t ellip={self.ellip:.4e} \n"
        info += f"# \t phi={self.phi/np.pi:.4e} π rad \t delta_varphi={self.delta_varphi/np.pi:.4e} π rad \n"
        info += f"# \t theta={self.theta/np.pi:.4e} π rad \t s_direction={self.s_direction} \n"
        info += f"# \n"
        return info



