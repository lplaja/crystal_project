import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from grid import UniformCartesianGrid, FonGrid
from scipy.constants import epsilon_0, mu_0, nano, femto, micro, centi
from scipy.constants import c as c_light

### HELPERS
def lambda2w(lambda0):
    return 2*np.pi*c_light/lambda0

def w2lambda(w):
    return 2*np.pi*c_light/w
    
def lambda2T(lambda0):
    return lambda0/c_light

def T2lambda(T):
    return c_light*T

def I2E(I):
    return np.sqrt(I/(c_light*epsilon_0/2))

def E2I(E):
    return E**2*(c_light*epsilon_0/2)

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

class HarmonicField():
    def __init__(self, time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float, phi_rad:float=0, 
                 env:callable=None, env_parameters:dict=None):
        self.time=time_s
        self.t=time_s.x
        self.dt=time_s.dx[0]
        self.I=I_W__cm2/centi**2
        self.lambda0=lambda0_nm*nano
        self.env=env
        self.env_parameters=env_parameters

        self.w0=lambda2w(self.lambda0)
        self.T0=lambda2T(self.lambda0)
        self.phi=phi_rad

        self.carrier=None
        if env is not None:
            if env_parameters is None:
                self.envelope=self.env(self.time.x[:,0])
            else:
                self.envelope=self.env(self.time.x[:,0], self.env_parameters)
        else:
            self.envelope=np.ones(self.time.npts)

        self.E=None
        self.A=None

class PolarizedHarmonicField(HarmonicField):
    def __init__(self, time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float, phi_rad:float=0, 
                 env:callable=None, env_parameters=None, chi_rad:float=0, ellip:float=0, theta_rad:float=0, 
                 s_direction_cartesian=np.array([0,0,1], dtype=float)):
        
        super().__init__(time_s, I_W__cm2, lambda0_nm, phi_rad, env, env_parameters)

        self.chi=chi_rad
        self.ellip=ellip
        self.theta=theta_rad
        self.s_direction=s_direction_cartesian

        self.Phi, self.delta_phi=self.ChiEllip2PhiDeltaPhi(self.chi, self.ellip)

        # print(f'vprint in line: 108 In PolarizedHarmonicField--> {self.chi=}')
        # print(f'vprint in line: 108 In PolarizedHarmonicField--> {self.ellip=}')
        # print(f'vprint in line: 108 In PolarizedHarmonicField--> {self.theta=}')
        # print(f'vprint in line: 108 In PolarizedHarmonicField--> {self.Phi=}')
        # print(f'vprint in line: 108 In PolarizedHarmonicField--> {self.delta_phi=}')
        self.E=None # These will be a 3D with E_parallel, E_perp and E_axial as columns.
        self.A=None # These will be a 3D with A_parallel, A_perp and A_axial as columns.

    def ChiEllip2PhiDeltaPhi(self,chi, ellip):
        # returs the polarization parameters phi and delta phi from the polarization ellipse tilt, chi, and ellipticity, ellip

        eps=1e-30
        def sign(a):
            return np.where(a>0, 1, -1)
                
        argument_num=np.sqrt(np.tan(2*chi)**2+ellip**2+eps)*sign(chi)
        argument_denom=np.sqrt(1-ellip**2)
        Phi=0.5*np.atan2(argument_num, argument_denom) 

        Phi=np.where((chi >= -np.pi/2) & (chi <= -np.pi/4), -Phi-np.pi/2, Phi)
        Phi=np.where((chi >= np.pi/4) & (chi <= np.pi/2), -Phi+np.pi/2, Phi)
        
        delta_phi=np.asin(ellip*np.sqrt((1+np.tan(2*chi)**2+eps)/(ellip**2+np.tan(2*chi)**2+eps)))*np.sign(chi-eps)

        return Phi, delta_phi

class ScalarHarmonicElectricField(HarmonicField):
    def __init__(self, time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float, phi_rad:float=0, 
                 env:callable=None, env_parameters=None):
        
        super().__init__(time_s, I_W__cm2, lambda0_nm, phi_rad, env, env_parameters)

        self.E0=I2E(self.I) # in V/m
        self.carrier=self.E0*np.cos(self.w0*time_s.x[:,0]+self.phi)

        self.E=self.envelope*self.carrier
        self.A=-np.cumsum(self.E)*self.dt
    def __repr__(self):
        info=f"# {self.__class__.__name__}: id= {id(self):x} \n"
        info+=f"# \t I={self.I:.3e} W/m$^2$ \t lambda0={self.lambda0:.4e} m \t phi={self.phi/np.pi:.4e} π rad \n"
        info+=f"# \t env={self.env}  \t env_parameters={self.env_parameters}\n"
        info+=f"# \n"
        return info

class ScalarHarmonicPotentialVectorField(HarmonicField):
    def __init__(self, time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float, phi_rad:float=0, 
                 env:callable=None, env_parameters=None):
        
        super().__init__(time_s, I_W__cm2, lambda0_nm, phi_rad, env, env_parameters)

        self.A0=-I2E(self.I)/self.w0 # in  V·s·m−1 or Wb·m−1

        self.carrier=self.A0*np.cos(self.w0*time_s.x[:,0]+self.phi)
        
        self.A=self.envelope*self.carrier

        self.E=-np.gradient(self.A, self.dt)
    def __repr__(self):
        info=f"# {self.__class__.__name__}: id= {id(self):x} \n"
        info+=f"# \t I={self.I:.3e} W/m$^2$ \t lambda0={self.lambda0:.4e} m \t phi={self.phi/np.pi:.4e} π rad \n"
        info+=f"# \t env={self.env}  \t env_parameters={self.env_parameters}\n"
        info+=f"# \n"
        return info

class polarizedHarmonicElectricField(PolarizedHarmonicField):
    def __init__(self, time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float, phi_rad:float=0, 
                env:callable=None, env_parameters=None, chi_rad:float=0, ellip:float=0, theta_rad:float=0, 
                s_direction_cartesian=np.array([0,0,1], dtype=float)):
        
        super().__init__(time_s, I_W__cm2, lambda0_nm, phi_rad, env, env_parameters, chi_rad, ellip, 
                         theta_rad, s_direction_cartesian)

        self.E0=I2E(self.I) # in V/m

        Phi=self.Phi
        delta_phi=self.delta_phi

        # print(f'vprint in line: 180 in Field --> {phi_rad=}')
        # print(f'vprint in line: 180 in Field --> {chi_rad=}')
        # print(f'vprint in line: 180 in Field --> {theta_rad=}')
        # print(f'vprint in line: 180 in Field --> {Phi=}')
        # print(f'vprint in line: 180 in Field --> {delta_phi=}')


        self.E0_parallel=self.E0*np.cos(Phi)*np.cos(theta_rad)
        self.E0_perp=self.E0*np.sin(Phi)*np.exp(-1j*delta_phi)*np.cos(theta_rad)
        self.E0_axial=self.E0*np.sin(theta_rad)
        # print(f'vprint in line: 183 in Field --> {self.E0_parallel=}')
        # print(f'vprint in line: 183 in Field --> {self.E0_perp=}')
        # print(f'vprint in line: 183 in Field --> {self.E0_axial=}')
        
        temp_phase=np.exp(1j*self.w0*time_s.x[:,0]+1j*self.phi)
        E_parallel=np.real(self.E0_parallel*temp_phase)
        E_perp=np.real(self.E0_perp*temp_phase)   
        E_axial=np.real(self.E0_axial*temp_phase)

        self.carrier=np.column_stack((E_parallel, E_perp, E_axial))


        self.E=self.envelope[:,np.newaxis]*self.carrier
        self.A=-np.cumsum(self.E, axis=0)*self.dt

        # print(f'vprint in line: 46 in Field --> {max(self.E[:,0])=}')
        # print(f'vprint in line: 46 in Field --> {max(self.E[:,1])=}')
    def __repr__(self):
        info=f"# {self.__class__.__name__}: id= {id(self):x} \n"
        info+=f"# \t I={self.I:.3e} W/m$^2$ \t lambda0={self.lambda0:.4e} m \t phi={self.phi/np.pi:.4e} π rad \n"
        info+=f"# \t env={self.env}  \t env_parameters={self.env_parameters}\n"
        info+=f"# \t chi={self.chi/np.pi:.3e} π rad  \t ellip={self.ellip:.2e} m \n"
        info+=f"# \t Phi={self.Phi/np.pi:.3e} π rad  \t delta_phi={self.delta_phi/np.pi:.3e} π rad  \n"
        info+=f"# \t theta={self.theta/np.pi:.3e} π rad  \t s_direction={self.s_direction}\n"
        info+=f"# \n"
        return info

    def showAnimation(self, i_t=[0,-1], interval=30, step=1):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # 1. Submuestreo de datos para agilizar el renderizado
        start, end = i_t
        if end == -1: end = len(self.E)
        
        # Tomamos un punto cada 'step' para no saturar el navegador
        data_slice = self.E[start:end:step]
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        limit = np.max(np.abs(data_slice)) * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        
        # Trayectoria de fondo
        ax.plot(self.E[:, 0], self.E[:, 1], color='gray', alpha=0.2, lw=1)

        # Quiver optimizado: pivot='tail' es clave para vectores de campo
        vector = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='#4A6741', pivot='tail')
        dot, = ax.plot([], [], 'ro', markersize=4)

        def init():
            vector.set_UVC(0, 0)
            dot.set_data([], [])
            return vector, dot

        def update(i):
            x, y = data_slice[i, 0], data_slice[i, 1]
            vector.set_UVC(x, y)
            dot.set_data([x], [y])
            return vector, dot

        ani = FuncAnimation(
            fig, update, frames=len(data_slice), 
            init_func=init, interval=interval, blit=True
        )
        
        return ani

class polarizedHarmonicPotentialVectorField(PolarizedHarmonicField):


    def __init__(self, time_s: UniformCartesianGrid, I_W__cm2:float, lambda0_nm:float, phi_rad:float=0, 
                env:callable=None, env_parameters=None, chi_rad:float=0, ellip:float=0, theta_rad:float=0, 
                s_direction_cartesian=np.array([0,0,1], dtype=float)):
        
        super().__init__(time_s, I_W__cm2, lambda0_nm, phi_rad, env, env_parameters, chi_rad, ellip, 
                         theta_rad, s_direction_cartesian)

        self.A0=-I2E(self.I)/self.w0 # in  V·s·m−1 or Wb·m−1

        Phi=self.Phi
        delta_phi=self.delta_phi

        self.A0_parallel=self.A0*np.cos(Phi)*np.cos(self.theta)
        self.A0_perp=self.A0*np.sin(Phi)*np.exp(-1j*delta_phi)*np.cos(self.theta)
        self.A0_axial=self.A0*np.sin(self.theta)

        temp_phase=np.exp(1j*self.w0*time_s.x[:,0]+1j*self.phi)
        A_parallel=np.real(self.A0_parallel*temp_phase)
        A_perp=np.real(self.A0_perp*temp_phase)
        A_axial=np.real(self.A0_axial*temp_phase)

        self.carrier=np.column_stack((A_parallel, A_perp, A_axial))

        self.A=self.envelope[:,np.newaxis]*self.carrier
        self.E=-np.gradient(self.A, self.dt, axis=0)

    def __repr__(self):
        info=f"# {self.__class__.__name__}: id= {id(self):x} \n"
        info+=f"# \t I={self.I:.3e} W/m$^2$ \t lambda0={self.lambda0:.4e} m \t phi={self.phi/np.pi:.4e} π rad \n"
        info+=f"# \t env={self.env}  \t env_parameters={self.env_parameters}\n"
        info+=f"# \t chi={self.chi/np.pi:.3e} π rad  \t ellip={self.ellip:.2e} \n"
        info+=f"# \t Phi={self.Phi/np.pi:.3e} π rad  \t delta_phi={self.delta_phi/np.pi:.3e} π rad  \n"
        info+=f"# \t theta={self.theta/np.pi:.3e} π rad  \t s_direction={self.s_direction}\n"
        info+=f"# \n"
        return info
            






