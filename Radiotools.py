



import numpy as np
from scipy.optimize import fsolve

# --- Constants ---
C_LIGHT = 2.998e10          # [cm/s]
R_SUN   = 6.96e10           # [cm]
C_RS    = C_LIGHT / R_SUN   # normalized speed [R_s / s]

# --- Density Models ---
def saito77(r):
    return 1.36e6 * r**-2.14 + 1.68e8 * r**-6.13

def leblanc98(r):
    return (3.3e5 * r**-2.0 +
            4.1e6 * r**-4.0 +
            8.0e7 * r**-6.0)

def parkerfit(r):
    h1 = 20.0 / 960.0
    nc = 3e11 * np.exp(-(r - 1.0) / h1)
    return (4.8e9 / r**14 +
            3e8 / r**6 +
            1.39e6 / r**2.3 +
            nc)

def newkirk(r):
    return 4.2e4 * 10.0**(4.32 / r)

# --- Plasma Frequency ---
def plasma_freq(ne_func, r):
    """Angular plasma frequency [rad/s]"""
    return 2 * np.pi * 8.93e3 * np.sqrt(ne_func(r))

def freq_from_density(ne):
    """Plasma frequency [Hz] from density"""
    return 8.93e3 * np.sqrt(ne)

def density_from_freq(f):
    """Electron density [cm^-3] from frequency [Hz]"""
    return (f / 8.93e3) ** 2

# --- Radius ↔ Frequency Conversion ---
def freq_to_radius(f_pe, model=parkerfit, r_init=1.1):
    """Find radius [R_s] for a given plasma frequency [Hz]"""
    equation = lambda r: f_pe - plasma_freq(model, r) / (2 * np.pi)
    return fsolve(equation, r_init)

def radius_to_freq(r, model=parkerfit):
    """Frequency [Hz] at radius r"""
    return plasma_freq(model, r) / (2 * np.pi)

# --- Drift Models ---
def drift_freq_from_time(t, v, t0, model=parkerfit):
    """
    Frequency drift f(t)
    t  : time [s]
    v  : speed in units of c
    t0 : reference time at 300 MHz
    """
    r0 = freq_to_radius(300e6, model)
    delta_r = (t - t0) * v * C_RS
    r_new = r0 + delta_r
    return radius_to_freq(r_new, model) / 1e6  # MHz

def drift_time_from_freq(f, v, t0, model=parkerfit):
    """
    Time drift t(f)
    f  : frequency [MHz]
    v  : speed in units of c
    t0 : reference time at 300 MHz
    """
    r_target = freq_to_radius(f * 1e6, model)
    r_ref = freq_to_radius(300e6, model)
    delta_t = (r_target - r_ref) / (v * C_RS)
    return t0 + delta_t

# --- Optional: Derivative for Leblanc Model ---
def dndr_leblanc98(r):
    return (-2.0 * 3.3e5 * r**-3.0
            -4.0 * 4.1e6 * r**-5.0
            -6.0 * 8.0e7 * r**-7.0)