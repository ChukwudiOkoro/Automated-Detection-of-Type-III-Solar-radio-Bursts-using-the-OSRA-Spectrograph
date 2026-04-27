import numpy as np
from scipy.optimize import fsolve

# --- Constants ---
C_LIGHT = 2.998e10          # Speed of light [cm/s]
R_SUN   = 6.96e10           # Solar radius [cm]
C_RS    = C_LIGHT / R_SUN   # Normalized speed: how many solar radii light travels per second [R_s/s]

# --- Density Models ---
# These functions return electron density [cm^-3] as a function of radial distance r [solar radii R_s]
# Different models represent different approximations of the solar corona electron density

def saito77(r):
    """
    Saito et al. (1977) density model.
    Valid for ~1-10 R_s.
    Two-power law: steep inner falloff + shallower outer falloff.
    """
    return 1.36e6 * r**-2.14 + 1.68e8 * r**-6.13

def leblanc98(r):
    """
    Leblanc et al. (1998) density model.
    Valid for ~1-5 R_s.
    Three-power law, often used for type III burst modeling.
    """
    return (3.3e5 * r**-2.0 +
            4.1e6 * r**-4.0 +
            8.0e7 * r**-6.0)

def parkerfit(r):
    """
    Parker wind model fit.
    Includes a dense inner corona (r^-14, r^-6), a power-law interplasma (r^-2.3),
    and an exponential term for near-Sun region.
    h1: scale height in R_s (20/960 = 0.0208 R_s ~ 1.45e9 cm)
    """
    h1 = 20.0 / 960.0                # Scale height [R_s]
    nc = 3e11 * np.exp(-(r - 1.0) / h1)  # Exponential core term
    return (4.8e9 / r**14 +
            3e8 / r**6 +
            1.39e6 / r**2.3 +
            nc)

def newkirk(r):
    """
    Newkirk (1961) density model.
    Often used for quiet Sun corona.
    Formula: 4.2e4 * 10^(4.32 / r)
    """
    return 4.2e4 * 10.0**(4.32 / r)

# --- Plasma Frequency ---
def plasma_freq(ne_func, r):
    """
    Angular plasma frequency [rad/s] at radius r for a given density model.
    Formula: ω_pe = 2π × f_pe, where f_pe = 8.93e3 × sqrt(n_e) [Hz]
    """
    return 2 * np.pi * 8.93e3 * np.sqrt(ne_func(r))

def freq_from_density(ne):
    """
    Convert electron density [cm^-3] to plasma frequency [Hz].
    f_pe [Hz] = 8980 * sqrt(n_e [cm^-3])  (approximately 8.93e3)
    """
    return 8.93e3 * np.sqrt(ne)

def density_from_freq(f):
    """
    Convert plasma frequency [Hz] to electron density [cm^-3].
    Inverse of freq_from_density().
    """
    return (f / 8.93e3) ** 2

# --- Radius ↔ Frequency Conversion ---
def freq_to_radius(f_pe, model=parkerfit, r_init=1.1):
    """
    Find the solar radius [R_s] where the plasma frequency equals a given value.
    
    Parameters:
    f_pe  : target plasma frequency [Hz]
    model : electron density model function
    r_init: initial guess for radius [R_s] (default 1.1)
    
    Returns:
    radius [R_s] where plasma frequency matches f_pe
    """
    # Equation: f_pe - plasma_freq(model, r)/(2π) = 0
    equation = lambda r: f_pe - plasma_freq(model, r) / (2 * np.pi)
    return fsolve(equation, r_init)

def radius_to_freq(r, model=parkerfit):
    """
    Calculate plasma frequency [Hz] at a given radius.
    Inverse of freq_to_radius().
    """
    return plasma_freq(model, r) / (2 * np.pi)

# --- Drift Models ---
# These functions model frequency drift over time for coronal radio bursts
# (e.g., type III bursts caused by electron beams traveling outward)

def drift_freq_from_time(t, v, t0, model=parkerfit):
    """
    Calculate frequency [MHz] at a given observation time for a drifting burst.
    
    Physical picture: An electron beam travels outward at speed v through the corona.
    Each frequency f corresponds to a specific plasma layer (where f = f_pe(r)).
    As the beam moves outward, it excites progressively lower frequencies.
    
    Parameters:
    t    : time [seconds]
    v    : beam speed in units of speed of light (c)
    t0   : reference time at which the burst is at 300 MHz
    model: density model used for the corona
    
    Returns:
    frequency [MHz] at time t
    """
    # Find radius where f_pe = 300 MHz
    r0 = freq_to_radius(300e6, model)
    
    # Distance traveled by the beam: Δr = v·c·Δt [R_s]
    # Note: v is in units of c, so actual speed = v * C_LIGHT
    # C_RS = C_LIGHT / R_SUN, so v * C_RS = (v * C_LIGHT) / R_SUN [R_s/s]
    delta_r = (t - t0) * v * C_RS
    
    # New radius = starting radius + movement
    r_new = r0 + delta_r
    
    # Convert radius back to frequency [MHz]
    return radius_to_freq(r_new, model) / 1e6

def drift_time_from_freq(f, v, t0, model=parkerfit):
    """
    Calculate expected arrival time for a given frequency in a drifting burst.
    Inverse of drift_freq_from_time().
    
    Parameters:
    f    : frequency [MHz]
    v    : beam speed in units of c
    t0   : reference time at 300 MHz
    model: density model
    
    Returns:
    time [seconds] when frequency f is observed
    """
    # Radius corresponding to target frequency
    r_target = freq_to_radius(f * 1e6, model)
    
    # Radius corresponding to reference frequency (300 MHz)
    r_ref = freq_to_radius(300e6, model)
    
    # Time delay = distance / speed, where speed = v * C_RS [R_s/s]
    delta_t = (r_target - r_ref) / (v * C_RS)
    
    return t0 + delta_t

# --- Optional: Derivative for Leblanc Model ---
def dndr_leblanc98(r):
    """
    Derivative of Leblanc et al. (1998) density model with respect to radius.
    Useful for calculating density gradients or drift rates analytically.
    d(r^n)/dr = n * r^(n-1)
    """
    return (-2.0 * 3.3e5 * r**-3.0    # derivative of r^-2 term
            -4.0 * 4.1e6 * r**-5.0    # derivative of r^-4 term
            -6.0 * 8.0e7 * r**-7.0)   # derivative of r^-6 term