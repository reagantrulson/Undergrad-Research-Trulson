import numpy as np
from scipy.integrate import solve_ivp
# constants
MU_EARTH= 398600.4418  # km^3/s^2
R_EARTH  = 6371 # km


# initial ISS conditions as of 5/27/26 - https://in-the-sky.org/spacecraft_elements.php?id=25544
ISS_ELEMENTS = {
    "e": 0.0007, # eccentricity
    "M": np.radians(256.517), # mean anomaly
    "a": float(6783), # semi-major axis in km
    "omega": np.radians(103.664), # argument of perigee
    "i": np.radians(51.634), # inclination
    "RAAN":  2.8388 * np.pi / 12,  # rad  right ascension of ascending node
}

# identical 2nd orbit but trailing by 15 min
TRAILING_M = np.deg2rad(15.0 / 92.0 * 360.0)  # ~58.7°
DEPUTY_ELEMENTS = {**ISS_ELEMENTS, "M": ISS_ELEMENTS["M"] + TRAILING_M}

# another set up that is not just a time phase shift
# Satellite 2 - different orbit
SAT2_ELEMENTS = {
    "e": 0.0012,                        # slightly more eccentric
    "M": np.radians(256.517),           # same starting mean anomaly
    "a": float(7000),                   # higher altitude
    "omega": np.radians(145.0),         # different argument of perigee
    "i": np.radians(56.0),              # different inclination
    "RAAN": 3.2 * np.pi / 12,          # different RAAN
}

def solve_kepler(e, M, tol=1e-12, max_iter=50): # returns eccentric anomaly E in radians
    E = M # initial guess
    for i in range(max_iter):
        # Newton's method
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        # check for convergence
        if abs(E_new - E) < tol:
            return E_new
        E = E_new
    return E

def eccentric_to_true(E, e):
    return np.arctan2(  # use E to find v, but make sure to stay in correct quadrant
        np.sin(E) * np.sqrt(1 - e ** 2),
        np.cos(E) - e
    )


def orbital_elements(a, e, M, i, RAAN, omega, t=0.0, mu = MU_EARTH):
    n = np.sqrt(mu / a ** 3) # mean motion (rad/s)
    M_n = M + n * t   # mean anomaly at time t
    E = solve_kepler(e, M_n) # eccentric anomaly
    v = eccentric_to_true(E, e) # true anomaly

    p = a * (1-e**2)    # semi-latus rectum
    r = a * (1 - e * (np.cos(E))) # orbital radius

    # perifocal frame state vectors
    r_p = np.array([r * np.cos(v), r * np.sin(v), 0.0])
    v_p = np.sqrt(mu / p) * np.array([-np.sin(v), e + np.cos(v), 0.0])

    # complete 3 step rotation
    R = rotate_orbit(RAAN, i, omega)
    r_n = R @ r_p
    v_n = R @ v_p

    return r_n, v_n


def rotate_orbit(RAAN, i, omega):
    # rotation matrix from perifocal to earth centered (ECI) frame
    cO, sO = np.cos(RAAN), np.sin(RAAN) # rotate plane around Earth
    c1, s1 = np.cos(i), np.sin(i) # tilt orbital plane to correct inclination
    c2, s2 = np.cos(omega), np.sin(omega) # rotate perigee to correct place in orbital plane

    # R3(RAAN) - z-axis rotation
    R3_O = np.array([
        [cO, -sO, 0],
        [sO, cO, 0],
        [0, 0, 1]
    ])

    # R1(i) - x-axis rotation
    R1_i = np.array([
        [1, 0, 0],
        [0, c1, -s1],
        [0, s1, c1]
    ])

    # R3(omega) - z-axis rotation
    R3_o = np.array([
        [c2, -s2, 0],
        [s2, c2, 0],
        [0, 0, 1]
    ])

    return R3_O @ R1_i @ R3_o


def two_body_ode(t, state, mu=MU_EARTH): # return time derivative of state
    r = state[:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r) # how far satellite is from Earth's center
    acc = -mu / r_norm ** 3 * r # 2 body equation of motion
    return np.concatenate([v, acc])

def propagate_orbit(r0, v0, t_span, t_eval, mu=MU_EARTH):
    state_0 = np.concatenate([r0, v0])
    sol = solve_ivp(
        two_body_ode,
        t_span,
        state_0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        args=(mu,),
    )
    return sol.y[:3].T, sol.y[3:].T  # (N,3) each


def twob_scenario(n_points=1000, mu=MU_EARTH): # ISS-like satellite and 2nd trailing for one full orbital period
    # set initial conditions
    el = ISS_ELEMENTS
    T = 2 * np.pi * np.sqrt(el["a"] ** 3 / mu)  # orbital period, keplers third law
    t_span = (0.0, T)
    t_eval = np.linspace(0.0, T, n_points) # create time grid

    # initial ECI states at t=0
    r1_0, v1_0 = orbital_elements(**el, t=0.0, mu=mu) # satellite 1
    r2_0, v2_0 = orbital_elements(**SAT2_ELEMENTS, t=0.0, mu=mu) # satellite 2

    r1, v1 = propagate_orbit(r1_0, v1_0, t_span, t_eval, mu)
    r2, v2 = propagate_orbit(r2_0, v2_0, t_span, t_eval, mu)

    return t_eval, r1, v1, r2, v2, T


def print_orbit_summary(r, v, label="Satellite", mu=MU_EARTH):
    # print key orbital parameters
    r0, v0 = r[0], v[0]
    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    energy = 0.5 * v_mag ** 2 - mu / r_mag  # specific orbital energy
    a_check = -mu / (2 * energy)  # semi-major axis
    h_vec = np.cross(r0, v0)  # angular momentum
    e_vec = np.cross(v0, h_vec) / mu - r0 / r_mag
    e_check = np.linalg.norm(e_vec)

    print(f"\n{label}:")
    print(f"  |r|         = {r_mag:.2f} km")
    print(f"  |v|         = {v_mag:.4f} km/s")
    print(f"  a (derived) = {a_check:.2f} km")
    print(f"  altitude    = {(r_mag - R_EARTH):.2f} km")



if __name__ == "__main__":
    print("Orbital Setup")

    t, r1, v1, r2, v2, T = twob_scenario(n_points=500)

    print(f"\nOrbital period T = {T:.1f} s  ({T / 60:.2f} min)")
    print(f"Time steps: {len(t)},  dt = {t[1] - t[0]:.2f} s")

    print_orbit_summary(r1, v1, "Satellite 1   (ISS)")
    print_orbit_summary(r2, v2, "Satellite 2  (trailing)")

    # kepler solver convergence demo
    print("\n--- Kepler solver convergence (M=1.0 rad, e=0.3) ---")
    M_test, e_test = 1.0, 0.3
    E_sol = solve_kepler(e_test, M_test)
    residual = E_sol - e_test * np.sin(E_sol) - M_test
    print(f"  E = {E_sol:.10f} rad")
    print(f"  Residual |E - e·sin(E) - M| = {abs(residual):.2e}  (should be < 1e-12)")






