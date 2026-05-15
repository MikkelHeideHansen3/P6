"""
SIR Coverage vs Threshold for Different Interferer Antenna Gains (G_bar)

This script evaluates:
1. Monte Carlo simulation
2. Analytical theory

for multiple values of G_bar, while keeping all other parameters fixed.

G_bar represents the relative antenna gain of interfering satellites.
For example:
    G_bar = 0.01  -> interferers are 20 dB weaker
    G_bar = 0.1   -> interferers are 10 dB weaker
    G_bar = 1.0   -> no attenuation of interferers
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad

np.random.seed(42)

# =========================================================
# FIXED PARAMETERS
# =========================================================

# Earth radius [m]
rE = 6400e3

# Satellite altitude [m]
ra = 550e3
rS = rE + ra

# Average number of satellites on the orbital sphere
mu = 190

# PPP density
lambda_ = mu / (4 * np.pi * rS**2)

# Monte Carlo iterations
n_iter = 5000   # Increase to 10000 for higher accuracy

# SIR thresholds
tau_dB = np.linspace(-10, 20, 30)
tau = 10 ** (tau_dB / 10)

# Path-loss exponent
alpha = 2

# G_bar values to test
G_bar_values = [0.01, 0.03, 0.1, 0.3, 1.0]

# =========================================================
# GEOMETRY
# =========================================================

R_min = rS - rE
R_max = np.sqrt(rS**2 - rE**2)

# Visible spherical cap area
area_cap = 2 * np.pi * rS * (rS - rE)

# =========================================================
# SAMPLE PPP ON SPHERE
# =========================================================

def sample_satellites(lambda_, rS):
    N = np.random.poisson(4 * np.pi * rS**2 * lambda_)

    theta = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(-1, 1, N)
    phi = np.arccos(u)

    x = rS * np.sin(phi) * np.cos(theta)
    y = rS * np.sin(phi) * np.sin(theta)
    z = rS * np.cos(phi)

    return np.vstack((x, y, z)).T


# =========================================================
# NORMALIZATION CONSTANT
# =========================================================

def nu_lambda_RS(lambda_, rS, rE):
    numerator = (
        2 * np.pi * lambda_ * rS / rE
        * np.exp(lambda_ * np.pi * rS / rE * (rS**2 - rE**2))
    )

    denominator = (
        np.exp(2 * lambda_ * np.pi * rS * (rS - rE)) - 1
    )

    return numerator / denominator


# =========================================================
# CONDITIONAL DISTANCE PDF
# =========================================================

def f_R_conditional(r, lambda_, rS, rE):
    if r < R_min or r > R_max:
        return 0.0

    nu = nu_lambda_RS(lambda_, rS, rE)

    return nu * r * np.exp(
        -lambda_ * np.pi * rS / rE * r**2
    )


# =========================================================
# LAPLACE TRANSFORM OF INTERFERENCE
# =========================================================

def laplace_interference(s, r, lambda_, G_bar):
    factor = lambda_ * np.pi * rS / rE

    scale = (s * G_bar) ** (-2 / alpha)

    lower = scale * r**2
    upper = scale * R_max**2

    def integrand(u):
        return 1 - 1 / (1 + u ** (-alpha / 2))

    integral, _ = quad(integrand, lower, upper, limit=200)

    return np.exp(
        -factor * (s * G_bar) ** (2 / alpha) * integral
    )


# =========================================================
# THEORETICAL SIR COVERAGE
# =========================================================

def sir_coverage_theory(tau_value, lambda_, G_bar):
    P_visible = 1 - np.exp(-lambda_ * area_cap)

    def integrand(r):
        s = tau_value * r**alpha
        return (
            laplace_interference(s, r, lambda_, G_bar)
            * f_R_conditional(r, lambda_, rS, rE)
        )

    integral, _ = quad(
        integrand,
        R_min,
        R_max,
        limit=200
    )

    return P_visible * integral


# =========================================================
# MONTE CARLO SIMULATION
# =========================================================

def simulate_SIR(lambda_, tau_array, n_iter, G_bar):
    user = np.array([0, 0, rE])
    results = []

    for t in tqdm(tau_array, leave=False):
        success = 0

        for _ in range(n_iter):
            sats = sample_satellites(lambda_, rS)

            if len(sats) == 0:
                continue

            dists = np.linalg.norm(sats - user, axis=1)

            visible = dists <= R_max
            if not np.any(visible):
                continue

            d_visible = dists[visible]

            # Serving satellite = nearest visible
            idx0 = np.argmin(d_visible)
            d0 = d_visible[idx0]

            # Desired signal
            H0 = np.random.exponential(1)
            signal = H0 * d0 ** (-alpha)

            # Interference
            interferers = np.delete(d_visible, idx0)

            if len(interferers) == 0:
                success += 1
                continue

            Hi = np.random.exponential(1, len(interferers))
            interference = np.sum(
                G_bar * Hi * interferers ** (-alpha)
            )

            SIR = signal / interference

            if SIR > t:
                success += 1

        results.append(success / n_iter)

    return np.array(results)


# =========================================================
# MAIN LOOP OVER G_bar VALUES
# =========================================================

plt.figure(figsize=(8, 6))

for G_bar in G_bar_values:

    print(f"\nProcessing G_bar = {G_bar}")

    # -------------------------
    # Monte Carlo simulation
    # -------------------------
    sir_sim = simulate_SIR(
        lambda_,
        tau,
        n_iter,
        G_bar
    )

    # -------------------------
    # Theory
    # -------------------------
    sir_theory = np.array([
        sir_coverage_theory(
            t,
            lambda_,
            G_bar
        )
        for t in tau
    ])

    # -------------------------
    # Error
    # -------------------------
    max_error = np.max(np.abs(sir_sim - sir_theory))
    print(f"Maximum absolute error = {max_error:.4e}")

    # -------------------------
    # Plot theory
    # -------------------------
    plt.plot(
        tau_dB,
        sir_theory,
        linewidth=2,
        label=rf"Theory ($\bar{{G}}={G_bar}$)"
    )

    # -------------------------
    # Plot simulation points
    # -------------------------
    plt.scatter(
        tau_dB,
        sir_sim,
        s=20,
        alpha=0.7
    )

# =========================================================
# FINAL PLOT SETTINGS
# =========================================================

plt.xlabel(r"SIR threshold $\tau$ [dB]", fontsize=12)
plt.ylabel(r"$P(\mathrm{SIR} > \tau)$", fontsize=12)
plt.title(r"Effect of Interferer Gain $\bar{G}$ on SIR Coverage")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()

#plt.savefig("SIR_coverage_vs_Gbar.pdf")
plt.show()