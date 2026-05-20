"""
SIR Coverage vs Threshold for Different Satellite Densities (mu)

This script evaluates both:
1. Monte Carlo simulation
2. Analytical theory

for multiple values of the average number of satellites on the orbital sphere.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad

np.random.seed(42)

# =========================================================
# FIXED PARAMETERS
# =========================================================

# Earth / satellite geometry
rE = 6400e3                 # Earth radius [m]
ra = 550e3                  # Satellite altitude [m]
rS = rE + ra                # Orbital radius [m]

# Monte Carlo iterations
n_iter = 15000

# SIR thresholds
gamma_dB = np.linspace(-10, 20, 30)
gamma = 10 ** (gamma_dB / 10)

# Path-loss exponent
alpha = 2

# Interferer antenna gain
G_bar = 0.1

# Different satellite densities to test
mu_values = [50, 100, 190, 300, 500]

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

def laplace_interference(s, r, lambda_):
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

def sir_coverage_theory(tau_value, lambda_):
    P_visible = 1 - np.exp(-lambda_ * area_cap)

    def integrand(r):
        s = tau_value * r**alpha
        return (
            laplace_interference(s, r, lambda_)
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
# MONTE CARLO SIR SIMULATION
# =========================================================

def simulate_SIR(lambda_, tau_array, n_iter):
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

            # Serving satellite = nearest
            idx0 = np.argmin(d_visible)
            d0 = d_visible[idx0]

            # Desired signal
            H0 = np.random.exponential(1)
            signal = H0 * d0 ** (-alpha)

            # Interferers
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
# MAIN LOOP OVER MU
# =========================================================

plt.figure(figsize=(8, 6))

for mu in mu_values:
    print(f"\nProcessing mu = {mu}")

    # Convert mu to PPP density
    lambda_ = mu / (4 * np.pi * rS**2)

    # -------------------------
    # Monte Carlo simulation
    # -------------------------
    sir_sim = simulate_SIR(
        lambda_,
        gamma,
        n_iter
    )

    # -------------------------
    # Theory
    # -------------------------
    sir_theory = np.array([
        sir_coverage_theory(t, lambda_)
        for t in gamma
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
        gamma_dB,
        sir_theory,
        linewidth=2,
        label=rf"Theory ($\mu={mu}$)"
    )

    # -------------------------
    # Plot simulation points
    # -------------------------
    plt.scatter(
        gamma_dB,
        sir_sim,
        s=20,
        alpha=0.7
    )

# =========================================================
# FINAL PLOT SETTINGS
# =========================================================

plt.xlabel(r"SIR threshold $\gamma$ [dB]", fontsize=12)
plt.ylabel(r"$P(\mathrm{SIR} > \gamma)$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig("SIR_coverage_vs_mu.pdf")
plt.show()
