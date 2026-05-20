"""
SNR Coverage vs Threshold for Different Satellite Altitudes (ra)

This script compares:
1. Monte Carlo simulation
2. Correct analytical theory

for multiple satellite altitudes while keeping the average number
of satellites on the orbital sphere (mu) fixed.
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

# Satellite altitudes to test [m]
ra_values = [300e3, 550e3, 800e3, 1200e3, 2000e3]

# Average number of satellites on the orbital sphere
mu = 190

# Monte Carlo iterations
n_iter = 15000   # Increase to 10000 for higher accuracy

# Thresholds
gamma_dB = np.linspace(-10, 20, 30)
gamma = 10 ** (gamma_dB / 10)

# Path-loss exponent
alpha = 2

# =========================================================
# PHYSICAL PARAMETERS
# =========================================================

Pt = 10 ** (30 / 10) / 1000     # 1 W
g = 10 ** (20 / 10)             # 100

f = 2e9
c = 3e8
K = (c / (4 * np.pi * f)) ** 2

k = 1.38e-23
T = 290
B = 1e5                        # 100 kHz
sigma2 = k * T * B

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
# MONTE CARLO SIMULATION
# =========================================================

def simulate_coverage(lambda_, rS, rE, tau_array, n_iter):
    results = []

    user = np.array([0, 0, rE])
    R_max = np.sqrt(rS**2 - rE**2)

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
            d0 = np.min(d_visible)

            # Rayleigh fading
            H0 = np.random.exponential(1)

            # Received signal power
            signal = Pt * g * K * H0 * d0**(-alpha)

            # SNR
            snr = signal / sigma2

            if snr > t:
                success += 1

        results.append(success / n_iter)

    return np.array(results)


# =========================================================
# NORMALIZATION CONSTANT
# =========================================================

def nu_lambda_RS(lambda_, rS, rE):
    numerator = (
        2 * np.pi * lambda_ * rS / rE
        * np.exp(
            lambda_ * np.pi * rS / rE
            * (rS**2 - rE**2)
        )
    )

    denominator = (
        np.exp(
            2 * lambda_ * np.pi * rS * (rS - rE)
        ) - 1
    )

    return numerator / denominator


# =========================================================
# CONDITIONAL DISTANCE PDF
# =========================================================

def f_R_conditional(r, lambda_, rS, rE, R_min, R_max):
    if r < R_min or r > R_max:
        return 0.0

    nu = nu_lambda_RS(lambda_, rS, rE)

    return (
        nu
        * r
        * np.exp(
            -lambda_ * np.pi * rS / rE * r**2
        )
    )


# =========================================================
# THEORETICAL SNR COVERAGE
# =========================================================

def snr_coverage_theory(tau_value, lambda_, rS, rE):
    R_min = rS - rE
    R_max = np.sqrt(rS**2 - rE**2)
    area_cap = 2 * np.pi * rS * (rS - rE)

    # Probability that at least one visible satellite exists
    P_visible = 1 - np.exp(-lambda_ * area_cap)

    def integrand(r):
        fading_term = np.exp(
            -tau_value * sigma2 * r**alpha
            / (Pt * g * K)
        )

        return fading_term * f_R_conditional(
            r,
            lambda_,
            rS,
            rE,
            R_min,
            R_max
        )

    integral, _ = quad(
        integrand,
        R_min,
        R_max,
        limit=100
    )

    return P_visible * integral


# =========================================================
# MAIN LOOP OVER ALTITUDES
# =========================================================

plt.figure(figsize=(8, 6))

for ra in ra_values:

    print(f"\nProcessing altitude ra = {ra/1e3:.0f} km")

    # Orbital radius
    rS = rE + ra

    # PPP density chosen so average number of satellites remains mu
    lambda_ = mu / (4 * np.pi * rS**2)

    # -------------------------
    # Monte Carlo simulation
    # -------------------------
    snr_sim = simulate_coverage(
        lambda_,
        rS,
        rE,
        gamma,
        n_iter
    )

    # -------------------------
    # Theory
    # -------------------------
    snr_theory = np.array([
        snr_coverage_theory(
            t,
            lambda_,
            rS,
            rE
        )
        for t in gamma
    ])

    # -------------------------
    # Error analysis
    # -------------------------
    max_error = np.max(np.abs(snr_sim - snr_theory))
    print(f"Maximum absolute error = {max_error:.4e}")

    # -------------------------
    # Plot theory
    # -------------------------
    plt.plot(
        gamma_dB,
        snr_theory,
        linewidth=2,
        label=rf"Theory (Altitude$={ra/1e3:.0f}\,\mathrm{{km}}$)"
    )

    # -------------------------
    # Plot simulation points
    # -------------------------
    plt.scatter(
        gamma_dB,
        snr_sim,
        s=20,
        alpha=0.7
    )

# =========================================================
# FINAL PLOT
# =========================================================

plt.xlabel(r"SNR threshold $\gamma$ [dB]", fontsize=12)
plt.ylabel(r"$P(\mathrm{SNR} > \gamma)$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig("PPP_SNR_vs_altitude.pdf")
plt.show()
