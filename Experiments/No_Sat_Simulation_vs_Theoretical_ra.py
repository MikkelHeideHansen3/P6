import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

# =========================
# PARAMETERS
# =========================

# Earth radius [m]
rE = 6400e3

# Satellite altitudes to compare [m]
ra_values = [300e3, 550e3, 800e3, 1200e3, 2000e3]

# Range of average number of satellites (x-axis)
mu_values = np.linspace(60, 300, 20)

# Monte Carlo iterations
n_iter = 15000

# =========================
# SAMPLE PPP ON SPHERE
# =========================

def sample_satellites(lambda_, rS):
    N = np.random.poisson(4 * np.pi * rS**2 * lambda_)

    theta = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(-1, 1, N)
    phi = np.arccos(u)

    x = rS * np.sin(phi) * np.cos(theta)
    y = rS * np.sin(phi) * np.sin(theta)
    z = rS * np.cos(phi)

    return np.vstack((x, y, z)).T


# =========================
# MONTE CARLO: P(Phi(A)=0)
# =========================

def simulate_no_satellite(lambda_, rS, rE, n_iter):
    count = 0
    user = np.array([0, 0, rE])

    # Maximum visible distance
    R_max = np.sqrt(rS**2 - rE**2)

    for _ in range(n_iter):
        sats = sample_satellites(lambda_, rS)

        # No satellites at all
        if len(sats) == 0:
            count += 1
            continue

        # Distances from user to all satellites
        dists = np.linalg.norm(sats - user, axis=1)

        # Visible satellites
        visible_mask = dists <= R_max

        # Event Phi(A) = 0
        if np.sum(visible_mask) == 0:
            count += 1

    return count / n_iter


# =========================
# PLOT
# =========================

plt.figure(figsize=(8, 6))

for ra in ra_values:

    print(f"\nProcessing altitude ra = {ra/1e3:.0f} km")

    # Orbital radius
    rS = rE + ra

    # Visible spherical cap area
    A_cap = 2 * np.pi * rS**2 * (1 - rE / rS)

    mc_results = []
    theory_results = []
    abs_errors = []

    for mu in tqdm(mu_values, leave=False):

        # PPP density
        lambda_ = mu / (4 * np.pi * rS**2)

        # Monte Carlo simulation
        p_mc = simulate_no_satellite(
            lambda_,
            rS,
            rE,
            n_iter
        )
        mc_results.append(p_mc)

        # Theory:
        # P(Phi(A)=0) = exp(-lambda * A_cap)
        p_theory = np.exp(-lambda_ * A_cap)
        theory_results.append(p_theory)

        # Absolute error
        abs_errors.append(abs(p_mc - p_theory))

    mc_results = np.array(mc_results)
    theory_results = np.array(theory_results)

    # Maximum error for this altitude
    print(f"Maximum absolute error = {np.max(abs_errors):.4e}")

    # Plot theory curve
    plt.plot(
        mu_values,
        theory_results,
        linewidth=2,
        label=rf"Theory ($r_a={ra/1e3:.0f}\,\mathrm{{km}}$)"
    )

    # Plot simulation points
    plt.scatter(
        mu_values,
        mc_results,
        s=20,
        alpha=0.7
    )

# =========================
# FINAL PLOT SETTINGS
# =========================

plt.xlabel(r"$\mu$", fontsize=12)
plt.ylabel(r"$P(\Phi(A)=0)$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig("Sim_of_no_sat_prob_for_different_altitudes_mu.pdf")
plt.show()