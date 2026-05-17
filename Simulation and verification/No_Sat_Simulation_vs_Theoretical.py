import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

# =========================
# PARAMETERS
# =========================

# Absolute errors
abs_errors = []

# Geometry
rE = 6400e3
ra = 550e3
rS = rE + ra

# Monte Carlo
n_iter = 15000

# Range of satellites (μ)
mu_values = np.linspace(60, 300, 20)

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
# NO-SATELLITE MONTE CARLO
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

        # Distances to user
        dists = np.linalg.norm(sats - user, axis=1)

        # Visible satellites
        visible_mask = dists <= R_max

        # Number of visible satellites
        N_visible = np.sum(visible_mask)

        # Event Φ(A) = 0
        if N_visible == 0:
            count += 1

    return count / n_iter


# =========================
# RUN SWEEP OVER μ
# =========================

mc_results = []
theory_results = []

# Visible spherical cap area
A_cap = 2 * np.pi * rS**2 * (1 - rE / rS)

print("Running sweep over mu values...")

for mu in tqdm(mu_values, desc="Processing μ"):

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

    # Theoretical probability
    p_theory = np.exp(-lambda_ * A_cap)
    theory_results.append(p_theory)

    # Absolute error
    error = abs(p_mc - p_theory)
    abs_errors.append(error)


# =========================
# PLOT
# =========================

plt.figure(figsize=(7, 5))

plt.plot(
    mu_values,
    theory_results,
    linewidth=2,
    label="Theory",
    zorder=2
)

plt.scatter(
    mu_values,
    mc_results,
    color='red',
    s=20,
    label="Monte Carlo",
    zorder=3
)

plt.xlabel(r"$\mu$")
plt.ylabel(r"$P(\Phi(A)=0)$")
plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
plt.legend()
plt.tight_layout()

plt.savefig("Sim_of_no_sat_prob.pdf")
plt.show()

# =========================
# ERROR
# =========================

print("Maximum absolute error:", np.max(abs_errors))
