This repository contains the Python scripts used to generate the simulations, analyses, and figures presented for the Bachelor Project "Coverage Analysis of
LEO Satellite Constellations".

# Repository Structure
The repository is divided into three main sections:
- `Simulation Setup and Verification`
- `System Parameter Analysis`
- `Empirical Comparison`


# Requirements
Install the required packages before running the scripts:

```bash
pip install numpy scipy matplotlib skyfield tqdm requests
```

The following Python modules are used throughout the project:

- `numpy - Version 2.1.3`
- `scipy - Version 1.14.1`
- `matplotlib - Version 3.9.2`
- `skyfield - Version 1.54`
- `tqdm - Version 4.67.3`
- `requests - Version 2.32.5`
- `datetime - Standard library`
- `collections - Standard library`
- `statistics - Standard library`
- `random - Standard library`
- `os - Standard library`

Tested with Python Version 3.12.10

# Reproducing Figures
Random seeds are specified within the scripts to ensure reproducibility of simulation results.
## Chapter 5 – Simulation Setup and Verification

- **Fig. 5.1**  
  `Satellite_Constellation_Figure.py`

- **Fig. 5.2**  
  `No_Sat_Simulation_vs_Theoretical.py`

- **Fig. 5.3(a)**  
  `SNR_Simulation_vs_Theoretical_Coverage_Probability.py`

- **Fig. 5.3(b)**  
  `SIR_Simulation_vs_Theoretical_Coverage_Probability.py`


## Chapter 6 – System Parameter Analysis
- **Fig. 6.1**  
  `No_Sat_Simulation_vs_Theoretical_Altitude.py`

- **Fig. 6.2(a)**  
  `SNR_Simulation_vs_Theoretical_Coverage_Probability_Altitude.py`

- **Fig. 6.2(b)**  
  `SIR_Simulation_vs_Theoretical_Coverage_Probability_Altitude.py`

- **Fig. 6.3(a)**  
  `SNR_Simulation_vs_Theoretical_Coverage_Probability_Density.py`

- **Fig. 6.3(b)**  
  `SIR_Simulation_vs_Theoretical_Coverage_Probability_Density.py`

- **Fig. 6.4**  
  `SNR_Simulation_vs_Theoretical_Coverage_Probability_g.py`

- **Fig. 6.5**  
  `SIR_Simulation_vs_Theoretical_Coverage_Probability_G_bar.py`


## Chapter 7 – Real World Comparison
- **Fig. 7.1(a)**  
  `Satellite_Visibility_Comparison.py`  
  Parameter: `elevation = 0`

- **Fig. 7.1(b)**  
  `Satellite_Visibility_Comparison.py`  
  Parameter: `elevation = 25`

- **Fig. 7.2**  
  `Starlink_PPP_Validation.py`  
  Parameter: `elevation = 0`

- **Fig. 7.3(a)**  
  `Starlink_PPP_Validation.py`  
  Parameters:
  - `elevation = 25`
  - `tuning_factor = 0.95`

- **Fig. 7.3(b)**  
  `Starlink_PPP_Validation.py`  
  Parameters:
  - `elevation = 25`
  - `tuning_factor = 0.95`

- **Fig. 7.4(a)**  
  `Starlink_PPP_Validation.py`  
  Parameters:
  - `elevation = 40`
  - `tuning_factor = 0.6`

- **Fig. 7.4(b)**  
  `Starlink_PPP_Validation.py`  
  Parameters:
  - `elevation = 40`
  - `tuning_factor = 0.6`

# Running the Scripts

All scripts can be executed directly from the command line:

```bash
python filename.py
```

# Data Sources

Some scripts retrieve TLE data using online sources through the `skyfield` package. An internet connection may therefore be required when running these scripts for the first time.
