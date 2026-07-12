# Escape Velocity Library — 2026 Edition

Tools to infer **galaxy cluster masses** from spectroscopic radius–velocity phase-space using the **escape-velocity edge**. This repository is the 2026 update of the 2025 Escape Velocity Library and implements the updated escape-mass pipeline used in **Rodriguez et al. 2026a, in prep.**

The basic goal is to identify the boundary of the projected cluster phase-space, model that boundary as a down-sampled version of the three-dimensional escape-velocity profile, and infer the cluster mass by comparing the observed edge to a theoretical escape profile.

> **Please cite:** Rodriguez et al. 2026a, in prep.  
> For the original weak-lensing/escape-velocity concordance analysis, please also cite:  
> [Rodriguez & Miller (2025), *The Concordance of Weak Lensing and Escape-velocity Mass Estimates for Galaxy Clusters*, ApJ, 995, 213](https://iopscience.iop.org/article/10.3847/1538-4357/ae18ce)

---

## Repository layout

```text
Escape_Velocity_Library-2026-Edition/
├── AGAMA_Zv_calibration/
│   ├── Zv_fits_qH2_-0.01.pkl
│   ├── Zv_fits_qH2_-0.03.pkl
│   ├── Zv_fits_qH2_-0.05.pkl
│   └── ...
├── Example/
│   ├── run_escape_mass_example.ipynb
│   └── Rines_galaxy_data.txt
├── Function_Libraries/
│   ├── escape_analysis_functions.py
│   └── escape_theory_functions.py
└── README.md
```

The primary user-facing example is:

```text
Example/run_escape_mass_example.ipynb
```

This notebook demonstrates the full mass-estimation pipeline on the galaxy spectroscopic data for the cluster **A7**.

---

## What the code does

At a high level, the pipeline:

1. Loads a galaxy spectroscopic catalog with columns approximately of the form

   ```text
   RA_deg   DEC_deg   redshift
   ```

2. Builds the projected radius–velocity phase-space around a cluster center.

3. Computes line-of-sight velocities using

   ```math
   v_{\rm los} = c\,\frac{z_g-z_c}{1+z_c},
   ```

   where $z_g$ is the galaxy redshift and $z_c$ is the cluster redshift.

4. Iteratively recenters the phase-space and removes interlopers using a shifting-gapper-style procedure.

5. Measures the projected escape-velocity edge in radial bins.

6. Compares the measured edge to a Dehnen/NFW-based theoretical escape profile.

7. Corrects for the finite-sampling suppression of the observed phase-space edge using the calibrated $Z_v$ model.

8. Runs an MCMC inference for $M_{200}$, usually reported as $\log_{10}(M_{200}/M_\odot)$.

The main single-cluster entry point is:

```python
from escape_analysis_functions import MassEstimator_two_stage
```

The two-stage pipeline first performs a broad pilot run to identify the preferred aperture/mass scale, then performs a production run with the edge profile fixed using the pilot-stage aperture.

---

## Main physical model

The escape profile is modeled in an accelerating cosmological background as

```math
v_{\rm esc}^2(r)
=
-2\left[\Psi(r)-\Psi(r_{\rm eq})\right]
-
q(z)H^2(z)\left(r^2-r_{\rm eq}^2\right),
```

where:

- $\Psi(r)$ is the matter-only gravitational potential,
- $q(z)$ is the deceleration parameter,
- $H(z)$ is the Hubble parameter,
- $r_{\rm eq}$ is the equivalence radius where inward gravitational acceleration balances the outward cosmological acceleration term.

The observed phase-space edge is suppressed relative to the true 3D escape profile because the spectroscopic sampling is finite. In radial bin $j$, the model is approximately

```math
\widehat{v}_{{\rm esc},j}
\sim
\frac{v_{\rm esc,th}(R_j;M_{200},Q)}{Z_{v,j}},
```

where $R_j$ is the radius at which the edge is measured and $Z_{v,j}\geq 1$ is the finite-sampling suppression factor.

---

## The AGAMA $Z_v$ calibration files

The directory

```text
AGAMA_Zv_calibration/
```

contains precomputed calibration files for the suppression factor $Z_v$. You do **not** need to install or run AGAMA to use this library; the required calibration products are already stored as pickle files.

The 2026 pipeline differs from the 2025 version in an important way: the $Z_v$ calibration is no longer indexed primarily by redshift and halo mass. Instead, it is indexed by the cosmological acceleration combination

```math
Q \equiv \frac{q(z)H^2(z)}{H_0^2}.
```

The calibration files are named like

```text
Zv_fits_qH2_-0.01.pkl
Zv_fits_qH2_-0.03.pkl
Zv_fits_qH2_-0.05.pkl
...
```

where the number in the filename is the nearest available value of $Q=q(z)H^2(z)/H_0^2$. The calibration grid covers approximately

```math
-1 \lesssim Q \lesssim 0.5,
```

with the value snapped to the nearest available calibration file, typically within $\Delta Q \simeq 0.02$–$0.03$. A value of $Q=0$ corresponds to removing the cosmological acceleration contribution, i.e. the static-universe limit for this term.

For each value of $Q$, the calibration stores a model for

```math
p(Z_{v,j}\mid N_j,Q),
```

where:

- $j$ is the radial bin,
- $N_j$ is the number of accepted galaxies in that radial bin,
- $Q=q(z)H^2(z)/H_0^2$,
- $Z_{v,j}$ is the suppression factor in that bin.

In each pickle file, the $Z_v$ distribution is modeled with a skew-$t$ distribution,

```math
Z_{v,j}
\sim
{\rm Skew}\text{-}t
\left(
\xi_j,
\omega_j,
\alpha_j,
\nu_j
\right),
\qquad
Z_{v,j}\geq 1.
```

Here:

- $\xi_j$ is the location parameter,
- $\omega_j$ is the scale parameter,
- $\alpha_j$ is the skewness parameter,
- $\nu_j$ is the fat-tail/degrees-of-freedom parameter.

For the skewness, location, and scale parameters, the calibration stores best-fit slopes and intercepts as functions of $\log_{10}N_j$. Schematically,

```math
\log_{10}\alpha_j(N_j)
=
a_{\alpha,j}\log_{10}N_j+b_{\alpha,j},
```

```math
\log_{10}\xi_j(N_j)
=
a_{\xi,j}\log_{10}N_j+b_{\xi,j},
```

and

```math
\log_{10}\omega_j(N_j)
=
a_{\omega,j}\log_{10}N_j+b_{\omega,j}.
```

The fat-tail parameter $\nu_j$ is also stored in each pickle file, but it is modeled with a sigmoid-like function of $N_j$, allowing the distribution to have heavier tails at low sampling and to approach a more Gaussian form at high sampling.

The calibration was estimated from several hundred line-of-sight draws across a range of $N$ values chosen to cover the approximate sampling range encountered in the data. For a given cluster redshift, cosmology, radial bin, and phase-space sampling, the pipeline selects the appropriate $Q$-indexed calibration file and evaluates the corresponding $Z_v$ distribution.

---

## Quick start: A7 example

The main example notebook is:

```text
Example/run_escape_mass_example.ipynb
```

It shows how to run the pipeline on a sample galaxy spectroscopic catalog for the cluster **A7**.


The output dictionary contains the posterior mass summary, typically including the median and 68% credible interval for

```math
\log_{10}\left(M_{200}/M_\odot\right).
```

The notebook also produces diagnostic plots showing the phase-space, measured edge, posterior distribution, and final model comparison.

---

## Important hyperparameters

Some systems require modest tuning of the phase-space or edge-definition hyperparameters. The most common ones are:

### `vesc_error_floor`

```python
vesc_error_floor = 30
```

This is the velocity uncertainty floor in km/s used in the edge likelihood. For the HeCS/HeCS-SZ-style spectroscopic data used in Rodriguez & Miller (2025), $30\,{\rm km\,s^{-1}}$ corresponds roughly to the spectroscopic redshift uncertainty propagated into the edge-profile uncertainty.

If your spectroscopic data have larger velocity uncertainties, this value should be increased accordingly.

### `coremin_cut`

```python
coremin_cut = 0.44
```

This sets the inner radius, in units of $r_{200}$, used by the interloper-rejection logic. In practice, this controls how aggressively the shifting-gapper procedure is allowed to reject galaxies in the cluster core.

Changing this value can be useful if the central phase-space is visibly over-cleaned or under-cleaned.

### `cut`

```python
cut = 4500
```

This is the maximum allowed absolute line-of-sight peculiar velocity, in km/s, used when constructing the phase-space:

```math
|v_{\rm los}| < {\tt cut}.
```

The default value is $4500\,{\rm km\,s^{-1}}$. In some systems, especially if a high-velocity interloper is affecting the edge, a smaller value such as

```python
cut = 3000
```

may be more appropriate.

### `NON_INC`

```python
NON_INC = True
```

This boolean controls whether the measured edge profile is encouraged to be non-increasing with radius. This is useful because escape-velocity profiles should generally decline with radius, and poorly sampled outer bins can otherwise produce unphysical rises.

For some disturbed, merging, or sparsely sampled systems, however, this constraint can make the edge identification worse. In those cases, try

```python
NON_INC = False
```

which uses the maximum velocities on the edges rather than enforcing the non-increasing behavior.

---

## Recommended diagnostic checks

The two-stage pipeline shows both the pilot-stage and production-stage results. It is important to inspect:

1. The pilot posterior.
2. The production posterior.
3. The phase-space edge profile.
4. Whether the measured edge visually follows the boundary of the galaxy phase-space.
5. Whether individual bins are dominated by obvious interlopers or very low sampling.

If the pilot posterior is highly non-Gaussian, strongly multi-modal, or clearly inconsistent with the visual phase-space boundary, consider changing:

```python
NON_INC = False
```

or reducing the velocity cut, for example:

```python
cut = 3000
```

instead of

```python
cut = 4500
```

Roughly 5–10% of systems may require some fine-tuning of `coremin_cut`, `cut`, or `NON_INC` to ensure that the measured edge profile traces the visual phase-space boundary.

---

## Installation

This is a pure-Python research code. A typical environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate

pip install numpy scipy pandas astropy emcee matplotlib jupyter
```

Depending on which notebooks and plotting options are used, you may also want:

```bash
pip install corner
```

---

## Required input format

For the standard sky-coordinate mode, the galaxy catalog should be an array with shape

```text
(N_galaxies, 3)
```

with columns:

```text
RA_deg   DEC_deg   redshift
```

where RA and Dec are decimal degrees and redshift is dimensionless.

The cluster position should be passed as:

```python
cluster_positional_data = (cl_ra_deg, cl_dec_deg, cl_z)
```

where `cl_z` is the cluster redshift.

The initial mass estimate should be passed in log space:

```python
M200_estimate = log10(M200 / Msun)
```

---

## Main outputs

The mass-estimation routine returns a dictionary. On a successful run, the key quantities include:

```python
results["median"]
results["one_sig_down"]
results["one_sig_up"]
results["samples"]
results["acceptance"]
```

where the mass values are in

```math
\log_{10}\left(M_{200}/M_\odot\right).
```

The plotting routines also show or save diagnostic phase-space and posterior plots, depending on the settings passed to `MassEstimator_two_stage`.

---

## Notes on the 2026 update

Relative to the 2025 version, the main changes are:

1. The $Z_v$ model is now a skew-$t$ distribution rather than a skew-normal distribution.

2. The calibration is now bin-wise:

   ```math
   \mathbf{N} = (N_1,\ldots,N_{\rm bins}),
   ```

   rather than being controlled by a single total sampling value.

3. The theoretical edge is evaluated at the actual projected radius of the galaxy defining the edge in each bin, rather than at fixed bin centers.

4. The calibration is indexed by

   ```math
   Q = q(z)H^2(z)/H_0^2,
   ```

   rather than redshift and mass separately.

5. $Z_v$ is marginalized using deterministic quadrature nodes, rather than by drawing a single stochastic realization in each likelihood evaluation.

6. The default user-facing mass estimator uses a two-stage procedure to reduce sensitivity to the initial mass guess and to converge to an aperture determined by the escape data.

---

## Data acknowledgement

The example data are based on galaxy spectroscopy from HeCS / HeCS-SZ-style cluster observations. Users should cite the appropriate data source if using these catalogs directly.

---

## Citation

If you use this library, please cite:

```text
Rodriguez et al. 2026a, in prep.
```

Please also cite the previous escape-velocity / weak-lensing concordance analysis:

```text
Rodriguez, A. & Miller, C. J. 2025,
The Concordance of Weak Lensing and Escape-velocity Mass Estimates for Galaxy Clusters,
ApJ, 995, 213.
https://iopscience.iop.org/article/10.3847/1538-4357/ae18ce
```
