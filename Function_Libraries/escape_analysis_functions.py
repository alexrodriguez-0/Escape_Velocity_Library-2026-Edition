"""Escape-velocity mass and cosmology inference pipeline.

This module implements the end-to-end workflow used to infer galaxy-cluster masses
from projected phase-space escape edges, and to propagate those constraints into
cosmological parameter inference.

Key updates implemented in this version of the pipeline
------------------------------------------------------
* **Z_v (phase-space suppression) model:** skew-normal → **skew‑t** per radial bin.
* **No explicit mass dependence in Z_v:** suppression is calibrated primarily as a
  function of sampling, using **N per radial bin** (rather than total N), and the
  edge is evaluated at the **actual radius of the selected edge galaxy** in each bin
  (rather than fixed bin centers).
* **Likelihood treatment:** instead of drawing a single Z_v realization, the
  likelihood can **marginalize over Z_v** using deterministic quadrature nodes.
* **Cosmology dependence of Z_v:** the calibration is indexed by **qH²** (a proxy
  for cosmological acceleration effects) rather than redshift alone.

The main user-facing entry points are:
  - :class:`EscapeVelocityModeling` (theory + Z_v calibration),
  - :class:`MCMCMassEstimator` (single-cluster mass inference),
  - :class:`CosmologyEstimator` and :class:`CosmologyEstimator_qH2` (cosmology fits),
  - :func:`MassEstimator_main` (batch driver with post-processing).

Notes
-----
This file depends on ``escape_theory_functions`` for cosmology and escape-profile
theory utilities (e.g., ``D_A``, ``rho_crit_z``, ``v_esc_dehnen``).
"""

import os
import pickle
from multiprocessing import Pool
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from astropy.stats import biweight_location
from emcee.autocorr import AutocorrError
from scipy.stats import skewnorm
from pathlib import Path
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.signal import savgol_filter
from scipy.special import gammaln, log1p, stdtr, logsumexp
from scipy import interpolate, stats
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass
from scipy.special import logsumexp
from scipy.stats import norm
import glob
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermegauss

from escape_theory_functions import (
    D_A,
    dehnen_nfwM200_errors,
    rho_crit_z,
    v_esc_dehnen,
    v_esc_dehnen_qH2,
    q_z_function, H_z_function,concentration_meta
)


# -----------------------------------------------------------------------------
# Escape-velocity theory & Zv calibration utilities
# -----------------------------------------------------------------------------


class EscapeVelocityModeling:
    """Theory and Z_v calibration interface.

    This class wraps two pieces of the analysis:

    1) **Theoretical escape curve** evaluation for a Dehnen-profile potential matched
       to an NFW halo (via a mass–concentration relation), returning the predicted
       3D escape-velocity profile projected onto the observational angular grid.

    2) **Phase-space suppression (Z_v) calibration** used to map the theoretical
       escape curve to the observed edge. In this pipeline, Z_v is modeled *per radial
       bin* as a calibrated **skew‑t** distribution whose parameters are functions of
       sampling in that bin (``N_hat``) and a cosmology proxy (``qH2``).

    The primary API is:
      - :meth:`v_esc_den_M200` for the theoretical edge,
      - :meth:`Zv_params` / :meth:`sample_Zv` for the Z_v distribution,
      - :meth:`Zv_quantile_nodes` for deterministic Z_v quadrature nodes and weights
        used to marginalize Z_v in a likelihood.

    Parameters
    ----------
    path_to_calibration : str or pathlib.Path, optional
        Directory containing Z_v calibration pickles. The calibration is assumed to be
        indexed by ``qH2`` (and typically by radial-bin count). Files are discovered
        lazily and cached.

    Notes
    -----
    * Z_v is enforced to satisfy Z_v >= 1 (suppression cannot increase the edge).
    * The skew‑t CDF/PPF utilities are built numerically on a grid and cached.
    """

    def __init__(self, path_to_calibration=None):
        """Initialize the modeler.

        Parameters
        ----------
        path_to_calibration : str or pathlib.Path, optional
            Path to the directory containing the Z_v calibration files. If provided,
            the directory is scanned on first use to build an internal mapping from
            ``qH2`` grid values to calibration pickle paths.
        """
        self.path_to_calibration = path_to_calibration

    def v_esc_den_M200(self, theta, z, M200, cosmo_params, case, qH2=None, conc_override=None):
        """Evaluate the theoretical escape-velocity curve for a given mass.

        Parameters
        ----------
        theta : array-like or astropy.units.Quantity
            Angular positions where the theory curve is evaluated.
        z : array-like
            Cluster redshift(s).
        M200 : array-like
            Halo mass(es) M200 in solar masses.
        cosmo_params : sequence
            Cosmological parameters.
        case : str
            Cosmology / distance convention identifier.
        qH2 : float, optional
            If provided, use the qH2-aware evaluator.
        conc_override : float or array-like, optional
            If provided, override the concentration used in dehnen_nfwM200_errors.
            May be a scalar (same concentration for all halos) or one value per halo.

        Returns
        -------
        tuple
            Return from v_esc_dehnen / v_esc_dehnen_qH2.
        """
        all_mass_0 = []
        all_r_s = []
        all_gamma = []

        M200 = np.atleast_1d(M200)
        z = np.atleast_1d(z)

        if conc_override is None:
            conc_arr = [None] * len(M200)
        else:
            conc_arr = np.atleast_1d(conc_override)
            if len(conc_arr) == 1 and len(M200) > 1:
                conc_arr = np.repeat(conc_arr, len(M200))
            if len(conc_arr) != len(M200):
                raise ValueError("conc_override must be scalar or have same length as M200")

        for i in range(len(M200)):
            (
                _M200_0,
                _R200,
                _conc,
                mass_0,
                r_s,
                gamma,
                _sigma_mass_0,
                _sigma_r_s,
                _sigma_gamma,
            ) = dehnen_nfwM200_errors(
                M200[i],
                z[i],
                cosmo_params,
                case,
                conc_override=conc_arr[i]
            )
            all_mass_0.append(mass_0)
            all_r_s.append(r_s)
            all_gamma.append(gamma)

        all_mass_0 = np.array(all_mass_0)
        all_r_s = np.array(all_r_s)
        all_gamma = np.array(all_gamma)

        if qH2 is None:
            return v_esc_dehnen(theta, z, all_mass_0, all_r_s, all_gamma, cosmo_params, case)
        else:
            return v_esc_dehnen_qH2(theta, z, all_mass_0, all_r_s, all_gamma, cosmo_params, case, qH2=qH2)

    @staticmethod
    def z_round(x, base=5):
        """
        Round redshift to nearest 0.05 for Zv calibration file lookup.

        Parameters
        ----------
        x : float
            Redshift value.
        base : int, default=5
            Rounding base (0.05 corresponds to 5 in 1e-2 units).

        Returns
        -------
        float
            Rounded redshift, with a floor of 0.01 to avoid a zero label.
        """
        x = x * 100.0
        z_rounded = (base * round(x / base)) / 100.0
        if z_rounded == 0.0:
            z_rounded = 0.01
        return z_rounded

    @staticmethod
    def random_draw(x, f_x, N):
        """Draw random samples from a discrete PDF via inverse-transform sampling.

        Parameters
        ----------
        x : array-like
            Grid points spanning the domain.
        f_x : array-like
            Non-negative PDF values evaluated at ``x`` (need not be normalized).
        N : int
            Number of samples to draw.

        Returns
        -------
        ndarray
            Samples drawn from the distribution defined by (x, f_x).

        Notes
        -----
        This helper is used for sampling Z_v from a numerically defined PDF.
        """
        # Get cdf normalized to maximum
        cdf = np.cumsum(f_x) / np.cumsum(f_x).max()
        # Inverse transform using interpolation
        inverse_cdf = interpolate.interp1d(cdf, x)
        samples = inverse_cdf(np.random.uniform(np.min(cdf), np.max(cdf), int(N)))
        return samples

    @staticmethod
    def predict_from_fitparameters(N, fit_parameters, nu_min=2.0, loc_sign=+1.0):
        """Convert calibration fit coefficients into skew‑t parameters.

        The calibration stores simple parametric relations for the skew‑t parameters as
        functions of sampling ``N``. This helper evaluates those relations to return the
        parameters used by the skew‑t distribution in each radial bin.

        Parameters
        ----------
        N : float or array-like
            Effective number of tracers in the bin(s).
        fit_parameters : sequence
            A container encoding the fitted relations, typically
            ``[(a_m, a_b), (loc_m, loc_b), (sc_m, sc_b), nu_params]`` where
            ``log10(param) = m*log10(N) + b`` for (alpha, location, scale), and ``nu_params``
            defines a saturating curve for the degrees of freedom (Hill / Richards form).
        nu_min : float, default=2.0
            Lower asymptote for the degrees of freedom. Values are clipped to be > 2
            (required for finite variance).
        loc_sign : {+1, -1}, default=+1
            Sign convention applied to the location parameter.

        Returns
        -------
        alpha : ndarray
            Skewness parameter(s).
        xi : ndarray
            Location parameter(s).
        omega : ndarray
            Scale parameter(s).
        nu : ndarray
            Degrees of freedom parameter(s).
        """
        N = np.asarray(N, float)

        # unpack
        (a_m, a_b), (loc_m, loc_b), (sc_m, sc_b), nu_params = fit_parameters

        # power laws
        logN = np.log10(N)
        skew  = 10.0**(a_m   * logN + a_b)
        loc   = loc_sign * 10.0**(loc_m * logN + loc_b)
        scale = 10.0**(sc_m * logN + sc_b)

        # ν: Hill (3 params) or Richards (4 params)
        if len(nu_params) == 3:
            nu_max, N0, p = nu_params
            nu = nu_min + (nu_max - nu_min) / (1.0 + (N0 / N)**p)
        elif len(nu_params) == 4:
            nu_max, N0, p, m = nu_params
            nu = nu_min + (nu_max - nu_min) / (1.0 + (N0 / N)**p)**m
        else:
            raise ValueError("nu_params must have 3 (Hill) or 4 (Richards) values.")

        return skew, loc, scale, nu


    # inside EscapeVelocityModeling
    def _init_qH2_grid(self):
        """Discover available qH2 calibration files and build a lookup table.

        This method scans ``path_to_calibration`` for calibration pickles and builds:
          - ``self._qH2_grid``: sorted unique qH2 grid values,
          - ``self._qH2_files``: mapping ``qH2 -> filepath``.

        Notes
        -----
        The exact filename convention is defined by the calibration-writing code.
        """
        pattern = os.path.join(self.path_to_calibration, "Zv_fits_qH2_*.pkl")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No Zv_fits_qH2_*.pkl files found in {self.path_to_calibration}")

        qH2_vals = []
        file_map = {}
        for fname in files:
            base = os.path.basename(fname)
            num_str = base.replace("Zv_fits_qH2_", "").replace(".pkl", "")
            q_val = float(num_str)
            qH2_vals.append(q_val)
            file_map[q_val] = fname

        self._qH2_grid  = np.array(sorted(qH2_vals))
        self._qH2_files = file_map

    def _nearest_qH2(self, qH2_use, tol=0.02):
        """Snap a requested qH2 value to the nearest available calibration grid point.

        Parameters
        ----------
        qH2_use : float
            Requested qH² value.
        tol : float, default=500.0
            Maximum absolute difference allowed between the request and the nearest grid
            point.

        Returns
        -------
        float
            The nearest qH² grid value.

        Raises
        ------
        ValueError
            If no grid point is within ``tol``.
        """
        if not hasattr(self, "_qH2_grid"):
            self._init_qH2_grid()

        qH2_use = float(qH2_use)
        diffs = np.abs(self._qH2_grid - qH2_use)
        idx   = int(np.argmin(diffs))
        dmin  = float(diffs[idx])
        qH2_near = float(self._qH2_grid[idx])

        # Enforce tolerance ONLY in the constraining regime (qH2 <= 0)
        if (qH2_use <= 0.0) and (dmin > tol):
            raise ValueError(
                f"No qH² calibration within {tol} of qH²={qH2_use:.3f}. "
                f"Closest grid value is {qH2_near:.3f} (Δ={dmin:.3f})."
            )

        return qH2_near


    def Zv_params(self, N, bins, qH2_use):
        """Load the Z_v calibration and return per-bin skew‑t parameters.

        Parameters
        ----------
        N : array-like, shape (nbins,)
            Effective number of tracers *per radial bin*.
        bins : int
            Number of radial bins (used to index the calibration content).
        qH2_use : float
            qH² value selecting the appropriate calibration file.

        Returns
        -------
        params : ndarray, shape (4, nbins)
            Array containing ``xi, omega, alpha, nu`` in that order.

        Notes
        -----
        Calibration files are cached after first load to avoid repeated disk I/O.
        """
        if self.path_to_calibration is None:
            raise ValueError("Path to calibration data must be set")

        # This will only raise if qH2_use <= 0 and no grid point within tol
        qH2q = self._nearest_qH2(qH2_use, tol=0.02)

        if not hasattr(self, "_calib_cache"):
            self._calib_cache = {}

        key = ("qH2", qH2q)
        if key not in self._calib_cache:
            calib_file = self._qH2_files[qH2q]
            with open(calib_file, "rb") as f:
                self._calib_cache[key] = pickle.load(f)

        all_fit_parameters = self._calib_cache[key]

        N = np.asarray(N, float)
        if N.shape[0] != bins:
            raise ValueError(f"N should have length {bins}, got {N.shape[0]}")

        a_list, loc_list, scale_list, nu_list = [], [], [], []
        for i in range(bins):
            skew_pred, loc_pred, scale_pred, nu_pred = self.predict_from_fitparameters(
                N[i], all_fit_parameters[i], nu_min=2.0, loc_sign=+1.0
            )
            a_list.append(skew_pred)
            loc_list.append(loc_pred)
            scale_list.append(scale_pred)
            nu_list.append(nu_pred)

        return np.array([
            np.array(loc_list),
            np.array(scale_list),
            np.array(a_list),
            np.array(nu_list),
        ])


    def sample_Zv(self, N_hat, nbins, qH2, size=1, rand=None):
        """Sample from the calibrated skew‑t Z_v distribution truncated at Z_v >= 1.

        For each bin j, the standardized skew‑t CDF is used to compute the probability mass
        above the truncation point Z=1, then uniform variates are mapped through the PPF to
        draw from the truncated distribution.

        Parameters
        ----------
        N_hat : array-like, shape (nbins,)
            Effective number of tracers per bin.
        nbins : int
            Number of bins.
        qH2 : float
            qH² value selecting the calibration.
        size : int, default=1
            Number of draws.
        rand : int, optional
            Seed for NumPy RNG.

        Returns
        -------
        ndarray
            Z_v samples. If ``size == 1`` returns ``(nbins,)``; otherwise ``(size, nbins)``.

        Notes
        -----
        This is the sampling routine that matches the Z_v marginalization logic used in the
        deterministic node construction.
        """
        xi, om, alpha, nu = self.Zv_params(N_hat, nbins, qH2)
        out = np.empty((size, nbins), float)
        if rand is not None:
            state = np.random.get_state()
            np.random.seed(rand)


        for j in range(nbins):
            x0  = float(xi[j]);  s = float(max(om[j], np.finfo(float).tiny))
            a   = float(alpha[j]); df = float(max(nu[j], 2.0001))
            cdf_std = self._build_skewt_cdf(a, df)
            ppf_std = self._build_skewt_ppf(a, df)

            p1 = float(cdf_std((1.0 - x0)/s))        # CDF at Z=1
            u  = p1 + (1.0 - p1)*np.random.random(size)   # uniform on [p1,1)
            X  = ppf_std(u)                           # standardized
            out[:, j] = np.maximum(x0 + s*X, 1.0)

        if rand is not None:
            np.random.set_state(state)

        return out.squeeze()   # (nbins,) if size==1, else (size, nbins)


    def _skewt_pdf_std(self, x, alpha, nu):
        """Standardized skew‑t PDF.

        Parameters
        ----------
        x : array-like
            Evaluation points in standardized space.
        alpha : float
            Skewness parameter.
        nu : float
            Degrees of freedom (> 2).

        Returns
        -------
        ndarray
            PDF values in standardized space.
        """
        x = np.asarray(x, dtype=float)
        t_pdf = stats.t.pdf(x, df=nu)
        arg   = alpha * x * np.sqrt((nu + 1.0) / (nu + x*x))
        t_cdf = stats.t.cdf(arg, df=nu + 1.0)
        return 2.0 * t_pdf * t_cdf


    def _skewt_support(self, alpha, nu, p_lo=1e-10, p_hi=1.0 - 1e-10, pad_frac=0.15):
        """Choose a finite x-range that captures essentially all probability mass.

        Parameters
        ----------
        alpha : float
            Skewness parameter.
        nu : float
            Degrees of freedom.
        p_lo, p_hi : float, default=(1e-10, 1-1e-10)
            Target lower/upper tail probabilities used to set the support.
        pad_frac : float, default=0.15
            Fractional padding added to the computed range for numerical safety.

        Returns
        -------
        xmin, xmax : float
            Support bounds in standardized space.
        """
        qlo = stats.t.ppf(p_lo, nu)
        qhi = stats.t.ppf(p_hi, nu)
        span = qhi - qlo
        pad  = pad_frac * span
        x_min = qlo - pad
        x_max = qhi + pad
        if abs(alpha) > 10:
            extra = 0.1 * span * (abs(alpha) / 10.0)
            x_min -= extra
            x_max += extra
        return float(x_min), float(x_max)


    def _build_skewt_cdf(self, alpha, nu, xmin=None, xmax=None, nx=4096):
        """Build and return a callable CDF for the standardized skew‑t.

        The CDF is constructed numerically on a grid and then returned as a monotone
        interpolation function.

        Parameters
        ----------
        alpha : float
            Skewness parameter.
        nu : float
            Degrees of freedom (> 2).
        xmin, xmax : float, optional
            Support bounds. If not provided, they are determined automatically.
        nx : int, default=4096
            Number of grid points.

        Returns
        -------
        cdf : callable
            Function mapping x -> F(x), vectorized over NumPy arrays.
        """
        if xmin is None or xmax is None:
            xmin, xmax = self._skewt_support(alpha, nu)

        # Guard: nx must be >= 256 to keep the grid smooth
        nx = int(max(256, nx))
        xg = np.linspace(xmin, xmax, nx)
        fg = self._skewt_pdf_std(xg, alpha, nu)

        # Make sure pdf is nonnegative & finite
        fg = np.where(np.isfinite(fg) & (fg >= 0.0), fg, 0.0)

        # Cumulative integral to get unnormalized CDF
        F = cumulative_trapezoid(fg, xg, initial=0.0)
        total = F[-1]
        if not np.isfinite(total) or total <= 0.0:
            # fallback: pretend it's ~t; avoid crashing
            total = np.trapz(fg, xg)
            if not np.isfinite(total) or total <= 0.0:
                total = 1.0
        F = F / total

        # Enforce strict monotonicity and [0,1] bounds
        # (tiny slope added to avoid flat plateaus that break inversion)
        eps_slope = 1e-12
        F = np.clip(F, 0.0, 1.0)
        F = np.maximum.accumulate(F)
        F = np.minimum(F, 1.0)
        F = F + np.linspace(0.0, eps_slope, F.size)

        def cdf(x):
            xx = np.asarray(x, dtype=float)
            return np.interp(xx, xg, F, left=0.0, right=1.0)

        return cdf


    def _build_skewt_ppf(self, alpha, nu, xmin=None, xmax=None, nx=4096):
        """Build and return a callable PPF (inverse CDF) for the standardized skew‑t.

        The inverse CDF is obtained by (i) building a numerically integrated CDF on a grid,
        (ii) enforcing monotonicity/stability, and (iii) interpolating x(F).

        Parameters
        ----------
        alpha : float
            Skewness parameter.
        nu : float
            Degrees of freedom (> 2).
        xmin, xmax : float, optional
            Support bounds in standardized space.
        nx : int, default=4096
            Number of grid points used to tabulate the CDF.

        Returns
        -------
        ppf : callable
            Function mapping u in (0,1) to x, vectorized over NumPy arrays.
        """
        if xmin is None or xmax is None:
            xmin, xmax = self._skewt_support(alpha, nu)

        nx = int(max(256, nx))
        xg = np.linspace(xmin, xmax, nx)
        fg = self._skewt_pdf_std(xg, alpha, nu)
        fg = np.where(np.isfinite(fg) & (fg >= 0.0), fg, 0.0)

        F = cumulative_trapezoid(fg, xg, initial=0.0)
        total = F[-1]
        if not np.isfinite(total) or total <= 0.0:
            total = np.trapz(fg, xg)
            if not np.isfinite(total) or total <= 0.0:
                total = 1.0
        F = F / total

        # Same monotonic/bounds enforcement as in CDF builder
        eps_slope = 1e-12
        F = np.clip(F, 0.0, 1.0)
        F = np.maximum.accumulate(F)
        F = np.minimum(F, 1.0)
        F = F + np.linspace(0.0, eps_slope, F.size)

        # Ensure F[0] < F[-1] to avoid division by zero in interpolation
        F0 = max(F[0], 0.0)
        F1 = min(F[-1], 1.0)
        if F1 - F0 < 1e-10:
            x_mid = 0.5 * (xmin + xmax)
            def ppf(u):
                u = np.asarray(u, dtype=float)
                return np.full_like(u, x_mid, dtype=float)
            return ppf

        def ppf(u):
            uu = np.asarray(u, dtype=float)
            uu = np.clip(uu, np.nextafter(0.0, 1.0), 1.0 - 1e-12)
            return np.interp(uu, F, xg)

        return ppf



    def Zv_quantile_nodes(self, N_hat, nbins, qH2, K=64, return_logW=True):
        """Deterministic Gauss–Legendre quadrature nodes for the truncated Z_v distribution.

        Constructs ``K`` fixed quadrature nodes and associated log-weights for
        numerically marginalizing Z_v | (Z_v >= 1) in the likelihood. The nodes
        are derived from Gauss–Legendre quadrature on the truncated skew‑t CDF
        per radial bin.

        Parameters
        ----------
        N_hat : array-like, shape (nbins,)
            Effective tracer counts per bin (drives Z_v calibration via ``Zv_params``).
        nbins : int
            Number of radial bins.
        qH2 : float
            qH² calibration coordinate selecting the appropriate calibration file.
        K : int, default=64
            Number of quadrature nodes per bin.
        return_logW : bool, default=True
            If True, return ``(Z, logW)``; otherwise return only ``Z``.

        Returns
        -------
        Z : ndarray, shape (K, nbins)
            Quadrature node values of Z_v (all >= 1).
        logW : ndarray, shape (K, nbins)
            Log-weights corresponding to the Gauss–Legendre quadrature rule.
            Returned only if ``return_logW=True``.

        Notes
        -----
        The nodes and weights define the approximation:

            E[f(Z_v)] ≈ sum_k exp(logW[k, j]) * f(Z[k, j])

        for each bin j. Weights are the same across bins (from GL rule); the
        node values differ because each bin has different skew‑t parameters.
        """
        # Gauss–Legendre nodes/weights on [-1, 1], mapped to (0, 1)
        gx, gw = leggauss(int(K))
        u = 0.5 * (gx + 1.0)   # nodes on (0, 1)
        w = gw / 2.0            # weights on (0, 1); sum(w) = 1

        xi, om, alpha, nu = self.Zv_params(N_hat, nbins, qH2)
        Z = np.empty((K, nbins), float)

        for j in range(nbins):
            x0 = float(xi[j])
            s  = float(max(om[j], np.finfo(float).tiny))
            a  = float(alpha[j])
            df = float(max(nu[j], 2.0001))

            # Build numerically integrated CDF and PPF for the standardized skew-t
            cdf_std = self._build_skewt_cdf(a, df)
            ppf_std = self._build_skewt_ppf(a, df)

            # Lower-truncation at Z >= 1: map uniform (0, 1) -> (p1, 1)
            p1  = float(cdf_std((1.0 - x0) / s))
            upr = p1 + (1.0 - p1) * u            # remapped nodes on (p1, 1)
            X   = np.asarray(ppf_std(upr), float) # standardized quantile values
            Z[:, j] = np.maximum(x0 + s * X, 1.0)

        if return_logW:
            # Broadcast the same GL weights to all bins: shape (K, nbins)
            logW = np.log(w)[:, None] + np.zeros((K, nbins))
            return Z, logW
        else:
            return Z

# -----------------------------------------------------------------------------
# Data preparation: centering, interloper removal, edge finding
# -----------------------------------------------------------------------------


class ClusterDataHandler:
    """Utilities for preparing galaxy phase-space data and measuring the escape edge.

    The analysis repeatedly needs to:
      - project galaxy positions relative to a cluster center,
      - compute line-of-sight velocities,
      - reject interlopers (shifting-gapper),
      - estimate the high-|v| edge in radial bins,
      - optionally smooth and enforce non-increasing edge constraint.

    This class is a thin namespace for those operations; it carries no persistent state.
    """

    def __init__(self):
        """Construct a data handler.

        Notes
        -----
        The handler is stateless; this initializer exists mainly for symmetry and future
        extension.
        """
        pass

    @staticmethod
    def shiftgapper(data, nbin_val, gap_val, coremin):
        """Shifting-gapper interloper rejection.

        Galaxies are sorted by projected radius, split into radial bins, and iteratively
        clipped in velocity space using a gap statistic until convergence.

        Parameters
        ----------
        data : ndarray, shape (N, 2)
            Columns are (r_proj, v_los) where r_proj is projected radius (same units as
            R200) and v_los is line-of-sight velocity in km/s.
        nbin_val : int
            Approximate number of galaxies per radial bin for the shifting-gapper.
        gap_val : float
            Velocity gap threshold (km/s).
        coremin : float
            Core radius inside which the shifting-gapper is not applied (same units as
            r_proj).

        Returns
        -------
        ndarray, shape (N_keep, 2)
            Filtered (r_proj, v_los) array after interloper rejection.
        """
        gap_prev = 1000
        npbin    = nbin_val
        nbins    = np.int32(np.ceil(data[:, 0].size / (npbin * 1.0)))
        data     = data[np.argsort(data[:, 0])]  # sort by r_proj before binning

        for i in range(nbins):
            databin = data[npbin * i:npbin * (i + 1)]
            datanew = None
            nsize = databin[:, 0].size
            datasize = nsize - 1

            if nsize > 5:
                while nsize - datasize > 0 and datasize >= 5:
                    nsize = databin[:, 0].size
                    databinsort = databin[np.argsort(databin[:, 1])]  # sort by v

                    q_hi = int(np.ceil(databinsort[:, 1].size / 4.0))
                    q_lo = q_hi
                    f = databinsort[:, 1][-q_hi] - databinsort[:, 1][q_lo]
                    gap = f / 1.349

                    if gap < gap_val:
                        break
                    if gap >= 2.0 * gap_prev:
                        gap = gap_prev

                    databelow = databinsort[databinsort[:, 1] <= 0]
                    gapbelow = databelow[:, 1][1:] - databelow[:, 1][:-1]
                    dataabove = databinsort[databinsort[:, 1] > 0]
                    gapabove = dataabove[:, 1][1:] - dataabove[:, 1][:-1]

                    try:
                        if np.max(gapbelow) >= gap:
                            vgapbelow = np.where(gapbelow >= gap)[0][-1]
                        else:
                            vgapbelow = -1
                        try:
                            datanew = np.append(datanew, databelow[vgapbelow + 1:], axis=0)
                        except Exception:
                            datanew = databelow[vgapbelow + 1:]
                    except ValueError:
                        pass

                    try:
                        if np.max(gapabove) >= gap:
                            vgapabove = np.where(gapabove >= gap)[0][0]
                        else:
                            vgapabove = 99999999
                        try:
                            datanew = np.append(datanew, dataabove[:vgapabove + 1], axis=0)
                        except Exception:
                            datanew = dataabove[:vgapabove + 1]
                    except ValueError:
                        pass

                    databin = datanew
                    datasize = datanew[:, 0].size
                    datanew = None

                if gap >= 2000.0:
                    gap_prev = gap
                else:
                    gap_prev = 2000.0

            try:
                datafinal = np.append(datafinal, databin, axis=0)
            except Exception:
                datafinal = databin

        w1 = np.where(data[:, 0] < coremin)[0]
        w2 = np.where(datafinal[:, 0] > coremin)[0]
        datafinal = np.array(data[w1].tolist() + datafinal[w2].tolist())

        return datafinal
    

    
    @staticmethod
    def shiftgapper_adaptive(data,npbin,gap_use,coremin):
        """
        Shifting-gapper interloper rejection outside a protected core region.

        The input galaxies are sorted by projected radius and divided into radial
        bins containing approximately ``npbin`` galaxies each. For each bin that
        contains more than 5 galaxies and lies entirely outside ``coremin``, the
        galaxies are sorted by line-of-sight velocity and iteratively clipped using
        a velocity-gap criterion.

        In each iteration, the characteristic velocity scale is estimated from the
        interquartile velocity range,

            gap = (v_75 - v_25) / 1.349,

        where the factor 1.349 converts the Gaussian interquartile range to an
        estimate of the standard deviation. If this estimated gap is smaller than
        ``gap_use``, no further clipping is applied in that bin. If the estimated
        gap is more than twice the gap used for the previous accepted bin, it is
        capped at the previous value to avoid abrupt changes in the clipping
        threshold.

        The clipping is performed separately below and above zero velocity. On the
        negative-velocity side, galaxies below the largest significant velocity gap
        are rejected. On the positive-velocity side, galaxies above the first
        significant velocity gap are rejected. The process repeats until the bin
        size no longer decreases or fewer than 5 galaxies remain.

        Galaxies with projected radius ``r_proj < coremin`` are restored from the
        original input catalog at the end, so the core region is not clipped. Unlike
        a version that clips all bins and restores the core afterward, this function
        only clips bins whose minimum radius is greater than ``coremin``. Therefore,
        bins that straddle the core boundary are left unmodified.

        Note that this is more of a "legacy" function for the older shiftgapper,
        we don't use the adaptive version.

        Parameters
        ----------
        data : ndarray, shape (N, 2)
            Galaxy phase-space data. Column 0 is projected radius ``r_proj`` and
            column 1 is line-of-sight velocity ``v_los`` in km/s. The projected
            radius should be in the same units as ``coremin``.

        npbin : int
            Approximate number of galaxies per radial bin.

        gap_use : float
            Minimum velocity-gap threshold in km/s. This is also used as the floor
            for the adaptive gap threshold between bins.

        coremin : float
            Projected radius inside which galaxies are protected from clipping.
            Radial bins are clipped only if their minimum projected radius is
            greater than this value.

        Returns
        -------
        datafinal : ndarray, shape (N_keep, 2)
            Filtered array of ``(r_proj, v_los)`` values after shifting-gapper
            interloper rejection outside the protected core region.
        """
        gap_prev = gap_use #initialize gap size for initial comparison (must be larger to start).
        nbins = np.int64(np.ceil(data[:,0].size/(npbin*1.0)))
        origsize = data[:,0].shape[0]
        data = data[np.argsort(data[:,0])] # sort by r to ready for binning
        for i in range(nbins):
            databin = data[npbin*i:npbin*(i+1)]
            datanew = None
            nsize = databin[:,0].size
            datasize = nsize-1
            if ((nsize > 5) & (np.min(databin[:,0]) >coremin)):
                while nsize - datasize > 0 and datasize >= 5:
                    nsize = databin[:,0].size
                    databinsort = databin[np.argsort(databin[:,1])] #sort by v
                    f = (databinsort[:,1])[databinsort[:,1].size-np.int64(np.ceil(databinsort[:,1].size/4.0))]-(databinsort[:,1])[np.int64(np.ceil(databinsort[:,1].size/4.0))]
                    gap = f/(1.349) #F rom Gaussian for large data
                    if gap < gap_use: break
                    if gap >= 2.0*gap_prev:
                        gap = gap_prev
                    databelow = databinsort[databinsort[:,1]<=0]
                    gapbelow =databelow[:,1][1:]-databelow[:,1][:-1]
                    dataabove = databinsort[databinsort[:,1]>0]
                    gapabove = dataabove[:,1][1:]-dataabove[:,1][:-1]
                    try:
                        if np.max(gapbelow) >= gap: vgapbelow = np.where(gapbelow >= gap)[0][-1]
                        else: vgapbelow = -1
                        try:
                            datanew = np.append(datanew,databelow[vgapbelow+1:],axis=0)
                        except:
                            datanew = databelow[vgapbelow+1:]
                    except ValueError:
                        pass
                    try:
                        if np.max(gapabove) >= gap: vgapabove = np.where(gapabove >= gap)[0][0]
                        else: vgapabove = 99999999
                        try:
                            datanew = np.append(datanew,dataabove[:vgapabove+1],axis=0)
                        except:
                            datanew = dataabove[:vgapabove+1]
                    except ValueError:
                        pass
                    databin = datanew
                    datasize = datanew[:,0].size
                    datanew = None
                if gap >=gap_use:
                    gap_prev = gap
                else:
                    gap_prev = gap_use

            try:
                datafinal = np.append(datafinal,databin,axis=0)
            except:
                datafinal = databin
        w1 = np.where(data[:, 0] < coremin)[0]
        w2 = np.where(datafinal[:, 0] > coremin)[0]
        datafinal = np.array(data[w1].tolist() + datafinal[w2].tolist())

        return datafinal
    
    @staticmethod
    def calculate_projected_quantities(gal_ras, gal_decs, gal_z, cl_ra, cl_dec, cl_z, cosmo_params, case):
        """Compute projected radii and line-of-sight velocities for a galaxy sample.

        Parameters
        ----------
        gal_ras, gal_decs : array-like
            Galaxy right ascension and declination in degrees.
        gal_zs : array-like
            Galaxy redshifts.
        cl_ra, cl_dec : float
            Cluster center RA/Dec in degrees.
        cl_z : float
            Cluster redshift.

        Returns
        -------
        r_proj : ndarray
            Projected radii in physical units (Mpc), using the angular diameter distance.
        v_los : ndarray
            Line-of-sight velocities in km/s, computed relative to the cluster redshift.

        Notes
        -----
        Distances are computed using the ``D_A`` helper from ``escape_theory_functions``.
        """
        d_A = D_A(cl_z, cosmo_params, case).value
        sep = angular_separation(
            np.radians(cl_ra), np.radians(cl_dec),
            np.radians(gal_ras), np.radians(gal_decs)
        )
        R_proj = sep * d_A  # [Mpc]
        c_kms = const.c.value / 1000.0
        v_los = c_kms * (gal_z - cl_z) / (1.0 + cl_z)
        return R_proj, v_los

    def iterate_center(
        self, gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z, R200, min_r, max_r, cut, cosmo_params, case
    ):
        """Perform one iteration of cluster centering and sample selection.

        This routine:
          1) projects galaxies relative to the current center,
          2) applies radial and velocity cuts,
          3) optionally re-centers using a robust location estimator.

        Parameters
        ----------
        gal_ras, gal_decs, gal_zs : array-like
            Galaxy sky coordinates and redshifts.
        cl_ra, cl_dec, cl_z : float
            Current cluster center and redshift.
        R200_prop : float
            Proposed R200 (Mpc) used to scale the radial selection.
        min_r, max_r : float
            Radial selection bounds in units of R200 (e.g. 0.2–2.0).
        cut : float
            |v_los| cut in km/s.
        cosmo_params, case : see :func:`D_A`

        Returns
        -------
        gal_ras_new, gal_decs_new, gal_zs_new : ndarray
            Selected galaxies after this iteration.
        r_proj, v_los : ndarray
            Projected radii (Mpc) and LOS velocities (km/s) for the selected galaxies.
        """
        r_proj, v_los = self.calculate_projected_quantities(
            gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z, cosmo_params, case
        )

        mask_vlos = np.abs(v_los) < cut
        mask_r = (r_proj > min_r * R200) & (r_proj < max_r * R200)

        w = np.where(mask_vlos & mask_r)[0]

        r_proj = np.asarray(r_proj)[w]
        v_los = np.asarray(v_los)[w]

        return gal_ras[w], gal_decs[w], gal_zs[w], r_proj, v_los


    def iterate_center_N_times(self, gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,
                              R200_prop, R200_estimate, min_r, max_r, cut, cosmo_params, case, N_iterations=20,rmax_wl_factor=2):
        """Repeat :meth:`iterate_center` a fixed number of times.

        Parameters
        ----------
        N : int
            Number of iterations.
        *args, **kwargs
            Forwarded to :meth:`iterate_center`.

        Returns
        -------
        Same as :meth:`iterate_center`.
        """
        gal_ras_news = []
        gal_decs_news = []
        gal_zs_news = []

        for i in range(N_iterations):
            if i == 0:
                gal_ras_new, gal_decs_new, gal_zs_new, r_proj, v_los = self.iterate_center(gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,R200_prop, min_r, max_r, cut, cosmo_params, case)
                # before building the clamp mask that multiplies rmax_wl_factor * R200_estimate
                if R200_estimate is not None:
                    m = (r_proj < float(rmax_wl_factor) * float(R200_estimate))
                    if not np.any(m):
                        raise ValueError("No galaxies remain after WL clamp.")
                else:
                    # mass-free path: do NOT clamp by R200; only keep a wide annulus + LOS cut
                    R_lo_Mpc, R_hi_Mpc = 0.3, 3
                    m = (r_proj >= R_lo_Mpc) & (r_proj <= R_hi_Mpc) & (np.abs(v_los) <= cut)
                    if not np.any(m):
                        raise ValueError("No galaxies left after metric cuts; relax R_lo/R_hi or vcut_kms.")

                gal_ras_news.append(np.mean(gal_ras_new))
                gal_decs_news.append(np.mean(gal_decs_new))
                # Use bi-weight location with c=9
                gal_zs_news.append(biweight_location(gal_zs_new, c=9))

            else:
                gal_ras_new, gal_decs_new, gal_zs_new, r_proj, v_los = self.iterate_center(gal_ras, gal_decs,gal_zs, gal_ras_news[-1],gal_decs_news[-1], gal_zs_news[-1], R200_prop, min_r, max_r,cut, cosmo_params, case)
                gal_ras_news.append(np.mean(gal_ras_new))
                gal_decs_news.append(np.mean(gal_decs_new))
                gal_zs_news.append(biweight_location(gal_zs_new, c=9))

        return gal_ras_news, gal_decs_news, gal_zs_news, r_proj, v_los

    #Savgol smoother for edge profiles
    @staticmethod
    def _nan_savgol(y: np.ndarray, window: int = 5, polyorder: int = 2, anchor0: bool = True) -> np.ndarray:
            """Apply Savitzky–Golay smoothing while preserving NaNs.

            Parameters
            ----------
            y : array-like
                Input vector to smooth.
            window : int
                Savitzky–Golay window length (must be odd).
            polyorder : int
                Polynomial order.

            Returns
            -------
            ndarray
                Smoothed array with NaN positions preserved.
            """
            y = np.asarray(y, float)
            out = y.copy()

            m = np.isfinite(y)
            if np.count_nonzero(m) < (polyorder + 1):
                return out  # not enough points to fit

            # Fill NaNs by linear interp (only for filtering)
            x = np.arange(y.size)
            y_fill = y.copy()
            y_fill[~m] = np.interp(x[~m], x[m], y[m])

            # Ensure odd window and valid vs series length
            if window % 2 == 0:
                window += 1
            window = min(window, y.size if y.size % 2 == 1 else y.size - 1)
            window = max(window, polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2)

            ys = savgol_filter(y_fill, window_length=window, polyorder=polyorder, mode="interp")

            # Restore NaNs; optionally anchor the first finite value
            ys[~m] = np.nan
            if anchor0 and np.isfinite(y[0]):
                ys[0] = y[0]
            return ys

    #Polynomial smoother for edge profiles, anchors outer bins due to edge effects
    @staticmethod
    def _smooth_anchor_monotonic(v_raw: np.ndarray,
                                 anchor_first: bool = False,
                                 anchor_last: bool = False,
                                 protect_last_two: bool = False,
                                 poly_deg: int = 2) -> np.ndarray:
        """Smooth a 1D profile and optionally enforce monotonicity with anchors.

        Parameters
        ----------
        y : array-like
            Input profile.
        anchor_first, anchor_last : bool, default=False
            If True, keep the first/last element fixed after smoothing.
        poly_deg : int, default=2
            Polynomial order for Savitzky–Golay smoothing.

        Returns
        -------
        ndarray
            Smoothed profile.
        """

        v_raw = np.asarray(v_raw, float)
        out   = v_raw.copy()

        # finite mask
        m = np.isfinite(v_raw)
        if np.count_nonzero(m) < (poly_deg + 1):
            return out

        x = np.arange(v_raw.size, dtype=float)

        # global polynomial fit
        coeffs = np.polyfit(x[m], v_raw[m], deg=poly_deg)
        v_fit  = np.polyval(coeffs, x)

        # restore NaNs in places that had no data
        v_fit[~m] = np.nan

        if anchor_first and np.isfinite(v_raw[0]):
            v_fit[0] = v_raw[0]

        if anchor_last:
            finite_idxs = np.where(m)[0]
            if finite_idxs.size > 0:
                last_idx = finite_idxs[-1]
                # don't allow smoothing to push the outskirts LOWER
                if np.isfinite(v_raw[last_idx]):
                    v_fit[last_idx] = v_raw[last_idx]

            if protect_last_two and finite_idxs.size > 1:
                second_last_idx = finite_idxs[-2]
                if np.isfinite(v_raw[second_last_idx]):
                    v_fit[second_last_idx] = v_raw[second_last_idx]


        return v_fit

    def get_edge(
        self,
        bins,
        galaxy_r,
        galaxy_v,
        cl_z,
        R200,
        min_r,
        max_r,
        cut,
        cosmo_params,
        case,
        NON_INC=True,
        smooth=False,
        q=1,
        min_in_bin=5,
        window=5,
        polyorder=2,
        use_exact_r=True,
        fallback_to_center=True,
    ):
        """Estimate the escape edge in equal-width radial bins.

        If use_exact_r=True, the radius for each bin is set to the radius of the
        galaxy whose |v| is closest to the chosen high-|v| quantile (or max for q=1).

        """

        def _cap_rises(edge: np.ndarray) -> np.ndarray:
            e = np.asarray(edge, float).copy()
            for i in range(1, e.size):
                if np.isfinite(e[i-1]) and np.isfinite(e[i]):
                    e[i] = min(e[i], e[i-1])
            return e

        edges_r = np.linspace(min_r * R200, max_r * R200, bins + 1)
        centers = 0.5 * (edges_r[:-1] + edges_r[1:])

        m = (
            np.isfinite(galaxy_r) &
            np.isfinite(galaxy_v) &
            (galaxy_r >= edges_r[0]) &
            (galaxy_r <= edges_r[-1]) &
            (np.abs(galaxy_v) < float(cut))
        )
        r = np.asarray(galaxy_r[m], float)
        v = np.asarray(galaxy_v[m], float)

        edge   = np.full(bins, np.nan, float)
        r_edge = np.full(bins, np.nan, float)
        N_eff  = np.zeros(bins, dtype=int)

        min_req = max(1, int(min_in_bin))
        q = float(q)

        for i in range(bins):
            left, right = edges_r[i], edges_r[i + 1]

            # Use half-open bins for consistency, include the right edge only in the last bin
            if i < bins - 1:
                sel = (r >= left) & (r < right)
            else:
                sel = (r >= left) & (r <= right)

            idx = np.where(sel)[0]
            n_i = idx.size
            N_eff[i] = n_i

            if n_i >= min_req:
                vv = np.abs(v[idx])

                # edge value
                edge_val = float(np.quantile(vv, q, method="linear"))
                edge[i] = edge_val

                # exact radius corresponding to the "edge" event/statistic
                if use_exact_r:
                    # pick the galaxy whose |v| is closest to the quantile value
                    j = int(np.argmin(np.abs(vv - edge_val)))
                    r_edge[i] = float(r[idx[j]])
                else:
                    r_edge[i] = float(centers[i])
            else:
                edge[i] = np.nan
                r_edge[i] = np.nan

        if NON_INC:
            edge = _cap_rises(edge)


        if smooth:
            edge_sm = self._nan_savgol(edge, window=window, polyorder=polyorder)
            edge = edge_sm
            if NON_INC:
                edge = _cap_rises(edge)

        out_r = r_edge.copy()
        if fallback_to_center:
            bad = ~np.isfinite(out_r)
            out_r[bad] = centers[bad]

        vesc_data_r = out_r.reshape(1, bins)
        theta = ((out_r * u.Mpc) / D_A(cl_z, cosmo_params, case)) * u.rad
        vesc_data_theta = theta.to(u.arcmin)
        vesc_data = edge.reshape(1, bins)

        return vesc_data_r, vesc_data_theta, vesc_data, N_eff



class MCMCMassEstimator:
    """Single-cluster Bayesian mass inference from escape edges.

    This estimator samples the posterior of ``log10(M200)`` (and optionally additional
    nuisance parameters) for a single cluster by comparing a measured edge profile to
    the theoretical escape profile. The mapping from theory to data includes the Z_v
    suppression factor. 

    Z_v treatment
    -------------
    Two modes are supported:
      - **Marginalized Z_v:** integrate over per-bin skew‑t Z_v distributions using
        deterministic nodes/weights from :meth:`EscapeVelocityModeling.Zv_quantile_nodes`.
      - **Stochastic Z_v:** draw Z_v samples and condition on them (legacy option).

    Likelihood options
    ------------------
    * Gaussian likelihood with a per-bin floor ``vesc_error_floor`` (km/s).
    * Optional Student‑t likelihood (robust to outliers).

    The MCMC driver uses ``emcee`` and supports multiprocessing via ``Pool``.
    """

    def __init__(
        self,
        escape_modeler, cluster_positional_data, galaxy_positional_data,
        coremin_cut, cut, bins,
        R200_estimate, log10M200_min, log10M200_max,
        cosmo_params, cosmo_name, DirectDataPass, vesc_error_floor,
        fix_R200, NON_INC, smooth,
        student_t=False,
    ):
        """Initialize the single-cluster mass estimator.

        Parameters
        ----------
        escape_modeler : EscapeVelocityModeling
            Provider of the theoretical escape profile and Z_v calibration.
        cluster_positional_data : sequence
            Cluster metadata, typically ``(ra_deg, dec_deg, z)``.
        galaxy_positional_data : ndarray
            Either an (N,3) array of (ra_deg, dec_deg, z) for raw galaxy catalogs, or an
            already-preprocessed array depending on ``DirectDataPass``.
        coremin_cut : float
            Core exclusion factor for shifting-gapper interloper rejection.
        cut : float
            |v_los| cut applied to the galaxy sample (km/s).
        bins : int
            Number of radial bins for edge extraction.
        R200_estimate : float
            Initial R200 estimate (Mpc) used for preprocessing.
        log10M200_min, log10M200_max : float
            Uniform prior bounds on log10(M200/Msun).
        cosmo_params, cosmo_name :
            Passed through to theory utilities in ``escape_theory_functions``.
        DirectDataPass : bool
            If True, ``galaxy_positional_data`` is assumed to already contain projected
            quantities required by edge extraction.
        vesc_error_floor : float
            Minimum per-bin uncertainty (km/s) used in the likelihood.
        fix_R200 : bool
            If True, keep R200 fixed to ``R200_estimate`` rather than recomputing from
            the proposed mass.
        NON_INC : bool
            Whether to enforce non-increasing edges in extraction.
        smooth : bool
            Whether to smooth the measured edge across bins.
        student_t: bool
            Whether to use a student-t (default) or Gaussian likelihood
        """
        self.escape_modeler = escape_modeler
        self.cluster_positional_data = cluster_positional_data
        self.galaxy_positional_data = galaxy_positional_data
        self.coremin_cut = coremin_cut
        self.cut = cut
        self.bins = bins
        self.R200_estimate = R200_estimate
        self.log10M200_min = log10M200_min
        self.log10M200_max = log10M200_max
        self.cosmo_params = cosmo_params
        self.cosmo_name = cosmo_name
        self.DirectDataPass = DirectDataPass
        self.vesc_error_floor = vesc_error_floor
        self.fix_R200 = fix_R200
        self.NON_INC = NON_INC
        self.smooth    = smooth
        self.student_t = student_t
        
        self.cl_z = float(self.cluster_positional_data[2])
        self.qH2 = (
            q_z_function(z=self.cl_z, cosmo_params=self.cosmo_params, case=self.cosmo_name)
            * (H_z_function(z=self.cl_z, cosmo_params=self.cosmo_params, case=self.cosmo_name).value ** 2)
        ) / ((self.cosmo_params[1] * 100) ** 2)
        self.rho_c = rho_crit_z(
            self.cl_z, self.cosmo_params, self.cosmo_name
        ).to(u.Msun / u.Mpc**3).value


        if self.fix_R200:
            info = _resolve_valid_initial_center(
                self.cluster_positional_data, self.galaxy_positional_data,
                self.R200_estimate,
                self.coremin_cut, self.cut, self.bins,
                self.cosmo_params, self.cosmo_name, self.DirectDataPass,
                self.NON_INC, self.smooth,
                N_min=5, N_max=320,
            )
            if not info["ok"]:
                raise InvalidSampleError(info["reason"])

            self.R200_estimate = info["R200_center"]

            self._cache = self._build_cache()
            


    def _build_cache(self):
        """Precompute all invariant likelihood inputs for fix_R200=True."""

        try:
            (
                galaxy_r, galaxy_v, N_hat,
                vesc_data_r, vesc_data_theta, vesc_data, _,
                galaxy_r_with_interlopers, galaxy_v_with_interlopers
            ) = mass_estimation_preprocessing(
                self.cluster_positional_data,
                self.galaxy_positional_data,
                self.R200_estimate,
                self.R200_estimate,
                self.coremin_cut, self.cut, self.bins,
                self.cosmo_params, self.cosmo_name,
                self.DirectDataPass,
                self.NON_INC,
                self.smooth,
                validate_N=True
            )
        except ValueError as e:
            raise InvalidSampleError(str(e))
    
        cache = {
            "N_hat": np.asarray(N_hat, float),
            "vesc_data_theta": vesc_data_theta,
            "y_obs": np.asarray(vesc_data, float).reshape(-1),
        }
        cache["sigma0"] = np.full_like(cache["y_obs"], float(self.vesc_error_floor), dtype=float)


        Z_nodes, logW = self.escape_modeler.Zv_quantile_nodes(
            cache["N_hat"], self.bins, self.qH2
        )
        cache["Z_nodes"] = np.asarray(Z_nodes, float)
        cache["logW"] = np.asarray(logW, float)
        cache["logW"] = cache["logW"] - logsumexp(cache["logW"], axis=0, keepdims=True)

        return cache
        

    def lnprior(self, omega):
        """Log prior for the mass parameters.

        Parameters
        ----------
        omega : array-like
            Parameter vector. The first element is interpreted as log10(M200/Msun).

        Returns
        -------
        float
            Log prior probability (0.0 for inside bounds, -inf otherwise).
        """
        p_log10M200 = omega[0]
        if not (self.log10M200_min < p_log10M200 < self.log10M200_max):
            return -np.inf
        return 0.0


    def lnlike(self, omega):
        """Log likelihood for the measured edge profile.

        The likelihood compares the measured edge ``v_esc,obs(r)`` to the theoretical
        prediction ``v_esc,th(r; M200)`` mapped through the Z_v suppression:

            v_esc,obs(r) ≈ v_esc,th(r; M200) / Z_v(r)

        In the default (marginalized) mode, the per-bin likelihood is integrated over the
        skew‑t distribution for Z_v using deterministic nodes and weights.

        Parameters
        ----------
        omega : array-like
            Parameter vector. ``omega[0]`` is log10(M200/Msun).

        Returns
        -------
        float
            Log likelihood value.
        """
        log10M = float(omega[0])
        M_prop = 10.0 ** log10M
        cl_z = self.cl_z
        qH2 = self.qH2
        rho_c = self.rho_c
        
    
        if self.fix_R200:
            cache = self._cache
            N_hat = cache["N_hat"]
            vesc_data_theta = cache["vesc_data_theta"]
            y_obs = cache["y_obs"]
            sigma0 = cache["sigma0"]
            R200_prop = self.R200_estimate   # Propposal r200 is always the initial one when fix_R200=True
            
        else:
            # When we don't fix R200, need to dynamically determine the N and edge data with the proposal M200
            R200_prop = (3.0 * M_prop / (200.0 * 4.0 * np.pi * rho_c)) ** (1.0 / 3.0)

            try:
                (
                    galaxy_r, galaxy_v, N_hat,
                    vesc_data_r, vesc_data_theta, vesc_data, _,
                    galaxy_r_with_interlopers, galaxy_v_with_interlopers
                ) = mass_estimation_preprocessing(
                    self.cluster_positional_data,
                    self.galaxy_positional_data,
                    R200_prop,
                    self.R200_estimate,
                    self.coremin_cut, self.cut, self.bins,
                    self.cosmo_params, self.cosmo_name,
                    self.DirectDataPass,
                    self.NON_INC,
                    self.smooth,
                    validate_N=True
                )
            except ValueError:
                return -np.inf

            y_obs = np.asarray(vesc_data, float).reshape(-1)
            N_hat = np.asarray(N_hat, float)
            sigma0 = np.full_like(y_obs, float(self.vesc_error_floor), dtype=float)
            
        _, y_th = self.escape_modeler.v_esc_den_M200(
            vesc_data_theta, np.repeat(cl_z, 1), np.repeat(M_prop, 1),
            self.cosmo_params, self.cosmo_name
        )
        y_th = np.asarray(y_th, float).reshape(-1)

        # We have two treatments for Zv, one is the marginalization via integration and the other is direct sampling
        


        if self.fix_R200:
            Z_nodes = cache["Z_nodes"]
            logW = cache["logW"]
        else:
            Z_nodes, logW = self.escape_modeler.Zv_quantile_nodes(N_hat, self.bins, qH2)
            Z_nodes = np.asarray(Z_nodes, float)
            logW = logW - logsumexp(logW, axis=0, keepdims=True)

        mu_kj = y_th[None, :] / Z_nodes
        ymodel = mu_kj


        # Optional: instead of Gaussian likelihood, use Student-t likelihood for outlier data point down-weighting
        if self.student_t:
            nu = 5.0  # degrees of freedom
            sigma = sigma0[None, :]
            resid = y_obs[None, :] - ymodel

            term1 = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
            term2 = -0.5 * np.log(nu * np.pi) - np.log(sigma)
            term3 = -0.5 * (nu + 1.0) * np.log1p((resid**2) / (nu * sigma**2))
            core = term1 + term2 + term3

        # Default: Gaussian likelihood
        else:
            sigma2 = (sigma0)**2
            core  = -0.5*((y_obs[None,:] - ymodel)**2 / sigma2[None,:]) \
                    - 0.5*np.log(2.0*np.pi*sigma2[None,:])

        ll_j =  logsumexp(core + logW, axis=0)
        ll = float(np.sum(ll_j))
        return ll if np.isfinite(ll) else -np.inf


    def lnprob(self, omega):
        """Log posterior = lnprior + lnlike.

        Parameters
        ----------
        omega : array-like
            Parameter vector.

        Returns
        -------
        float
            Log posterior probability.
        """
        lp = self.lnprior(omega)
        if not np.isfinite(lp):
            return -np.inf

        ll = self.lnlike(omega)
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    def fit(self, grid_K=64, maxiter=200, tol=1e-3, n_random_starts=0, run_preflight=True):
        """
        Compute a 1D MAP estimate of `log10(M200/Msun)` for a single cluster.
        """
        original_R200_estimate = self.R200_estimate

        if run_preflight:
            info = _resolve_valid_initial_center(
                self.cluster_positional_data, self.galaxy_positional_data,
                self.R200_estimate,
                self.coremin_cut, self.cut, self.bins,
                self.NON_INC, self.cosmo_name, self.DirectDataPass,
                self.NON_INC, self.smooth,
                N_min=5, N_max=320,
            )
            if not info["ok"]:
                raise InvalidSampleError(info["reason"])

            if info["R200_center"] is not None:
                self.R200_estimate = info["R200_center"]

        try:
            a, b = self.log10M200_min, self.log10M200_max

            grid = np.linspace(a, b, int(grid_K))
            vals = np.array([self.lnprob([x]) for x in grid], float)
            if not np.any(np.isfinite(vals)):
                raise RuntimeError("fit(): no finite posterior values on coarse grid.")

            i0 = int(np.nanargmax(vals))
            best_x = float(grid[i0])
            best_val = float(vals[i0])

            seeds = [best_x]
            if n_random_starts > 0:
                seeds.extend(list(np.random.uniform(a, b, size=int(n_random_starts))))

            def _negpost(x):
                val = self.lnprob([float(x)])
                return 1.0e100 if not np.isfinite(val) else -val

            nit_total = 0
            for s in seeds:
                res = scipy.optimize.minimize_scalar(
                    _negpost,
                    bounds=(a, b),
                    method="bounded",
                    options={"xatol": tol, "maxiter": int(maxiter)}
                )
                nit_total += int(getattr(res, "nfev", 0))
                if res.success:
                    val = -float(res.fun)
                    if np.isfinite(val) and val > best_val:
                        best_val = val
                        best_x = float(res.x)

            hstep = max(1.0e-4, 1.0e-2 * max(1.0, abs(best_x)))

            def _g(x):
                return self.lnprob([float(x)])

            try:
                d2 = (_g(best_x + hstep) - 2 * _g(best_x) + _g(best_x - hstep)) / (hstep * hstep)
            except Exception:
                d2 = np.nan

            sigma_log10M = np.nan
            if np.isfinite(d2) and d2 < 0:
                sigma_log10M = float(np.sqrt(1.0 / (-d2)))

            z = float(self.cluster_positional_data[2])
            rho_c = rho_crit_z(z, self.cosmo_params, self.cosmo_name).value
            M200_map = 10.0 ** best_x
            R200_map = (3.0 * M200_map / (200.0 * 4.0 * np.pi * rho_c)) ** (1.0 / 3.0)

            result = {
                "log10M_map": float(best_x),
                "M200_map": float(M200_map),
                "R200_map_Mpc": float(R200_map),
                "sigma_log10M": sigma_log10M,
                "M200_1sigma_minus": float(10.0 ** (best_x - sigma_log10M)) if np.isfinite(sigma_log10M) else np.nan,
                "M200_1sigma_plus": float(10.0 ** (best_x + sigma_log10M)) if np.isfinite(sigma_log10M) else np.nan,
                "post_at_map": float(best_val),
                "success": np.isfinite(best_val),
                "nit": int(nit_total),
            }

            if run_preflight:
                result["preflight_dlog10M_used"] = info["dlog10M_used"]
                result["preflight_R200_used"] = info["R200_center"]

            self.result_map = result
            return result

        finally:
            # restore original center so fit() does not mess with the estimator
            self.R200_estimate = original_R200_estimate

    
def _resolve_valid_initial_center(
    cluster_positional_data, galaxy_positional_data,
    R200_estimate,
    coremin_cut, cut, bins,
    cosmo_params, cosmo_name, DirectDataPass,
    NON_INC, smooth,
    dlog10M_trials=(0.0, -0.05, 0.05, -0.1, 0.1),
    N_min=5,
    N_max=320,
):
    """
    Try the supplied center first, then nearby log-mass nudges, and return
    the first center that passes preflight validation.

    Returns
    -------
    info : dict
        {"ok", "log10M_center", "R200_center", "dlog10M_used", "reason"}
    """
    if R200_estimate is None:
        return {"ok": True, "log10M_center": None, "R200_center": None,
                "dlog10M_used": None, "reason": None}

    z = float(cluster_positional_data[2])
    rho_c = rho_crit_z(z, cosmo_params, cosmo_name).value

    M200_center = (4.0 * np.pi / 3.0) * 200.0 * rho_c * (float(R200_estimate) ** 3)
    log10M_center = np.log10(M200_center)

    last_reason = None
    for dlog10M in dlog10M_trials:
        log10M_try = log10M_center + float(dlog10M)
        R200_try = (3.0 * (10.0 ** log10M_try) / (200.0 * 4.0 * np.pi * rho_c)) ** (1.0 / 3.0)

        try:
            preflight_validate_or_raise(
                cluster_positional_data, galaxy_positional_data,
                R200_try, coremin_cut, cut, bins,
                cosmo_params, cosmo_name, DirectDataPass,
                N_min=N_min, N_max=N_max,
                NON_INC=NON_INC, smooth=smooth,
            )
            return {"ok": True,
                    "log10M_center": float(log10M_try),
                    "R200_center": float(R200_try),
                    "dlog10M_used": float(dlog10M),
                    "reason": None}
        except InvalidSampleError as e:
            last_reason = str(e)

    return {"ok": False, "log10M_center": None, "R200_center": None,
            "dlog10M_used": None, "reason": last_reason}

def mass_estimation_preprocessing(
    cluster_positional_data, galaxy_positional_data,
    R200_prop, R200_estimate, coremin_cut, cut, bins,
    cosmo_params, cosmo_name, DirectDataPass, NON_INC, smooth, validate_N=True
):


    """
    Preprocess a cluster galaxy catalog into edge-ready phase-space quantities.

    This function implements the *data* part of the escape-velocity pipeline and is
    shared by both the mass-only and cosmology inference pathways. It takes an input
    galaxy catalog and returns:

    - the cleaned phase-space sample `(r_proj, v_los)` after interloper rejection, and
    - the binned escape edge `(r_edge, v_edge)` used by the likelihood, including the
      per-bin tracer counts `N_hat` that drive the Z_v calibration.

    In this updated pipeline, two details are especially important for reproducibility:

    1. **N_hat definition (no explicit mass dependence in Z_v):**
       `N_hat` is computed *per radial bin* (after the same selection/interloper logic
       used for the edge). The Z_v calibration is parameterized in terms of this
       per-bin sampling, eliminating the need for an explicit mass dependence.

    2. **Edge radial coordinate (bin-center → exact r):**
       The edge routine returns the **exact radial position** of the galaxy that sets
       the escape boundary in each bin (rather than the bin center). This improves
       consistency between the observed edge and the theoretical prediction evaluated
       at the same radii.

    Processing stages
    -----------------
    If `DirectDataPass=False` (default):

    1. Project sky coordinates to projected radius using the supplied cosmology.
    2. Convert redshifts to LOS velocities relative to the cluster redshift.
    3. Apply an absolute velocity cut (`|v_los| < cut`).
    4. Iteratively re-center the cluster using the galaxies (to reduce miscentering).
    5. Remove interlopers with the shifting-gapper algorithm.
    6. Measure the escape edge with `bins` radial bins.

    If `DirectDataPass=True`, the function skips steps (1)–(4) and assumes the caller
    already supplied projected radii and LOS velocities in the expected units.

    Parameters
    ----------
    cluster_positional_data : tuple of (float, float, float)
        `(cl_ra_deg, cl_dec_deg, cl_z)` in decimal degrees and dimensionless redshift.

    galaxy_positional_data : array-like
        If `DirectDataPass=False`, array of shape `(N, 3)` with columns
        `[RA_deg, DEC_deg, z]`.

        If `DirectDataPass=True`, array containing projected quantities. The minimal
        expectation is that it can be interpreted as `(r_proj_Mpc, v_los_kms)` per row.

    R200_prop : float
        Current/propagated `R200` (Mpc). This is used to set radial selection ranges,
        the shifting-gapper core definition, and (by default) the edge binning domain.

    R200_estimate : float or None
        An external/reference `R200` (Mpc), used in some call paths for scaling or
        for consistency checks. If not available, pass None.

    coremin_cut : float
        Core exclusion factor for shifting-gapper interloper rejection, expressed as a
        fraction of `R200_prop`.

    cut : float
        Absolute LOS velocity cut (km/s) for selecting galaxies used in the edge.

    bins : int
        Number of radial bins used to measure the edge.

    cosmo_params, cosmo_name : sequence, str
        Cosmology used for projections and `rho_crit(z)` computations.

    DirectDataPass : bool
        Skip sky→projected conversions if True.

    NON_INC : bool
        Enforce non-increasing constraint of the measured edges (guards against unphysical upturns due to sampling noise).

    smooth : bool
        If True, apply the optional smoothing logic in the edge finder
        (:meth:`ClusterDataHandler.get_edge`).

    validate_N : bool, default=True
        If True, validate that `N_hat` lies in the calibrated range used by the Z_v
        model (default range is typically 5–320 galaxies per bin). A failure raises
        a ValueError (caught by higher-level wrappers).

    Returns
    -------
    galaxy_r : ndarray, shape (N_keep,)
        Projected radii (Mpc) of galaxies kept after interloper rejection.

    galaxy_v : ndarray, shape (N_keep,)
        LOS velocities (km/s) of galaxies kept after interloper rejection.

    N_hat : ndarray, shape (bins,)
        Per-bin effective tracer counts used by the edge finder and Z_v calibration.

    vesc_data_r : ndarray, shape (1, bins)
        Radial positions (Mpc) at which the escape edge is measured. In this pipeline
        these correspond to the *exact* radius of the edge-setting galaxy in each bin.

    vesc_data_theta : ndarray, shape (1, bins)
        Angular positions (radians) corresponding to `vesc_data_r` at the cluster
        redshift (useful when the theory model is evaluated on an angular grid).

    vesc_data : ndarray, shape (1, bins)
        Measured escape edge velocities (km/s), i.e. `|v_edge|` per radial bin.

    cl_z : float
        The (possibly re-centered) cluster redshift used to define LOS velocities.

    galaxy_r_with_interlopers, galaxy_v_with_interlopers : ndarray
        Projected radii and velocities of galaxies flagged as interlopers by the
        shifting-gapper stage. These are useful for diagnostic plots.

    Notes
    -----
    - The exact selection logic (radial range, binning domain, and gapper parameters)
      is defined in the body of this function and in :class:`ClusterDataHandler`.
    - If you change `bins`, you must regenerate the Z_v calibration for the same bin
      definition, otherwise the suppression model will be inconsistent.

    Examples
    --------
    >>> out = mass_estimation_preprocessing(
    ...     cluster_positional_data=(150.25, -10.75, 0.23),
    ...     galaxy_positional_data=galaxy_data,      # (N,3) RA,Dec,z
    ...     R200_prop=1.8, R200_estimate=1.8,
    ...     coremin_cut=0.44, cut=4500, bins=10,
    ...     cosmo_params=[0.3, 0.7], cosmo_name="FlatLambdaCDM",
    ...     DirectDataPass=False, NON_INC=True, smooth=False
    ... )
    >>> galaxy_r, galaxy_v, N_hat, r_edge, theta_edge, v_edge, cl_z, r_int, v_int = out
    """
    #change here
    min_r, max_r = 0.2,1.0
    Nbin, gap = 20, 600  # shifting-gapper params
    N_min, N_max = 5, 320

    data_handler = ClusterDataHandler()
    cl_ra, cl_dec, cl_z = cluster_positional_data

    if not DirectDataPass:

        gal_ras = galaxy_positional_data[:, 0].astype(float)
        gal_decs = galaxy_positional_data[:, 1].astype(float)
        gal_zs = galaxy_positional_data[:, 2].astype(float)

        # First pass centering/selection
        gal_ras_new, gal_decs_new, gal_zs_new, r_proj, v_los = data_handler.iterate_center(
            gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z, R200_prop, min_r, max_r, cut, cosmo_params, cosmo_name
        )
        if len(gal_ras_new) == 0:
            raise ValueError("Insufficient galaxy data after first centering pass.")

        # Refined centering with early exit and optional WL clamp
        (
            gal_ras_news, gal_decs_news, gal_zs_news,
            r_proj, v_los
        ) = data_handler.iterate_center_N_times(
            gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,
            R200_prop, R200_estimate, min_r, max_r, cut, cosmo_params, cosmo_name
        )

        cl_z = gal_zs_news[-1]

        # Ensure the re-centering has converged to within (default) 1% variations of theta_200
        def recenter_convergence_test(gal_ras_news, gal_decs_news, cl_z, R200,
                                      cosmo_params, cosmo_name, frac_tol=0.01):
            """Diagnostic: test convergence of the iterative centering routine.

            This nested helper is used to quantify how quickly repeated centering iterations
            converge in projected position and redshift.

            Returns
            -------
            dict
                Convergence diagnostics (contents depend on the calling context).
            """
            if len(gal_ras_news) < 2:
                return False, np.nan, np.nan, np.nan

            c_prev = SkyCoord(gal_ras_news[-2]*u.deg, gal_decs_news[-2]*u.deg)
            c_curr = SkyCoord(gal_ras_news[-1]*u.deg, gal_decs_news[-1]*u.deg)
            dtheta = (c_prev.separation(c_curr).to(u.rad)).value    # last-step shift

            theta200 = R200/D_A(cl_z, cosmo_params, cosmo_name).value
            frac = (dtheta / theta200)

            return (frac <= frac_tol), frac, dtheta, theta200

        converged, frac, dtheta, theta200 = recenter_convergence_test(gal_ras_news, gal_decs_news, cl_z, R200_prop, cosmo_params, cosmo_name)
        if not converged:
            raise ValueError(f"Cluster re-centering not converged: last shift = {frac:.3%} of θ_200 "
                             f"({dtheta:.3e} vs θ_200={theta200:.3e}). Either proceed with caution, increase N_iterations in iterate_center_N_times,  or increase frac_tol threshold criterion in recenter_convergence_test.")



    if DirectDataPass:
        # We need to ensure we filter data only between 0.2 and 1 R200 like iterate_center_N_times
        r_proj, v_los = galaxy_positional_data
        edges_lo = min_r * R200_prop
        edges_hi = max_r * R200_prop
        m_ann = (np.asarray(r_proj) >= edges_lo) & (np.asarray(r_proj) <= edges_hi)
        r_proj = np.asarray(r_proj, float)[m_ann]
        v_los  = np.asarray(v_los,  float)[m_ann]
        v_los = v_los - biweight_location(v_los, c=9) #re-center to rest frame velocity when we pass direct radius/velocity data


    # Interloper removal
    # We re-estimate R200_prop from the the ce-centered redshift, not that it should matter much.
    # Re-derive R200_prop from the re-centered redshift for consistency
    rho_crit        = rho_crit_z(cl_z, cosmo_params, cosmo_name).to(u.Msun / u.Mpc**3).value
    M200_recentered = (4.0 / 3.0) * np.pi * rho_crit * 200.0 * R200_prop**3
    R200_prop       = (3.0 * M200_recentered / (200.0 * 4.0 * np.pi * rho_crit)) ** (1.0 / 3.0)

    # Run the shifting-gapper interloper rejection
    data      = np.vstack((r_proj, v_los)).T
    coremin   = R200_prop * coremin_cut  # core exclusion radius
    datafinal = data_handler.shiftgapper(data, Nbin, gap, coremin)
    galaxy_r, galaxy_v = datafinal[:, 0], datafinal[:, 1]

    # Identify interloper galaxies for diagnostic use
    all_rec   = np.core.records.fromarrays([r_proj,    v_los],    names='r,v')
    keep_rec  = np.core.records.fromarrays([galaxy_r, galaxy_v], names='r,v')
    inter_rec = np.setdiff1d(all_rec, keep_rec)

    galaxy_r_with_interlopers = inter_rec['r']
    galaxy_v_with_interlopers = inter_rec['v']

    # Re-center the velocity frame after interloper removal
    v_offset   = biweight_location(galaxy_v, c=9)
    galaxy_v   = galaxy_v - v_offset

    vesc_data_r, vesc_data_theta, vesc_data, N = data_handler.get_edge(
        bins, galaxy_r, galaxy_v, cl_z, R200_prop, min_r, max_r, cut, cosmo_params, cosmo_name, NON_INC, smooth
    )
    if validate_N:
        N_arr = np.asarray(N, float)
        bad = (~np.isfinite(N_arr)) | (N_arr < N_min) | (N_arr > N_max)
        if np.any(bad):
            bad_bins = np.where(bad)[0]
            bad_vals = N_arr[bad]
            raise ValueError(
                f"N_hat outside calibrated range [{N_min}, {N_max}] in bins {bad_bins.tolist()}: "
                f"values={bad_vals.tolist()}"
            )

    vesc_data_r = np.asarray(vesc_data_r[0], float)
    vesc_data = np.asarray(vesc_data[0], float)
    vesc_data_theta = vesc_data_theta.to(u.radian).value.reshape(1, bins)

    return galaxy_r, galaxy_v, N, vesc_data_r, vesc_data_theta, vesc_data, cl_z,  galaxy_r_with_interlopers, galaxy_v_with_interlopers


def get_velocity_dispersion_data(
    cluster_positional_data, galaxy_positional_data,
    R200_estimate, coremin_cut, cut,
    cosmo_params, cosmo_name
):
    """Prepare a cleaned galaxy sample for velocity-dispersion measurements.

    This helper performs projection, selection, and shifting-gapper rejection, returning
    the projected radii and LOS velocities for a cluster. It is similar to
    :func:`mass_estimation_preprocessing` but does not compute an escape edge.

    Parameters
    ----------
    cluster_positional_data : sequence
        (ra_deg, dec_deg, z).
    galaxy_positional_data : ndarray, shape (N, 3)
        Galaxy (ra_deg, dec_deg, z).
    R200_estimate : float
        R200 estimate (Mpc).
    coremin_cut : float
        Core exclusion factor for shifting-gapper.
    cut : float
        |v_los| cut (km/s).
    cosmo_params, cosmo_name :
        Passed to theory utilities.

    Returns
    -------
    galaxy_r, galaxy_v : ndarray
        Projected radii (Mpc) and LOS velocities (km/s) after cleaning.
    """
    min_r, max_r = 0., 1.0
    Nbin, gap = 20, 600  # shifting-gapper params
    N_min, N_max = 5, 320

    data_handler = ClusterDataHandler()
    cl_ra, cl_dec, cl_z = cluster_positional_data

    R200_prop = R200_estimate

    gal_ras = galaxy_positional_data[:, 0].astype(float)
    gal_decs = galaxy_positional_data[:, 1].astype(float)
    gal_zs = galaxy_positional_data[:, 2].astype(float)


    # First pass centering/selection
    gal_ras_new, gal_decs_new, gal_zs_new, r_proj, v_los = data_handler.iterate_center(
        gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z, R200_prop, min_r, max_r, cut, cosmo_params, cosmo_name
    )
    if len(gal_ras_new) == 0:
        raise ValueError("Insufficient galaxy data after first centering pass.")

    # Refined centering with early exit and optional WL clamp
    (
        gal_ras_news, gal_decs_news, gal_zs_news,
        r_proj, v_los
    ) = data_handler.iterate_center_N_times(
        gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,
        R200_prop, R200_estimate, min_r, max_r, cut, cosmo_params, cosmo_name
    )

    cl_z = gal_zs_news[-1]

    # Ensure the re-centering has converged to within (default) 1% variations of theta_200
    def recenter_convergence_test(gal_ras_news, gal_decs_news, cl_z, R200,
                                  cosmo_params, cosmo_name, frac_tol=0.01):
        """Diagnostic: test convergence of the iterative centering routine.

        This nested helper is used to quantify how quickly repeated centering iterations
        converge in projected position and redshift.

        Returns
        -------
        dict
            Convergence diagnostics (contents depend on the calling context).
        """
        if len(gal_ras_news) < 2:
            return False, np.nan, np.nan, np.nan

        c_prev = SkyCoord(gal_ras_news[-2]*u.deg, gal_decs_news[-2]*u.deg)
        c_curr = SkyCoord(gal_ras_news[-1]*u.deg, gal_decs_news[-1]*u.deg)
        dtheta = (c_prev.separation(c_curr).to(u.rad)).value    # last-step shift

        theta200 = R200/D_A(cl_z, cosmo_params, cosmo_name).value
        frac = (dtheta / theta200)

        return (frac <= frac_tol), frac, dtheta, theta200

    converged, frac, dtheta, theta200 = recenter_convergence_test(gal_ras_news, gal_decs_news, cl_z, R200_prop, cosmo_params, cosmo_name)
    if not converged:
        raise ValueError(f"Cluster re-centering not converged: last shift = {frac:.3%} of θ_200 "
                         f"({dtheta:.3e} vs θ_200={theta200:.3e}). Either proceed with caution, increase N_iterations in iterate_center_N_times,  or increase frac_tol threshold criterion in recenter_convergence_test.")


    # Interloper removal
    # We re-estimate R200_prop from the the ce-centered redshift, not that it should matter much.
    rho_crit = rho_crit_z(cl_z, cosmo_params, cosmo_name).to(u.Msun / u.Mpc**3).value
    M200_recentered = (4/3) * np.pi * rho_crit * 200 * R200_prop**3
    R200_prop = (3.0 * M200_recentered / (200.0 * 4.0 * np.pi * rho_crit)) ** (1.0 / 3.0)

    data    = np.vstack((r_proj, v_los)).T
    coremin = R200_prop * coremin_cut  # core exclusion radius for shifting-gapper
    datafinal = data_handler.shiftgapper(data, Nbin, gap, coremin)
    galaxy_r, galaxy_v = datafinal[:, 0], datafinal[:, 1]

    v_offset = biweight_location(galaxy_v, c=9)
    # offset between center post interloper removal
    galaxy_v = galaxy_v - v_offset

    return galaxy_r, galaxy_v

# -----------------------------------------------------------------------------
# MCMC driver and post-processing for Mass Estimation
# -----------------------------------------------------------------------------

class InvalidSampleError(RuntimeError):
    """Raised when a cluster sample fails preprocessing or validation.

    This is used by the preflight stage to stop processing early for unusable inputs
    (e.g., empty selection, tracer counts outside calibration bounds, too few bins).
    """
    pass

def preflight_validate_or_raise(
    cluster_positional_data, galaxy_positional_data,
    R200_estimate,
    coremin_cut, cut, bins,
    cosmo_params, cosmo_name, DirectDataPass,
    #change here
    min_r=0.2, max_r=1.0,
    N_min=5, N_max=320,NON_INC=True, smooth=False):

    """Run a lightweight preprocessing pass and raise if the sample is unusable.

    Parameters
    ----------
    cluster_positional_data, galaxy_positional_data, R200_estimate, coremin_cut, cut, bins :
        See :func:`mass_estimation_preprocessing`.
    cosmo_params, cosmo_name, DirectDataPass :
        See :func:`mass_estimation_preprocessing`.
    min_r, max_r : float, default=(0.2, 2.0)
        Radial selection range in units of R200.
    N_min, N_max : int, default=(5, 320)
        Allowed per-bin sampling range for Z_v calibration.
    NON_INC, smooth : bool
        Edge extraction options.

    Returns
    -------
    dict
        Dictionary of preflight products (galaxy sample, edge data, N_hat, etc).

    Raises
    ------
    InvalidSampleError
        If preprocessing fails or the sample violates validation checks.
    """
    R200_pivot = R200_estimate  # use the supplied R200 as the preprocessing pivot
    try:
        (galaxy_r, galaxy_v, N_hat,
         vesc_data_r, vesc_data_theta, vesc_data, cl_z, galaxy_r_with_interlopers, galaxy_v_with_interlopers
        ) = mass_estimation_preprocessing(
            cluster_positional_data, galaxy_positional_data,
            R200_pivot, R200_estimate, coremin_cut, cut, bins,
            cosmo_params, cosmo_name, DirectDataPass,NON_INC,smooth)

    except ValueError as e:
        raise InvalidSampleError(f"Preflight failed in preprocessing: {e}")

    N_hat = np.asarray(N_hat, float)


    # build the same R200 used inside preprocessing at the centered z
    rho_crit = rho_crit_z(cl_z, cosmo_params, cosmo_name).to(u.Msun / u.Mpc**3).value
    M200_recentered = (4/3) * np.pi * rho_crit * 200 * R200_estimate**3
    R200 = (3.0 * M200_recentered / (200.0 * 4.0 * np.pi * rho_crit)) ** (1.0 / 3.0)


    # sanity on the edge arrays
    vesc_data_r = np.asarray(vesc_data_r[0], float)
    vesc_data   = np.asarray(vesc_data[0],   float)

    if (not np.all(np.isfinite(vesc_data_r))) or \
       (not np.all(np.isfinite(vesc_data))):
        raise InvalidSampleError("Edge arrays contain non-finite values (insufficient sampling).")

    return dict(
        cl_z=cl_z, R200=R200, N_hat=N_hat,
        galaxy_r=galaxy_r, galaxy_v=galaxy_v,
        vesc_data_r=vesc_data_r, vesc_data=vesc_data)

def run_mcmc_mass_estimation(
    escape_modeler,
    cluster_positional_data,
    galaxy_positional_data,
    coremin_cut,
    cut,
    bins,
    M200_estimate,
    log10M200_min,
    log10M200_max,
    cosmo_params,
    cosmo_name,
    DirectDataPass,
    vesc_error_floor,
    nwalkers=100,
    nsteps=1000,
    n_processes=None,
    progress=True,
    fix_R200=False,
    NON_INC=True,
    smooth=False
):
    """
    Run the single-cluster `emcee` sampler for `log10(M200/Msun)` using the escape-edge likelihood.

    This function is the computational core of the *mass-only* inference problem:
    given a cluster galaxy catalog, it preprocesses the data (unless already projected),
    measures the escape edge, and samples the posterior of the cluster mass.

    Workflow
    --------
    1. (Optional) **Preflight validation** via :func:`preflight_validate_or_raise` to
       ensure the catalog is usable (enough tracers, valid N_hat range, non-empty edge).
    2. Construct :class:`MCMCMassEstimator`, which encapsulates the prior, likelihood,
       and the Z_v marginalization machinery.
    3. Initialize walkers uniformly in `[log10M200_min, log10M200_max]`.
    4. Run an `emcee.EnsembleSampler` in parallel using a multiprocessing pool.
    5. Discard a default burn-in fraction (currently 20%) and return flattened samples
       with basic summary statistics.

    Z_v treatment
    -------------
    The likelihood used by :class:`MCMCMassEstimator` maps the theoretical edge through
    the Z_v suppression and, in this updated pipeline, *marginalizes over* Z_v using a
    skew‑t calibration. The calibration coordinate uses `N_hat` (per-bin counts) and
    `qH2` (cosmology dependence) rather than an explicit mass dependence.

    Parameters
    ----------
    escape_modeler : EscapeVelocityModeling
        Provider of the theoretical escape curve and calibrated Z_v model.

    cluster_positional_data, galaxy_positional_data : see :func:`MassEstimator_main`
        Cluster metadata and galaxy catalog.

    coremin_cut, cut, bins : float, float, int
        Preprocessing/interloper/edge settings passed to
        :func:`mass_estimation_preprocessing`.

    M200_estimate : float
        Initial estimate of `log10(M200/Msun)` used to obtain an initial `R200` for
        preprocessing (unless `fix_R200=True`).

    log10M200_min, log10M200_max : float
        Prior bounds for the mass parameter sampled by `emcee`.

    cosmo_params, cosmo_name : see :func:`MassEstimator_main`
        Cosmology used for distances and critical density.

    DirectDataPass : bool
        If True, treat `galaxy_positional_data` as already in projected phase-space units.

    vesc_error_floor : float
        Error floor (km/s) used in the per-bin edge likelihood.

    nwalkers : int, default=100
        Number of `emcee` walkers.

    nsteps : int, default=1000
        Steps per walker.

    n_processes : int or None, default=None
        Worker processes for the multiprocessing pool.

    progress : bool, default=True
        Show an `emcee` progress bar.

    fix_R200 : bool, default=False
        If True, keep `R200` fixed at the preprocessing value during likelihood evaluation.

    NON_INC : bool, default=True
        Enforce non-increasing edges during edge extraction.

    smooth : bool, default=False
        Enable optional bin-to-bin smoothing in edge extraction / Z_v smoothing hooks.

    Returns
    -------
    results : dict
        On success, returns:
          - `median`, `one_sig_down`, `one_sig_up` for `log10(M200/Msun)`
          - `samples` flattened posterior samples (shape `(Nsamp,)`)
          - `acceptance` mean acceptance fraction

        On failure (invalid sample), returns:
          - `status` = `"invalid"`
          - `reason` describing the failure
          - NaN summaries.

    Notes
    -----
    This function intentionally returns a compact results dictionary suitable for
    batching. If you need richer diagnostics (e.g., raw chains, per-bin residuals,
    edge arrays), use :class:`MCMCMassEstimator` directly or instrument the code
    around :func:`mass_estimation_preprocessing`.
    """
    # Make sure we are in range of data, otherwise abort MCMC early
    cl_z = cluster_positional_data[2]
    rho_c = rho_crit_z(cl_z, cosmo_params, cosmo_name).to(u.Msun/u.Mpc**3).value

    if M200_estimate is not None:
        # Fiducial WL-based R200
        R200_estimate = (3.0 * (10 ** M200_estimate) / (200.0 * 4.0 * np.pi * rho_c))**(1.0/3.0)

        # Preflight around the WL mass in case the exact radius is slightly unlucky
        dlog10M_trials = [0.0, -0.1, 0.1, -0.2, 0.2]

        preflight_ok = False
        preflight_reason = None
        R200_preflight_used = None
        M200_preflight_used = None

        for dlog10M in dlog10M_trials:
            M200_try = M200_estimate + dlog10M
            R200_try = (3.0 * (10 ** M200_try) / (200.0 * 4.0 * np.pi * rho_c))**(1.0/3.0)

            try:
                _pre = preflight_validate_or_raise(
                    cluster_positional_data, galaxy_positional_data,
                    R200_try,
                    coremin_cut, cut, bins,
                    cosmo_params, cosmo_name, DirectDataPass, NON_INC, smooth
                )
                preflight_ok = True
                R200_preflight_used = R200_try
                M200_preflight_used = M200_try
                break

            except InvalidSampleError as e:
                preflight_reason = str(e)

        if not preflight_ok:
            return {
                "status": "invalid",
                "reason": preflight_reason,
                "median": np.nan,
                "one_sig_down": np.nan,
                "one_sig_up": np.nan,
            }
        R200_estimate = float(R200_preflight_used)
        M200_estimate_preflight = float(M200_preflight_used)

    else:
        R200_estimate = None


    estimator = MCMCMassEstimator(
        escape_modeler, cluster_positional_data, galaxy_positional_data,
        coremin_cut, cut, bins, R200_estimate, log10M200_min, log10M200_max,
        cosmo_params, cosmo_name, DirectDataPass, vesc_error_floor,
        fix_R200=fix_R200, NON_INC=NON_INC, smooth=smooth
    )


    # Initialize walker positions uniformly across the prior
    p0   = np.transpose([np.random.uniform(log10M200_min, log10M200_max, size=nwalkers)])

    # Set up multiprocessing pool for parallel likelihood evaluations
    pool = Pool(processes=n_processes)

    try:
        # Initialize and run sampler
        ndim    = 1
        sampler = emcee.EnsembleSampler(nwalkers, ndim, estimator.lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=progress)

        burn = nsteps // 5
        thin = 1
        # --- Extract aligned posterior samples ---
        chain = sampler.get_chain(discard=burn, thin=thin, flat=True)       # (Nsamps, ndim)
        logp  = sampler.get_log_prob(discard=burn, thin=thin, flat=True)    # (Nsamps,)
        mask  = np.isfinite(logp)
        chain = chain[mask, :]
        logp  = logp[mask]

        if chain.size == 0:
            raise RuntimeError("No valid samples after burn-in/thinning; increase nsteps or check likelihood.")

        # If you only need log10 M200 (assumed column 0)
        samples = chain[:, 0]

        # --- Quick diagnostics & plot ---
        acc = np.mean(sampler.acceptance_fraction)
        med = np.median(samples)
        lo, hi = np.percentile(samples, [16, 84])

        print(f"mean acceptance: {acc:.3f}")
        print(f"log10 M200 = {med:.3f} (+{hi - med:.3f} / -{med - lo:.3f}) dex")

        plt.figure()
        plt.hist(samples, bins=20, density=True)
        plt.xlabel(r'Posterior $\log_{10} M_{200}/M_\odot$')
        plt.xlim(np.min(samples),np.max(samples))
        #plt.axvline(x=M200_estimate,c='r')
        plt.tight_layout()
        plt.show()

        # Central 68% interval
        one_sig_down, median, one_sig_up = np.quantile(samples, [0.158655, 0.5, 0.841345])


        print(
            'Escape Velocity Mass Estimate:',
            np.round(median, 2),
            '+', np.round(one_sig_up - median, 2),
            '-', np.round(median - one_sig_down, 2)
        )

        results = {
            "status": "ok",
            'median': median,
            'one_sig_up': one_sig_up,
            'one_sig_down': one_sig_down,
            'samples': samples,
            'acceptance': acc,
        }

        return results

    finally:
        pool.close()
        pool.join()


def mass_estimation_post_processing(
    escape_modeler,
    results,
    cluster_positional_data,
    galaxy_positional_data,
    M200_estimate,
    coremin_cut,
    cut,
    bins,
    cosmo_params,
    cosmo_name,
    fix_R200,
    DirectDataPass,
    vesc_error_floor,
    cluster_name,
    smooth_Zv=True,
    savefig_path=None,
    save_format="pdf",
    NON_INC=True,
    smooth=False
):
    """
    Post-process and visualize the output of :func:`run_mcmc_mass_estimation`.

    This routine converts the raw sampling output into the summary products you
    typically want to report or inspect:

    - posterior summaries (median and central credible interval),
    - diagnostics for acceptance and chain quality,
    - phase-space plots with the measured edge and theory curves,
    - optional mixture-model summaries for multi-modal posteriors.

    The function is designed to be called by :func:`MassEstimator_main` after a
    successful sampling run.

    Parameters
    ----------
    escape_modeler : EscapeVelocityModeling
        Theory + Z_v calibration provider (used to compute theory edges for plotting
        and to summarize the implied Z_v distribution at the inferred mass).

    results : dict
        Output from :func:`run_mcmc_mass_estimation`. Must contain `samples` and
        the summary keys, unless `results["status"] == "invalid"`.

    cluster_positional_data, galaxy_positional_data : see :func:`MassEstimator_main`
        Inputs used for preprocessing and diagnostic plotting.

    M200_estimate : float
        The initial (log10) mass estimate used for reference curves/annotations.

    coremin_cut, cut, bins, cosmo_params, cosmo_name, fix_R200, DirectDataPass
        Processing and modeling settings that must match those used during sampling.

    vesc_error_floor : float
        The assumed per-bin error floor (km/s) used to draw error bars on the edge.

    cluster_name : str
        Plot label and figure naming stem.

    smooth_Zv : bool, default=True
        If True, apply bin-to-bin smoothing to the Z_v summary used in plots (does not
        change the already-computed posterior; it only affects displayed curves).

    savefig_path : str or None, default=None
        Directory to write figures. If None, figures are not written.

    save_format : str, default="pdf"
        Output file extension used when saving figures.

    NON_INC : bool, default=True
        If True, enforce the same non-increasing edge assumption in the plotted edge logic.

    smooth : bool, default=False
        Passed through to the edge extraction/plotting helpers to keep visualization
        consistent with the inference configuration.

    Returns
    -------
    None
        This routine primarily has side effects (plots, printed summaries). Any values
        needed downstream should be taken from `results`.

    Notes
    -----
    - This function does **not** rerun inference; it only summarizes and visualizes.
    - If you want to change burn-in/thinning choices or compute additional posterior
      diagnostics, do that using the raw `samples` in `results` (or extend this routine).
    """



    def _mixture_summary(v_theory_vec, N_hat, qH2, escape_modeler, *, predictive=False, sigma=None, rng=None):
        """Summarize a 1D sample distribution with a simple Gaussian-mixture heuristic.

        Parameters
        ----------
        x : ndarray
            1D samples.

        Returns
        -------
        dict
            Mixture summary statistics (modes, weights, and credible intervals).
        """
        v_theory_vec = np.asarray(v_theory_vec, float).ravel()
        N_hat = np.asarray(N_hat, float).ravel()
        nb = v_theory_vec.size

        # Get Z_v quadrature nodes and weights (same as likelihood)
        Z_nodes, logW = escape_modeler.Zv_quantile_nodes(N_hat, nb, qH2)
        Z_nodes = np.asarray(Z_nodes, float)  # shape: (K, nb)

        if smooth_Zv:
            for k in range(Z_nodes.shape[0]):
                Z_nodes[k, :] = ClusterDataHandler._smooth_anchor_monotonic(
                    Z_nodes[k, :]
                )

        # Normalize weights
        logW = logW - logsumexp(logW, axis=0, keepdims=True)
        W = np.exp(logW)  # shape: (K, 1) or (K, nb)

        # Suppressed theory at each node: shape (K, nb)
        Y = v_theory_vec[None, :] / Z_nodes

        # Optionally add measurement noise for predictive band
        if predictive:
            if sigma is None:
                raise ValueError("sigma must be provided when predictive=True")
            sig = np.asarray(sigma, float).ravel()[None, :]  # broadcast to (1, nb)
            if rng is None:
                Y = Y + np.random.normal(0.0, sig, size=Y.shape)
            else:
                Y = Y + rng.normal(0.0, sig, size=Y.shape)

        # Weighted summary per bin
        # W is (K, 1) or (K, nb), Y is (K, nb)
        if W.shape[1] == 1:
            W_broadcast = W
        else:
            W_broadcast = W

        y_mean = np.sum(W_broadcast * Y, axis=0)  # weighted mean

        y_p16 = np.zeros(nb)
        y_p84 = np.zeros(nb)

        for j in range(nb):
            w_j = W[:, 0] if W.shape[1] == 1 else W[:, j]
            y_j = Y[:, j]

            # Sort by y value
            idx = np.argsort(y_j)
            y_sorted = y_j[idx]
            w_sorted = w_j[idx]

            # Cumulative weights
            cum_w = np.cumsum(w_sorted)
            cum_w /= cum_w[-1]  # normalize to [0, 1]

            # Interpolate to find percentiles
            y_p16[j] = np.interp(0.16, cum_w, y_sorted)
            y_p84[j] = np.interp(0.84, cum_w, y_sorted)

        return y_mean, y_p16, y_p84


    log10M_med = float(results['median'])
    log10M_lo  = float(results['one_sig_down'])
    log10M_hi  = float(results['one_sig_up'])
    M200_med   = 10.0**log10M_med
    M200_lo    = 10.0**log10M_lo
    M200_hi    = 10.0**log10M_hi

    cl_z = float(cluster_positional_data[2])
    rho  = rho_crit_z(cl_z, cosmo_params, cosmo_name).to(u.Msun/u.Mpc**3).value

    if M200_estimate is not None:
        R200_estimate_phys = (3.0 * (10**M200_estimate) / (200.0 * 4.0 * np.pi * rho))**(1.0/3.0)
    else:
        R200_estimate_phys = None

    # Pick the nominal R200 we'd use for preprocessing.
    if fix_R200:
        R200_unperturbed = R200_estimate_phys
    else:
        R200_unperturbed = (3.0 * M200_med / (200.0 * 4.0 * np.pi * rho))**(1.0/3.0)

    # Resolve to a valid (possibly slightly perturbed) center so that
    # preprocessing has adequate per-bin sampling. Without this, the
    # unperturbed mass can leave N_hat below the Z_v calibration floor
    # and mass_estimation_preprocessing raises.
    info = _resolve_valid_initial_center(
        cluster_positional_data, galaxy_positional_data,
        R200_unperturbed,
        coremin_cut, cut, bins,
        cosmo_params, cosmo_name, DirectDataPass,
        NON_INC, smooth,
        dlog10M_trials=(0.0, -0.05, 0.05, -0.1, 0.1),
        N_min=5, N_max=320,
    )
    if not info["ok"]:
        raise InvalidSampleError(
            f"[post] Could not resolve a valid center for post-processing: {info['reason']}"
        )

    R200_med = (
        float(info["R200_center"]) if info["R200_center"] is not None else R200_unperturbed
    )

    (
        galaxy_r, galaxy_v, N_hat,
        vesc_data_r, vesc_data_theta, vesc_data, cl_z,
        galaxy_r_with_interlopers, galaxy_v_with_interlopers
    ) = mass_estimation_preprocessing(
        cluster_positional_data, galaxy_positional_data,
        R200_med, R200_estimate_phys, coremin_cut, cut, bins,
        cosmo_params, cosmo_name, DirectDataPass, NON_INC, smooth
    )

    vesc_data_err = np.full(int(bins), float(vesc_error_floor), dtype=float)

    r_over_R = (np.asarray(vesc_data_r, float) / float(R200_med)).reshape(-1)  # r/R200

    z_use = np.repeat(cl_z, 1)
    _, v_med = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(M200_med, 1), cosmo_params, cosmo_name)
    _, v_lo  = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(M200_lo, 1),  cosmo_params, cosmo_name)
    _, v_hi  = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(M200_hi, 1),  cosmo_params, cosmo_name)
    v_med = np.asarray(v_med[0], float)
    v_lo  = np.asarray(v_lo[0],  float)
    v_hi  = np.asarray(v_hi[0],  float)

    qH2 = (
        q_z_function(z=cl_z, cosmo_params=cosmo_params, case=cosmo_name)
        * (H_z_function(z=cl_z, cosmo_params=cosmo_params, case=cosmo_name).value ** 2)
    ) / ((cosmo_params[1] * 100) ** 2)
    y_mean, y_p16, y_p84 = _mixture_summary(v_med, N_hat, qH2, escape_modeler, predictive=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(cluster_name, fontsize=40)

    # Phase-space points
    ax.scatter(np.asarray(galaxy_r, float) / float(R200_med),
               np.asarray(galaxy_v, float), c='black', s=10, alpha=0.8)
    ax.scatter(np.asarray(galaxy_r_with_interlopers, float) / float(R200_med),
               np.asarray(galaxy_v_with_interlopers, float), c='r', s=10, alpha=0.8)
    # Data edge + red band = *measurement-only* (no Zv in data term)
    vesc_data = np.asarray(vesc_data, float).reshape(-1)
    ax.plot(r_over_R,  vesc_data, c='black', lw=1.6, label='Edge')
    ax.plot(r_over_R, -vesc_data, c='black', lw=1.6)
    ax.fill_between(r_over_R,  vesc_data - vesc_data_err,  vesc_data + vesc_data_err,
                    color='r', alpha=0.30, zorder=0)
    ax.fill_between(r_over_R, -vesc_data - vesc_data_err, -vesc_data + vesc_data_err,
                    color='r', alpha=0.30, zorder=0)

    # Model: mixture mean (blue line) + Zv-only 68% band (blue fill), mirrored
    ax.plot(r_over_R,  y_mean, color='b', lw=2.0, label='Dynamical Fit')
    ax.plot(r_over_R, -y_mean, color='b', lw=2.0)
    ax.fill_between(r_over_R,  y_p16,  y_p84,  color='b', alpha=0.25, zorder=0)
    ax.fill_between(r_over_R, -y_p84, -y_p16, color='b', alpha=0.25, zorder=0)


    ax.set_xlabel(r'$r_{\perp}/r_{200}$', size=40)
    ax.set_ylabel(r'$v_{\rm los}\,[\mathrm{km\,s^{-1}}]$', size=40)
    ax.legend(loc='upper right')
    ax.set_ylim(-4500, 4500)
    #change here
    ax.set_xlim(0.19,1.01)

    if savefig_path is not None:
        outdir = Path(savefig_path)
        outdir.mkdir(parents=True, exist_ok=True)
        ext = "pdf" if save_format.lower() == "pdf" else "png"
        outfile = outdir / f"{cluster_name}.{ext}"
        fig.savefig(outfile, format=ext, dpi=300 if ext == "png" else None)
        print(f"\n[post] Saved figure to: {outfile}")

    plt.show()
    plt.close(fig)

    

def MassEstimator(
    path_to_Zv_calibration,
    cluster_positional_data,
    galaxy_positional_data,
    cluster_name,
    M200_estimate=None,
    cosmo_params=[0.3, 0.7],
    cosmo_name='FlatLambdaCDM',
    DirectDataPass=False,
    fix_R200=True,
    vesc_error_floor=30,
    nwalkers=100,
    nsteps=1000,
    n_processes=None,
    coremin_cut=0.44,
    cut=4500,
    bins=5,
    show_plots=True,
    savefig_path=None,
    save_format="pdf",
    NON_INC=True,
    smooth=False,
    log10M200_min = None,
    log10M200_max = None
):
    """
    Run the full escape-velocity (escape-edge) mass-estimation pipeline for a single cluster
    for a known M200.

    Parameters
    ----------
    path_to_Zv_calibration : str
        Path to the directory containing Z_v calibration products used by
        :class:`EscapeVelocityModeling`. This updated pipeline expects a skew‑t
        calibration in which the fitted parameters are functions of `(N_hat, bin, qH2)`.

    cluster_positional_data : tuple of (float, float, float)
        Cluster sky position and redshift, `(cl_ra_deg, cl_dec_deg, cl_z)`, where
        RA/Dec are in decimal degrees and `cl_z` is dimensionless.

    galaxy_positional_data : array-like
        Galaxy catalog. Behavior depends on `DirectDataPass`:

        - If `DirectDataPass=False` (default), provide an array of shape `(N, 3)` with
          columns `[RA_deg, DEC_deg, z]` (decimal degrees, dimensionless redshift).
        - If `DirectDataPass=True`, provide already-projected phase-space quantities in
          the format expected by :func:`mass_estimation_preprocessing` (typically an
          array with columns `(r_proj_Mpc, v_los_kms)` plus any optional bookkeeping).

    cluster_name : str
        Label used in plots and saved figure names (e.g., `"A1689"`).

    M200_estimate : float
        Initial mass estimate **in log10 space**, i.e. `log10(M200/Msun)`. This is used
        to set prior bounds if `log10M200_min/max` are not provided, and (optionally)
        to define the initial `R200` used during preprocessing when `fix_R200=False`.

    cosmo_params : sequence
        Cosmological parameters consistent with `cosmo_name`. This is used for distance
        conversions (projection) and for computing `rho_crit(z)` when mapping mass↔R200.

    cosmo_name : str
        Name of the cosmology model (e.g., `"FlatLambdaCDM"`, `"wCDM"`, etc.) consistent
        with the helper functions in this module.

    DirectDataPass : bool, default=False
        If True, skip sky-coordinate projection and assume `galaxy_positional_data` is
        already in projected phase-space units.

    fix_R200 : bool, default=False
        If True, keep `R200` fixed to the value implied by `M200_estimate` during the
        likelihood evaluation. If False, `R200` is recomputed for each proposed mass.

    vesc_error_floor : float, default=30
        Floor for per-bin uncertainties (km/s) used in the edge likelihood. This sets a
        minimum vertical error bar to mimic los velocity errors.

    nwalkers : int, default=100
        Number of walkers in the `emcee` ensemble sampler. For a 1D mass inference,
        typical values of 50–200 are common.

    nsteps : int, default=1000
        Number of steps per walker.

    n_processes : int or None, default=None
        Number of worker processes for parallel likelihood evaluations. If None, uses
        `os.cpu_count()`.

    coremin_cut : float, default=0.44
        Core exclusion factor (in units of `R200`) used in shifting-gapper interloper
        rejection. Galaxies with `r < coremin_cut * R200` are protected from aggressive
        clipping (prevents over-cleaning the core).

    cut : float, default=4500
        Absolute LOS velocity cut (km/s) applied when building phase-space and measuring
        edges.

    bins : int, default=10
        Number of radial bins used to extract the edge. **Do not change** this unless
        the Z_v calibration was generated for the same binning choice.

    show_plots : bool, default=True
        If True, create diagnostic plots (phase-space + edge + posterior summary).

    savefig_path : str or None, default=None
        Directory where plots are saved. If None, figures are not written to disk.

    save_format : str, default="pdf"
        File extension used when saving plots (e.g., `"pdf"`, `"png"`).

    NONC_INC : bool, default=True
        If True, enforce non-increasing edges

    smooth : bool, default=False
        If True, apply the optional smoothing logic in the edge extraction stage
        (and/or Z_v bin-to-bin smoothing where enabled).

    log10M200_min, log10M200_max : float or None, default=None
        Prior bounds for `log10(M200/Msun)`. If both are None, bounds are set to
        `M200_estimate ± 1 dex`.

    Returns
    -------
    results : dict
        Dictionary containing posterior samples and summaries. On success, keys include
        (at minimum):

        - `median` : float
            Posterior median of `log10(M200/Msun)`.
        - `one_sig_down`, `one_sig_up` : float
            16th and 84th percentiles of `log10(M200/Msun)`.
        - `samples` : ndarray
            Flattened posterior samples (shape `(Nsamp,)`).
        - `acceptance` : float
            Mean `emcee` acceptance fraction.

        If preprocessing fails validation, returns `{"status": "invalid", "reason": ...}`
        with NaN summaries.

    Notes
    -----
    - This pipeline version replaces the skew‑normal Z_v model with a **skew‑t** model.
    - Z_v is treated as a **nuisance** quantity and is typically **marginalized out**
      (instead of drawing a single Z_v realization per likelihood evaluation).
    - The Z_v calibration uses `qH2 = q(z) * H(z)^2` (or its fixed-cosmology equivalent)
      as the cosmology-dependent coordinate, rather than redshift alone.

    Examples
    --------
    Run a single cluster from a raw (RA, Dec, z) galaxy catalog:

    >>> model_path = "/path/to/Zv_calibration/"
    >>> cl = (197.872, 26.560, 0.183)  # (RA_deg, Dec_deg, z)
    >>> gal = np.loadtxt("cluster_members.txt")  # shape (N, 3): RA, Dec, z
    >>> cosmo_params = [0.3, 0.7]  # (Omega_m, h) for FlatLambdaCDM (example)
    >>> results = MassEstimator_main(
    ...     model_path, cl, gal, "A1689",
    ...     M200_estimate=15.0,  # log10(M200/Msun)
    ...     cosmo_params=cosmo_params, cosmo_name="FlatLambdaCDM",
    ...     bins=10, nwalkers=150, nsteps=2000, show_plots=True
    ... )
    """
    modeler = EscapeVelocityModeling(path_to_calibration=path_to_Zv_calibration)

    if n_processes is None:
        n_processes = os.cpu_count()

    # If prior bounds not explicitly provided, center them on M200_estimate ± 1 dex
    if M200_estimate is not None:
        log10M200_min = M200_estimate - 1.0
        log10M200_max = M200_estimate + 1.0

    results = run_mcmc_mass_estimation(
        modeler, cluster_positional_data, galaxy_positional_data,
        coremin_cut, cut, bins, M200_estimate, log10M200_min, log10M200_max,
        cosmo_params, cosmo_name, DirectDataPass,vesc_error_floor,
        nwalkers=nwalkers, nsteps=nsteps, n_processes=n_processes, progress=True, fix_R200=fix_R200,NON_INC=NON_INC,smooth=smooth
    )

    # Generate diagnostic plots if the run succeeded
    if (results.get("status") != "invalid") and show_plots:

        mass_estimation_post_processing(
            modeler, results, cluster_positional_data, galaxy_positional_data,
            M200_estimate, coremin_cut, cut, bins, cosmo_params, cosmo_name,
            fix_R200, DirectDataPass, vesc_error_floor, cluster_name,
            savefig_path=savefig_path, save_format=save_format,
            NON_INC=NON_INC, smooth=smooth
        )



    return results    
    

def MassEstimator_two_stage(
    path_to_Zv_calibration,
    cluster_positional_data,
    galaxy_positional_data,
    cluster_name,
    M200_estimate=None,
    cosmo_params=[0.3, 0.7],
    cosmo_name='FlatLambdaCDM',
    DirectDataPass=False,
    fix_R200=False,
    vesc_error_floor=30,
    nwalkers_pilot=128,
    nsteps_pilot=500,
    nwalkers_data=128,
    nsteps_data=1000,
    n_processes=None,
    coremin_cut=0.44,
    cut=4500,
    bins=5,
    show_plots=True,
    savefig_path=None,
    save_format="pdf",
    NON_INC=True,
    smooth=False,
    log10M200_min=13,
    log10M200_max=17,
):
    """Two-stage mass estimator with automatic R200 refinement.
    
    This is the user-facing entry point for obtaining a dynamical mass estimate
    from a galaxy phase-space diagram using the escape-velocity method. The routine
    ties together four conceptual stages:

    1. **Calibration + theory setup**
       Loads the Z_v calibration (skew‑t model) and provides the theoretical escape
       profile for a Dehnen potential matched to an NFW halo (via a mass–concentration
       relation).

    2. **Data preprocessing**
       Converts an input galaxy catalog into projected phase-space coordinates,
       iteratively recenters the cluster, applies line-of-sight velocity cuts, and
       removes interlopers using a shifting-gapper approach.

    3. **Edge extraction**
       Measures the escape-velocity edge in `bins` radial bins. In this updated pipeline,
       the edge routine records the **exact radial location** of the edge galaxy in each
       bin (rather than using bin centers), and defines the Z_v calibration coordinate
       using the **per-bin** tracer counts N_hat.

    4. **Bayesian inference**
       Runs an `emcee` ensemble sampler for the posterior of `log10(M200/Msun)`.
       The likelihood compares the measured edge to the theoretical edge mapped through
       the Z_v suppression. By default, Z_v is **marginalized out** using deterministic
       quadrature nodes/weights from the calibrated skew‑t distribution. Optional
       smoothing can be applied to Z_v across bins, and (elsewhere in the codebase)
       Student‑t likelihoods can be used to reduce sensitivity to outlier bins.

    If `show_plots=True`, the routine also produces the standard diagnostic phase-space
    plot and optional posterior summaries (and can save them to disk).


    Runs a **pilot** MCMC to identify the dominant posterior mode(s) in
    ``log10(M200/Msun)``, then uses the pilot mode as a fixed ``R200`` seed for a
    longer **data** MCMC with a tighter prior bracket. This two-stage design
    reduces sensitivity to a poorly specified initial ``M200_estimate`` and avoids
    costly exploration of prior regions far from the data-preferred mass.

    Parameters
    ----------
    path_to_Zv_calibration : str
        Path to the Z_v calibration directory. Passed to :func:`MassEstimator_main`.
    cluster_positional_data, galaxy_positional_data, cluster_name :
        As in :func:`MassEstimator_main`.
    M200_estimate : float or None
        Initial mass guess (log10 Msun). If provided, overrides the default prior
        bounds ``log10M200_min/max`` with ``M200_estimate ± 1 dex``.
    nwalkers_pilot, nsteps_pilot : int
        Walker count and step count for the pilot run.
    nwalkers_data, nsteps_data : int
        Walker count and step count for the final data run.
    All other parameters are forwarded to :func:`MassEstimator_main`.

    Returns
    -------
    results : dict
        Results dictionary from the final data-run :func:`MassEstimator_main` call.
        If the pilot run fails, returns the (invalid) pilot results dict.

    Notes
    -----
    The mode detection uses a KDE with ``find_peaks`` to identify distinct posterior
    modes. If a weak-lensing mass ``M200_estimate`` is supplied, it is used as a
    tiebreak when two modes are comparably populated.
    """
    # Override prior bounds with a more well-informed range if M200_estimate is given
    if M200_estimate is not None:
        log10M200_min = M200_estimate - 1
        log10M200_max = M200_estimate + 1
        
    #print(M200_estimate, log10M200_min, log10M200_max)
    def mode_basins(samples, prominence=0.03, grid=4096):
        x = np.asarray(samples, float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [0.1, 99.9])
        xs = np.linspace(lo, hi, grid)
        kde = gaussian_kde(x)
        dens = kde(xs)

        peaks, props = find_peaks(dens, prominence=prominence * dens.max())
        # Unimodal posterior: return a single basin covering the full domain
        if len(peaks) <= 1:
            return [dict(Lb=-np.inf, Ub=np.inf, peak=float(xs[np.argmax(dens)]),
                         frac=1.0, unimodal=True)]

        # Identify trough positions between consecutive peaks
        troughs = []
        for i in range(len(peaks) - 1):
            a, b = peaks[i], peaks[i + 1]
            troughs.append(a + np.argmin(dens[a:b + 1]))
        troughs  = np.array(troughs, int)
        trough_x = xs[troughs]

        bounds = np.concatenate(([-np.inf], trough_x, [np.inf]))
        basins = []
        for i in range(len(peaks)):
            Lb, Ub = bounds[i], bounds[i+1]
            mask = (x > Lb) & (x <= Ub)
            frac = mask.mean()
            # basin peak location from KDE peak index
            peak = float(xs[peaks[i]])
            basins.append(dict(Lb=float(Lb), Ub=float(Ub), peak=peak, frac=float(frac), unimodal=False))
        return basins

    def basin_interval(x, Lb, Ub, mass=0.985, pad_frac=0.10):
        m = (x > Lb) & (x <= Ub)
        xb = x[m]
        qL, qU = np.quantile(xb, [(1-mass)/2, 1-(1-mass)/2])
        w = (qU - qL)
        return float(qL - pad_frac*w), float(qU + pad_frac*w), float(m.mean())

    def choose_basin(cands, Mwl=None, sig_wl=0.2, ratio_thresh=1.5, min_frac=0.10):
        # Drop basins with population fraction below min_frac
        cands = [c for c in cands if c["frac"] >= min_frac]
        if len(cands) == 0:
            # Fallback: if every candidate was dropped, restore the largest-fraction basin
            cands = sorted(cands, key=lambda c: c["frac"], reverse=True)

        ranked = sorted(cands, key=lambda c: c["frac"], reverse=True)
        if len(ranked) == 1:
            return ranked[0], {"reason":"unimodal_or_filtered"}

        f1, f2 = ranked[0]["frac"], ranked[1]["frac"]
        if (Mwl is None) or (f2 <= 0) or (f1/f2 >= ratio_thresh):
            return ranked[0], {"reason":"escape_mass_dominant"}

        d = [abs(c["peak"] - Mwl)/max(sig_wl, 1e-6) for c in ranked]
        return ranked[int(np.argmin(d))], {"reason":"wl_tiebreak"}

    
    
    # ----- Stage 1: pilot run -----
    print("Running pilot MCMC...")
    results = MassEstimator(
    path_to_Zv_calibration,
    cluster_positional_data,
    galaxy_positional_data,
    cluster_name,
    M200_estimate,
    cosmo_params,
    cosmo_name,
    DirectDataPass=DirectDataPass,
    fix_R200=fix_R200,
    vesc_error_floor=vesc_error_floor,
    nwalkers=nwalkers_pilot,
    nsteps=nsteps_pilot,
    n_processes=n_processes,
    coremin_cut=coremin_cut,
    cut=cut,
    bins=bins,
    show_plots=False if fix_R200==False else True,
    savefig_path=savefig_path,
    save_format=save_format,
    NON_INC=NON_INC,
    smooth=smooth,
    log10M200_min = log10M200_min,
    log10M200_max = log10M200_max)

    #If M200 is provided and fix_R200 is false, then the pilot run is the full run
    if (M200_estimate is not None) and (fix_R200==True):
        return results

    #If M200 is provided and fix_R200 is false, then the pilot run is the full run
    elif (M200_estimate is None) and (fix_R200==True):
        raise ValueError("You cannot perform a run if you are fixing to an unknown R200")
    
    #If we don't fix R200, we then perform stage 2 inference
    if results['status']!='invalid':

        posterior_samples=results['samples']
        x = posterior_samples

        basins = mode_basins(x, prominence=0.005, grid=4096)

        # compute candidate (L,U) for each basin
        cands = []
        for b in basins:
            Lb, Ub = b["Lb"], b["Ub"]
            L, U, frac = basin_interval(x, Lb, Ub)
            cands.append({**b, "L": L, "U": U, "frac": frac})


        # tie-break only
        chosen, why = choose_basin(cands, M200_estimate, ratio_thresh=1.5)
        L, U = chosen["L"], chosen["U"]
        #print("Mode choice:", why, "interval:", L, U, "peak:", chosen["peak"], "frac:", chosen["frac"])
        print('Densest peak estimate:', chosen["peak"])
        print('Reason:', why)


        # ----- Stage 2: full data run (fixed R200 from pilot) -----
        print("Running main MCMC...")
        results = MassEstimator(
        path_to_Zv_calibration,
        cluster_positional_data,
        galaxy_positional_data,
        cluster_name,
        chosen["peak"], #M200_estimate
        cosmo_params,
        cosmo_name,
        DirectDataPass=DirectDataPass,
        fix_R200=True,
        vesc_error_floor=vesc_error_floor,
        nwalkers=nwalkers_data,
        nsteps=nsteps_data,
        n_processes=n_processes,
        coremin_cut=coremin_cut,
        cut=cut,
        bins=bins,
        show_plots=show_plots,
        savefig_path=savefig_path,
        save_format=save_format,
        NON_INC=NON_INC,
        smooth=smooth,
        log10M200_min = log10M200_min,
        log10M200_max = log10M200_max)



        return results

    else:
        # Pilot failed; return the invalid result dict so callers can handle gracefully
        return results 
