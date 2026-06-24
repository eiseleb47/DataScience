"""Planck CMB TT simulation data for the NPE exercise (ProblemSet 10).

This file is BOTH:

  * a **loader** -- ``load_planck_npe_data()`` reads the shipped HDF5 file and needs only
    numpy + h5py; and
  * a **regeneration script** -- ``python make_planck_npe_data.py`` recomputes the HDF5 from
    scratch with CAMB (``import camb`` is deferred into ``regenerate`` so importing this module
    never requires CAMB).

The HDF5 stores the *clean*, noise-free CAMB spectra.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py

# ----------------------------------------------------------------------------------------------
# Configuration -- mirrors notebooks/lecture_10_planck_camb_npe.ipynb
# ----------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_H5 = HERE / "planck_camb_tt.h5"
PLANCK_TT_FILE = HERE / "COM_PowerSpect_CMB-TT-full_R3.01.txt"

LMAX = 1500
N_SIMULATIONS = 2000

# Fiducial cosmology (Planck 2018). Only Omega_m and Omega_b vary; the rest are fixed.
H0 = 67.32
h = H0 / 100.0
omega_m_fid = 0.3153
ombh2_fid = 0.022383
omega_b_fid = ombh2_fid / h**2
As = np.exp(3.0448) / 1e10
ns = 0.96605
tau = 0.0543

# Approximate 1-sigma uncertainties used to define the (uniform) prior box, +/- 4 sigma wide.
omega_m_sigma = 0.0073
ombh2_sigma = 0.00015
omega_b_sigma = ombh2_sigma / h**2

PRIOR_LOW = np.array([omega_m_fid - 4 * omega_m_sigma, omega_b_fid - 4 * omega_b_sigma], dtype=np.float64)
PRIOR_HIGH = np.array([omega_m_fid + 4 * omega_m_sigma, omega_b_fid + 4 * omega_b_sigma], dtype=np.float64)
THETA_FID = np.array([omega_m_fid, omega_b_fid], dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# Loader (no CAMB required)
# ----------------------------------------------------------------------------------------------
def load_planck_npe_data(path=DEFAULT_H5):
    """Load the clean CAMB training spectra and the Planck observation from HDF5.

    Returns a ``SimpleNamespace`` with numpy arrays:
        theta            (n, 2)      sampled (Omega_m, Omega_b)
        spectra          (n, n_ell)  CLEAN D_ell^TT in K^2 (no noise)
        ell              (n_ell,)    multipoles
        planck_dl        (n_ell,)    official unbinned Planck TT D_ell in K^2
        planck_err_minus (n_ell,)    lower error bar
        planck_err_plus  (n_ell,)    upper error bar
        planck_sigma     (n_ell,)    symmetric 1-sigma error (0.5*(minus+plus))
        fid_theta        (2,)        fiducial (Omega_m, Omega_b)
        fid_spectrum     (n_ell,)    clean fiducial D_ell
    and scalar/array metadata: H0, h, As, ns, tau, lmax, prior_low, prior_high.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with `python {Path(__file__).name}` "
            "(requires CAMB), or obtain the shipped file."
        )
    with h5py.File(path, "r") as f:
        ns_out = SimpleNamespace(
            theta=f["theta"][:],
            spectra=f["spectra"][:],
            ell=f["ell"][:],
            planck_dl=f["planck_dl"][:],
            planck_err_minus=f["planck_err_minus"][:],
            planck_err_plus=f["planck_err_plus"][:],
            planck_sigma=f["planck_sigma"][:],
            fid_theta=f["fid_theta"][:],
            fid_spectrum=f["fid_spectrum"][:],
            prior_low=f.attrs["prior_low"],
            prior_high=f.attrs["prior_high"],
            H0=float(f.attrs["H0"]),
            h=float(f.attrs["h"]),
            As=float(f.attrs["As"]),
            ns=float(f.attrs["ns"]),
            tau=float(f.attrs["tau"]),
            lmax=int(f.attrs["lmax"]),
        )
    return ns_out


# ----------------------------------------------------------------------------------------------
# Regeneration (requires CAMB)
# ----------------------------------------------------------------------------------------------
def _load_official_planck_tt(ell_max=LMAX):
    """Official unbinned Planck TT spectrum, converted from microK^2 to K^2."""
    if not PLANCK_TT_FILE.exists():
        raise FileNotFoundError(PLANCK_TT_FILE)
    data = np.loadtxt(PLANCK_TT_FILE)
    use = data[:, 0] <= ell_max
    ell = data[use, 0].astype(np.float32)
    dl = (data[use, 1] / 1e12).astype(np.float32)
    err_minus = (data[use, 2] / 1e12).astype(np.float32)
    err_plus = (data[use, 3] / 1e12).astype(np.float32)
    return ell, dl, err_minus, err_plus


def camb_tt_spectrum(theta, ell_eval, lmax=LMAX):
    """Clean CAMB TT D_ell in K^2 at the requested ell values. Requires CAMB."""
    import camb  # deferred: only needed for regeneration, never for loading

    omega_m, omega_b = np.asarray(theta, dtype=np.float64)
    ombh2 = omega_b * h**2
    omch2 = (omega_m - omega_b) * h**2
    if ombh2 <= 0 or omch2 <= 0:
        raise ValueError(f"Unphysical densities: ombh2={ombh2}, omch2={omch2}")

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="K")
    dl_tt = powers["total"][: lmax + 1, 0]
    ell_full = np.arange(lmax + 1)
    return np.interp(ell_eval, ell_full, dl_tt).astype(np.float32)


def regenerate(out_path=DEFAULT_H5, n=N_SIMULATIONS, seed=7):
    """Recompute the clean CAMB simulation bank + Planck observation and write the HDF5 file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ell, planck_dl, planck_err_minus, planck_err_plus = _load_official_planck_tt()
    planck_sigma = 0.5 * (planck_err_minus + planck_err_plus)
    n_features = len(ell)

    theta = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(n, 2)).astype(np.float32)
    spectra = np.empty((n, n_features), dtype=np.float32)
    for i in range(n):
        spectra[i] = camb_tt_spectrum(theta[i], ell_eval=ell)
        if (i + 1) % max(1, n // 10) == 0:
            print(f"computed {i + 1}/{n}")

    fid_spectrum = camb_tt_spectrum(THETA_FID, ell_eval=ell)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("theta", data=theta, compression="gzip")
        f.create_dataset("spectra", data=spectra, compression="gzip")
        f.create_dataset("ell", data=ell)
        f.create_dataset("planck_dl", data=planck_dl)
        f.create_dataset("planck_err_minus", data=planck_err_minus)
        f.create_dataset("planck_err_plus", data=planck_err_plus)
        f.create_dataset("planck_sigma", data=planck_sigma.astype(np.float32))
        f.create_dataset("fid_theta", data=THETA_FID.astype(np.float32))
        f.create_dataset("fid_spectrum", data=fid_spectrum)
        f.attrs["H0"] = H0
        f.attrs["h"] = h
        f.attrs["As"] = As
        f.attrs["ns"] = ns
        f.attrs["tau"] = tau
        f.attrs["lmax"] = LMAX
        f.attrs["prior_low"] = PRIOR_LOW
        f.attrs["prior_high"] = PRIOR_HIGH

    print(f"Wrote {n} clean simulations ({n_features} multipoles) to {out_path}")
    return out_path


if __name__ == "__main__":
    regenerate()
