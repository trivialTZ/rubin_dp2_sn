import logging

import pandas as pd
import sncosmo
from astropy.table import Table
from sfdmap2 import sfdmap
from sncosmo.fitting import flatten_result
import os
from lightcurvelynx.astro_utils.passbands import PassbandGroup
import numpy as np

os.environ["SFD_DIR"] = "/astro/users/midai/sfdmap2/sfddata-master"

logger = logging.getLogger(__name__)

lc_colmap = {
    "flux": "psfFlux",
    "fluxerr": "psfFluxErr",
    "mwebv": "mwebv",
    "time": "midpointMjdTai",
}

saltres_cols =['success', 'ncall', 'chisq', 'ndof', 'z', 'z_err', 't0', 't0_err', 'x0',
               'x0_err', 'x1', 'x1_err', 'c', 'c_err', 'mwebv', 'mwebv_err', 'z_z_cov',
               'z_t0_cov', 'z_x0_cov', 'z_x1_cov', 'z_c_cov', 'z_mwebv_cov',
               't0_z_cov', 't0_t0_cov', 't0_x0_cov', 't0_x1_cov', 't0_c_cov',
               't0_mwebv_cov', 'x0_z_cov', 'x0_t0_cov', 'x0_x0_cov', 'x0_x1_cov',
               'x0_c_cov', 'x0_mwebv_cov', 'x1_z_cov', 'x1_t0_cov', 'x1_x0_cov',
               'x1_x1_cov', 'x1_c_cov', 'x1_mwebv_cov', 'c_z_cov', 'c_t0_cov',
               'c_x0_cov', 'c_x1_cov', 'c_c_cov', 'c_mwebv_cov', 'mwebv_z_cov',
               'mwebv_t0_cov', 'mwebv_x0_cov', 'mwebv_x1_cov', 'mwebv_c_cov',
               'mwebv_mwebv_cov', 'id', 'fit_error']
dtypes = [np.float64]*(len(saltres_cols)-2)+ [int] + [str]

def fit_single_lc(
    lc,
    modelsource="salt3",
    modelpars=None,
    mpbounds=None,
    modelcov=False,
    usebands="all",
    mwebv_from_coord=True,
    passbands=None,
    **kwargs,
):
    # load lsst passbands and register to sncosmo
    try:
        sncosmo.get_bandpass('lynx_lsst_g')
    except Exception as e:
        passbands = PassbandGroup.from_preset("LSST", filters=['u','g', 'r', 'i', 'z', 'y'])
        for passband in passbands:
            band = sncosmo.Bandpass(passband.transmission_table[passband.transmission_table[:, 1]>1.e-4, 0], 
                                    passband.transmission_table[passband.transmission_table[:, 1]>1.e-4, 1], 
                                    name='lynx_lsst_' + passband.filter_name)
            sncosmo.register(band, name='lynx_lsst_' + passband.filter_name)
    
    """Fit a single light curve given single row of a NestedFrame"""
    
    if not isinstance(lc, (pd.Series, dict)):
        raise ValueError(f"This function takes a NestedFrame with single row. Current data type: {type(lc)}")

    if modelpars is None:
        modelpars = ["t0", "x0", "x1", "c", "z"]
    if mpbounds is None:
        mpbounds = {}
    if "z_est_err" not in lc.keys():
        lc["z_est_err"] = 0.15
    mpbounds.update({"z": (np.max([0.0, lc["z_est"]-3.0*lc["z_est_err"]]),lc["z_est"]+3.0*lc["z_est_err"])})

    if "SFD_DIR" in os.environ:
        dustmap = sfdmap.SFDMap()
    else:
        raise RuntimeError(
            "Environment variable SFD_DIR must point to the SFD data directory. "
            "See installation instructions at: https://github.com/kbarbary/sfdmap"
        )
        
    model = sncosmo.Model(
        source=modelsource, effects=[sncosmo.F99Dust()], effect_names=["mw"], effect_frames=["obs"]
    )

    lc["diaSource_dia_object_lc.zp"] = 31.4
    lc["diaSource_dia_object_lc.zpsys"] = "ab"

    if mwebv_from_coord:
        ra = lc["ra_dia_object_lc"]
        dec = lc["dec_dia_object_lc"]
        mwebv = dustmap.ebv(ra, dec)
    else:
        mwebv = lc[lc_colmap["mwebv"]]
    # logger.info(f"fitting {lc["diaObjectId_dia_object_lc"]}, mwebv={mwebv}.")
    model.set(mwebv=mwebv)
    model.set(z=lc["z_est"])

    if usebands != "all":
        lc = lc.query(f"diaSource_dia_object_lc.band in list{usebands}").dropna()
        if len(lc) == 0:
            logger.info(f"No data in selected bands:{usebands}")
    time = lc[f"diaSource_dia_object_lc.{lc_colmap['time']}"]
    flux = lc[f"diaSource_dia_object_lc.{lc_colmap['flux']}"]
    fluxerr = lc[f"diaSource_dia_object_lc.{lc_colmap['fluxerr']}"]
    band = "lynx_lsst_" + lc["diaSource_dia_object_lc.band"]

    lc_dict = {"time":time,
               "band":band,
               "flux":flux,
               "fluxerr":fluxerr,
               "zp": [31.4]*len(time),
               "zpsys": ["ab"]*len(time)}
    try:
        result, fitted_model = sncosmo.fit_lc(
            lc_dict,
            model,
            modelpars,
            modelcov=modelcov,
            bounds=mpbounds.copy(),
            guess_z = True,
            guess_t0 = True,
            **kwargs,
        )
    
        res = flatten_result(result)
        res["id"] = int(lc["diaObjectId_dia_object_lc"])
        res["fit_error"] = 'None'
        for key,dtype in zip(saltres_cols[:-2], dtypes[:-2]):
            res[key] = dtype(res[key])
    
        return dict(res)
    except Exception as e:
        res = {}
        for key,dtype in zip(saltres_cols[0:-2], dtypes[0:-2]):
            res[key] = dtype(np.nan)
        res["id"] = int(lc["diaObjectId_dia_object_lc"])
        res["fit_error"] = repr(e)
        return res


class BandReferenceLevels(sncosmo.PropagationEffect):
    """Parameter container; additive levels are applied in bandflux, not the SED."""
    _minwave = 0.0
    _maxwave = np.inf

    def __init__(self, filters):
        self._param_names = list(filters)
        self.param_names_latex = list(filters)
        self._parameters = np.zeros(len(filters))

    def propagate(self, wave, flux, phase=None):
        return flux


class DifferenceSALTModel(sncosmo.Model):
    """SALT3 + one signed constant per observer band, expressed in nJy.

    For photometric fits with modelcov=False only. Reference levels are
    nuisance parameters, not measured template fluxes or an SED component.
    """
    def __init__(self, filters, source="salt3"):
        self.reference_filters = tuple(filters)
        super().__init__(source=source,
                         effects=[sncosmo.F99Dust(), BandReferenceLevels(filters)],
                         effect_names=["mw", "ref"], effect_frames=["obs", "obs"])

    def __copy__(self):
        copied = type(self)(self.reference_filters, source=self.source)
        copied.parameters = self.parameters.copy()
        return copied

    def bandflux(self, band, time, zp=None, zpsys=None):
        flux = np.asarray(super().bandflux(band, time, zp=zp, zpsys=zpsys))
        bands = np.broadcast_to(np.asarray(band), flux.shape)
        offset = np.zeros_like(flux)
        if zp is not None:
            zps = np.broadcast_to(np.asarray(zp), flux.shape)
            systems = np.broadcast_to(np.asarray(zpsys), flux.shape)
        for item in set(bands.flat):
            bp = sncosmo.get_bandpass(item)
            filt = bp.name.removeprefix("lynx_lsst_")
            if filt not in self.reference_filters:
                raise ValueError(f"No reference parameter for band {bp.name}")
            mask = bands == item
            # Convert nJy (AB zp=31.4) to photons, then to the requested units.
            photons = self.get("ref" + filt) * 10**(-0.4 * 31.4)
            photons *= sncosmo.get_magsystem("ab").zpbandflux(bp)
            if zp is None:
                offset[mask] = photons
            else:
                for system in set(systems[mask].flat):
                    selected = mask & (systems == system)
                    offset[selected] = (photons * 10**(0.4 * zps[selected]) /
                                        sncosmo.get_magsystem(system).zpbandflux(bp))
        result = flux + offset
        return result.item() if result.ndim == 0 else result

    def bandfluxcov(self, *args, **kwargs):
        raise ValueError("Reference-level retry currently requires modelcov=False")


legacy_fields = ["success", "chisq", "ndof", "z", "t0", "x0", "x1", "c"]
baseline_numeric = (["legacy_" + key for key in legacy_fields] +
                    ["baseline_recovered", "baseline_nband", "baseline_delta_bic"] +
                    ["baseline_" + band + suffix for band in "ugrizy"
                     for suffix in ["", "_err"]])
baseline_text = ["fit_model", "baseline_status"]
salt_output_columns = saltres_cols + baseline_numeric + baseline_text
salt_output_dtypes = ([np.float64] * (len(saltres_cols) - 2) + [int, str] +
                      [np.float64] * len(baseline_numeric) + [str] * len(baseline_text))


def normalize_salt_output(result):
    """Use identical scalar types in legacy/recovered/empty HATS partitions."""
    return {name: dtype(result[name])
            for name, dtype in zip(salt_output_columns, salt_output_dtypes)}


def passes_salt_cuts(result):
    return (result["success"] == 1 and -3.99 < result["x1"] < 3.99 and
            -0.399 < result["c"] < 0.799 and result["mwebv"] < 0.25 and
            result["ndof"] > 0 and 0 < result["chisq"] / result["ndof"] < 20)


def fit_single_lc_with_baselines(
    lc, bounds=None, phase_range=(-15, 45), modelcov=False,
    retry_band_baselines=True,
):
    """Keep passing SALT3 fits; retry rejected fits with signed band levels.

    Each ref{band} is an additive nJy constant fitted jointly with z, t0,
    x0, x1 and c. Finite surviving DiaSource rows, including negative flux,
    are used; the notebook retains its separate forced-photometry phase cuts.

    Retry candidates must pass the existing SALT/chi-square cuts, have at
    least three supported bands, positive degrees of freedom, a nonsingular
    covariance, x0 > 5*x0_err, an interior redshift and improved same-row BIC.
    Three redshift seeds are tried: z_est, half z_est and the legacy result.
    Prefer accepted solutions with more epochs, then lower reduced chi-square.

    legacy_* records the original fit summary. The exported SALT covariance
    is marginalized over the reference parameters; only individual reference
    errors, not their full joint covariance, are exported. Fitted levels do
    not establish a template-contamination mechanism. modelcov must be False
    for a retry. Set retry_band_baselines=False to return the legacy result
    with the same extended output schema. fit_single_lc remains unchanged.
    """
    bounds = {"x1": (-4, 4), "c": (-0.4, 0.8)} if bounds is None else bounds.copy()
    legacy = fit_single_lc(lc.copy(), mpbounds=bounds.copy(),
                           phase_range=phase_range, modelcov=modelcov)
    output = dict(legacy)
    output.update({name: np.nan for name in baseline_numeric})
    output.update({"legacy_" + key: legacy[key] for key in legacy_fields})
    output.update(baseline_recovered=0.0, fit_model="salt3", baseline_status="not_needed")
    for band in "ugrizy":
        output["baseline_" + band] = 0.0
    if not retry_band_baselines or passes_salt_cuts(legacy):
        return normalize_salt_output(output)
    if modelcov:
        raise ValueError("Band-baseline retry is only implemented for modelcov=False")
    output["baseline_status"] = "no_acceptable_reference_fit"

    try:
        time = np.asarray(lc["diaSource_dia_object_lc.midpointMjdTai"], dtype=float)
        flux = np.asarray(lc["diaSource_dia_object_lc.psfFlux"], dtype=float)
        error = np.asarray(lc["diaSource_dia_object_lc.psfFluxErr"], dtype=float)
        filters = np.asarray(lc["diaSource_dia_object_lc.band"], dtype=str)
        valid = np.isfinite(time + flux + error) & (error > 0)
        data = Table({"time": time[valid], "flux": flux[valid], "fluxerr": error[valid],
                      "band": np.char.add("lynx_lsst_", filters[valid]),
                      "zp": np.full(valid.sum(), 31.4), "zpsys": np.full(valid.sum(), "ab")})
        zest = float(lc["z_est"])
        zerr = float(lc.get("z_est_err", 0.15))
        zlo, zhi = max(0.0, zest - 3*zerr), zest + 3*zerr
        fitbounds = dict(bounds, z=(zlo, zhi))
        dust = float(sfdmap.SFDMap().ebv(lc["ra_dia_object_lc"], lc["dec_dia_object_lc"]))
        if not np.isfinite(dust) or dust >= 0.25:
            return normalize_salt_output(output)
        base = sncosmo.Model(source="salt3", effects=[sncosmo.F99Dust()],
                             effect_names=["mw"], effect_frames=["obs"])
        base.set(mwebv=dust)
        # Use the same full-bandpass support rule as sncosmo.fit_lc throughout z bounds.
        support = np.all(base.bandoverlap(data["band"], z=[zlo, zhi]), axis=1)
        data = data[support]
        used = sorted({str(b).removeprefix("lynx_lsst_") for b in data["band"]})
        if len(used) < 3 or len(data) <= 5 + len(used):
            output["baseline_status"] = "insufficient_reference_fit_support"
            return normalize_salt_output(output)
        # A flux range is invariant to an additive reference level; use it only for seeds.
        seed_band = max(used, key=lambda b: np.ptp(data["flux"][data["band"] == "lynx_lsst_"+b]) /
                        np.median(data["fluxerr"][data["band"] == "lynx_lsst_"+b]))
        seed_rows = data[data["band"] == "lynx_lsst_"+seed_band]
        zstarts = [zest, max(zlo + 1e-4, zest / 2), legacy.get("z", np.nan)]
        candidates = []
        for zstart in dict.fromkeys(zstarts):
            if not np.isfinite(zstart) or not zlo < zstart < zhi:
                continue
            try:
                model = DifferenceSALTModel(used)
                model.set(z=zstart, mwebv=dust, x0=1.0, x1=0.0, c=0.0, t0=0.0)
                grid = np.linspace(-15, 40, 111) * (1 + zstart)
                curve = model.bandflux("lynx_lsst_"+seed_band, grid, zp=31.4, zpsys="ab")
                t0 = float(seed_rows["time"][np.argmax(seed_rows["flux"])]) - grid[np.argmax(curve)]
                amplitude = max(np.ptp(seed_rows["flux"]) / np.max(curve), 1e-12)
                model.set(t0=t0, x0=amplitude)
                for band in used:
                    rows = data[data["band"] == "lynx_lsst_"+band]
                    prediction = model.bandflux(rows["band"], rows["time"], zp=31.4, zpsys="ab")
                    model.set(**{"ref"+band: float(np.median(rows["flux"] - prediction))})
                pars = ["z", "t0", "x0", "x1", "c"] + ["ref"+b for b in used]
                result, fitted = sncosmo.fit_lc(
                    data, model, pars, bounds=fitbounds, modelcov=False,
                    guess_z=False, guess_t0=False, guess_amplitude=False,
                    phase_range=phase_range, maxcall=3000, warn=False)
                # Refit the converged phase subset on fixed rows; drop unconstrained
                # reference parameters for bands that have no rows in this subset.
                selected = data[result.data_mask]
                active = sorted({str(b).removeprefix("lynx_lsst_") for b in selected["band"]})
                if len(active) < 3 or len(selected) <= 5 + len(active):
                    continue
                pars = ["z", "t0", "x0", "x1", "c"] + ["ref"+b for b in active]
                result, fitted = sncosmo.fit_lc(
                    selected, fitted, pars, bounds=fitbounds, modelcov=False,
                    guess_z=False, guess_t0=False, guess_amplitude=False,
                    maxcall=3000, warn=False)
                flat = flatten_result(result)
                phases = (selected["time"] - flat["t0"]) / (1 + flat["z"])
                if (not passes_salt_cuts(flat) or np.any(phases < phase_range[0]) or
                        np.any(phases > phase_range[1]) or result.covariance is None):
                    continue
                cov = result.covariance
                sigma = np.sqrt(np.diag(cov))
                corr = cov / np.outer(sigma, sigma)
                if (not np.all(np.isfinite(corr)) or np.min(np.linalg.eigvalsh(corr)) <= 1e-10 or
                        flat["x0"] <= 5 * flat["x0_err"] or
                        min(flat["z"]-zlo, zhi-flat["z"]) <= 1e-4):
                    continue
                # Compare both models on exactly these same rows, including negative fluxes.
                base.set(**{key: flat[key] for key in ["z", "t0", "x0", "x1", "c", "mwebv"]})
                zero, _ = sncosmo.fit_lc(
                    selected, base, ["z", "t0", "x0", "x1", "c"], bounds=fitbounds,
                    modelcov=False, guess_z=False, guess_t0=False, guess_amplitude=False,
                    maxcall=3000, warn=False)
                delta_bic = zero.chisq - result.chisq - len(active)*np.log(len(selected))
                if not zero.success or not np.isfinite(delta_bic) or delta_bic <= 0:
                    continue
                candidates.append((len(selected), -result.chisq/result.ndof,
                                   flat, active, delta_bic))
            except (ValueError, RuntimeError, FloatingPointError):
                continue
        if candidates:
            # Prefer solutions supported by more epochs, then smaller reduced chi-square.
            _, _, fitted, active, delta_bic = max(candidates, key=lambda item: item[:2])
            output.update({key: fitted[key] for key in saltres_cols if key not in ["id", "fit_error"]})
            output.update(fit_error="None", baseline_recovered=1.0,
                          fit_model="salt3_plus_band_reference", baseline_status="accepted",
                          baseline_nband=float(len(active)), baseline_delta_bic=float(delta_bic))
            for band in active:
                output["baseline_"+band] = fitted["ref"+band]
                output["baseline_"+band+"_err"] = fitted["ref"+band+"_err"]
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        output["baseline_status"] = "reference_fit_error: " + str(exc)
    return normalize_salt_output(output)
