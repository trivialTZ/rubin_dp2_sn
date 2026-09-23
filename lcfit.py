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
    """Expose baseline parameters to sncosmo; leave the source SED unchanged."""
    _minwave, _maxwave = 0.0, np.inf

    def __init__(self, filters):
        self._param_names = list(filters)
        self.param_names_latex = list(filters)
        self._parameters = np.zeros(len(filters))

    def propagate(self, wave, flux, phase=None):
        return flux


class DifferenceSALTModel(sncosmo.Model):
    """SALT3 plus signed per-band nJy levels; AB photometry, modelcov=False only."""
    def __init__(self, filters, source="salt3"):
        self.reference_filters = tuple(filters)
        super().__init__(source=source,
                         effects=[sncosmo.F99Dust(), BandReferenceLevels(filters)],
                         effect_names=["mw", "ref"], effect_frames=["obs", "obs"])

    def __copy__(self):
        # fit_lc copies its model; preserve this subclass and its baselines.
        model = type(self)(self.reference_filters, source=self.source)
        model.parameters = self.parameters.copy()
        return model

    def bandflux(self, band, time, zp=None, zpsys=None):
        if zp is None or not np.all(np.asarray(zpsys) == "ab"):
            raise ValueError("Band baselines require an explicit AB zeropoint")
        flux = np.asarray(super().bandflux(band, time, zp=zp, zpsys=zpsys))
        bands = np.broadcast_to(np.asarray(band), flux.shape)
        baseline = np.zeros_like(flux)
        for item in set(bands.flat):
            name = sncosmo.get_bandpass(item).name.removeprefix("lynx_lsst_")
            baseline[bands == item] = self.get("ref" + name)
        # The baseline is in nJy (AB zp=31.4); respect sncosmo's requested zp.
        flux = flux + baseline * 10**(0.4 * (np.asarray(zp) - 31.4))
        return flux.item() if flux.ndim == 0 else flux

    def bandfluxcov(self, *args, **kwargs):
        raise ValueError("Band-baseline fitting requires modelcov=False")


legacy_fields = ["success", "chisq", "ndof", "z", "t0", "x0", "x1", "c"]
baseline_numeric = (["legacy_" + key for key in legacy_fields] +
                    ["baseline_recovered", "baseline_nband", "baseline_delta_bic"] +
                    ["baseline_" + band + suffix for band in "ugrizy" for suffix in ["", "_err"]])
salt_output_columns = saltres_cols + baseline_numeric + ["fit_model", "baseline_status"]
salt_output_dtypes = dtypes + [np.float64] * len(baseline_numeric) + [str, str]


def normalize_salt_output(result):
    """Keep identical column types in original, recovered and empty partitions."""
    return {name: dtype(result[name]) for name, dtype in zip(salt_output_columns, salt_output_dtypes)}


def passes_salt_cuts(result):
    return (result["success"] == 1 and -3.99 < result["x1"] < 3.99 and
            -0.399 < result["c"] < 0.799 and result["mwebv"] < 0.25 and
            result["ndof"] > 0 and 0 < result["chisq"] / result["ndof"] < 20)


def fit_single_lc_with_baselines(lc, bounds=None, phase_range=(-15, 45), modelcov=False,
                               retry_band_baselines=True):
    """Keep passing SALT fits; retry failures with one constant per observed band.

    Baselines are free nuisance parameters, not measured template fluxes.
    SALT errors include baseline uncertainty; only individual baseline errors
    are exported. The notebook's final forced-photometry phase cuts still apply.
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
        raise ValueError("Band-baseline fitting requires modelcov=False")
    output["baseline_status"] = "no_acceptable_reference_fit"

    try:
        # Prepare finite signed difference fluxes; retain negative measurements.
        prefix = "diaSource_dia_object_lc."
        data = Table({
            "time": np.asarray(lc[prefix + "midpointMjdTai"], dtype=float),
            "flux": np.asarray(lc[prefix + "psfFlux"], dtype=float),
            "fluxerr": np.asarray(lc[prefix + "psfFluxErr"], dtype=float),
            "band": np.asarray(lc[prefix + "band"], dtype=str),
        })
        valid = np.isfinite(data["time"] + data["flux"] + data["fluxerr"]) & (data["fluxerr"] > 0)
        data = data[valid]
        data["band"] = np.char.add("lynx_lsst_", np.asarray(data["band"], dtype=str))
        data["zp"] = 31.4
        data["zpsys"] = "ab"
        zest, zerr = float(lc["z_est"]), float(lc.get("z_est_err", 0.15))
        zlo, zhi = max(0.0, zest - 3*zerr), zest + 3*zerr
        bounds["z"] = (zlo, zhi)
        dust = float(sfdmap.SFDMap().ebv(lc["ra_dia_object_lc"], lc["dec_dia_object_lc"]))
        if not np.isfinite(dust) or dust >= 0.25:
            return normalize_salt_output(output)
        base = sncosmo.Model(source="salt3", effects=[sncosmo.F99Dust()],
                             effect_names=["mw"], effect_frames=["obs"])
        base.set(mwebv=dust)
        data = data[np.all(base.bandoverlap(data["band"], z=[zlo, zhi]), axis=1)]
        bands = sorted({str(b).removeprefix("lynx_lsst_") for b in data["band"]})
        if len(bands) < 3 or len(data) <= 5 + len(bands):
            output["baseline_status"] = "insufficient_reference_fit_support"
            return normalize_salt_output(output)

        # Seed from the band with the clearest variation; a constant cannot change its range.
        rows_by_band = {b: data[data["band"] == "lynx_lsst_" + b] for b in bands}
        seed_rows = max(rows_by_band.values(),
                        key=lambda rows: np.ptp(rows["flux"]) / np.median(rows["fluxerr"]))
        salt_parameters = ["z", "t0", "x0", "x1", "c"]
        options = dict(bounds=bounds, modelcov=False, guess_z=False, guess_t0=False,
                       guess_amplitude=False, maxcall=3000, warn=False)
        candidates = []
        # Try the peak-flux redshift estimate, half that value, and the original fitted redshift.
        for zstart in dict.fromkeys([zest, max(zlo + 1e-4, zest / 2), legacy.get("z", np.nan)]):
            if not np.isfinite(zstart) or not zlo < zstart < zhi:
                continue
            try:
                model = DifferenceSALTModel(bands)
                model.set(z=zstart, mwebv=dust, x0=1.0, x1=0.0, c=0.0, t0=0.0)
                grid = np.linspace(-15, 40, 111) * (1 + zstart)
                curve = model.bandflux(seed_rows["band"][0], grid, zp=31.4, zpsys="ab")
                t0 = float(seed_rows["time"][np.argmax(seed_rows["flux"])]) - grid[np.argmax(curve)]
                x0 = max(np.ptp(seed_rows["flux"]) / np.max(curve), 1e-12)
                model.set(t0=t0, x0=x0)
                for band, rows in rows_by_band.items():
                    prediction = model.bandflux(rows["band"], rows["time"], zp=31.4, zpsys="ab")
                    model.set(**{"ref" + band: float(np.median(rows["flux"] - prediction))})

                # Fit SALT and baselines together, then refit the final phase subset on fixed rows.
                parameters = salt_parameters + ["ref" + b for b in bands]
                result, model = sncosmo.fit_lc(
                    data, model, parameters, phase_range=phase_range, **options)
                selected = data[result.data_mask]
                active = sorted({str(b).removeprefix("lynx_lsst_") for b in selected["band"]})
                if len(active) < 3 or len(selected) <= 5 + len(active):
                    continue
                parameters = salt_parameters + ["ref" + b for b in active]
                result, model = sncosmo.fit_lc(selected, model, parameters, **options)
                flat = flatten_result(result)
                phases = (selected["time"] - flat["t0"]) / (1 + flat["z"])
                if (not passes_salt_cuts(flat) or result.covariance is None or
                        np.any(phases < phase_range[0]) or np.any(phases > phase_range[1])):
                    continue
                # Reject degenerate fits, amplitudes below five sigma, and redshift boundaries.
                sigma = np.sqrt(np.diag(result.covariance))
                correlation = result.covariance / np.outer(sigma, sigma)
                if (not np.all(np.isfinite(correlation)) or np.min(np.linalg.eigvalsh(correlation)) <= 1e-10 or
                        flat["x0"] <= 5 * flat["x0_err"] or min(flat["z"] - zlo, zhi - flat["z"]) <= 1e-4):
                    continue

                # Accept extra parameters only if BIC improves on exactly the same observations.
                base.set(**{key: flat[key] for key in salt_parameters + ["mwebv"]})
                zero, _ = sncosmo.fit_lc(selected, base, salt_parameters, **options)
                delta_bic = zero.chisq - result.chisq - len(active)*np.log(len(selected))
                if zero.success and np.isfinite(delta_bic) and delta_bic > 0:
                    candidates.append((len(selected), -result.chisq/result.ndof, flat, active, delta_bic))
            except (ValueError, RuntimeError, FloatingPointError):
                continue

        if candidates:
            # Prefer more epochs, then smaller reduced chi-square; retain the original fit summary.
            _, _, fit, active, delta_bic = max(candidates, key=lambda item: item[:2])
            output.update({key: fit[key] for key in saltres_cols[:-2]})
            output.update(fit_error="None", baseline_recovered=1.0, fit_model="salt3_plus_band_reference",
                          baseline_status="accepted", baseline_nband=float(len(active)),
                          baseline_delta_bic=delta_bic)
            for band in active:
                output["baseline_" + band] = fit["ref" + band]
                output["baseline_" + band + "_err"] = fit["ref" + band + "_err"]
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        output["baseline_status"] = "reference_fit_error: " + str(exc)
    return normalize_salt_output(output)


def lsst_model_flux(model, band, time, baseline=0.0):
    """Plot SALT flux at AB zp=31.4 with the fitting passband and nJy baseline."""
    name = "lynx_lsst_" + band
    try:
        sncosmo.get_bandpass(name)
    except Exception:
        for pb in PassbandGroup.from_preset("LSST", filters=list("ugrizy")):
            table = pb.transmission_table
            keep = table[:, 1] > 1.e-4
            sncosmo.register(sncosmo.Bandpass(table[keep, 0], table[keep, 1],
                             name="lynx_lsst_" + pb.filter_name), force=True)
    return model.bandflux(name, time, zp=31.4, zpsys="ab") + baseline
