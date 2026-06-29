import logging

import pandas as pd
import sncosmo
from astropy.table import Table
from sfdmap2 import sfdmap
from sncosmo.fitting import flatten_result
import os
from lightcurvelynx.astro_utils.passbands import PassbandGroup

logger = logging.getLogger(__name__)

lc_colmap = {
    "flux": "psfFlux",
    "fluxerr": "psfFluxErr",
    "mwebv": "mwebv",
    "time": "midpointMjdTai",
}

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
            band = sncosmo.Bandpass(passband.transmission_table[:, 0], passband.transmission_table[:, 1], name='lynx_lsst_' + passband.filter_name)
            sncosmo.register(band, name='lynx_lsst_' + passband.filter_name)

    """Fit a single light curve given single row of a NestedFrame"""

    if not isinstance(lc, pd.Series):
        raise ValueError("This function takes a NestedFrame with single row")

    if modelpars is None:
        modelpars = ["t0", "x0", "x1", "c", "z"]
    if mpbounds is None:
        mpbounds = {}
    mpbounds.update({"z": (lc["z_est"]-0.2,lc["z_est"]+0.2)})

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

    lc["diaSource_dia_object_lc"]["zp"] = 31.4
    lc["diaSource_dia_object_lc"]["zpsys"] = "ab"

    if mwebv_from_coord:
        ra = lc["ra_dia_object_lc"]
        dec = lc["dec_dia_object_lc"]
        mwebv = dustmap.ebv(ra, dec)
    else:
        mwebv = lc[lc_colmap["mwebv"]]
    logger.info(f"fitting {lc["diaObjectId_dia_object_lc"]}, mwebv={mwebv}.")
    model.set(mwebv=mwebv)

    if usebands != "all":
        lc = lc.query(f"lightcurve.band in list{usebands}").dropna()
        if len(lc) == 0:
            logger.info(f"No data in selected bands:{usebands}")
    lc["diaSource_dia_object_lc"]["time"] = lc["diaSource_dia_object_lc"][f"{lc_colmap['time']}"]
    lc["diaSource_dia_object_lc"]["flux"] = lc["diaSource_dia_object_lc"][f"{lc_colmap['flux']}"]
    lc["diaSource_dia_object_lc"]["fluxerr"] = lc["diaSource_dia_object_lc"][f"{lc_colmap['fluxerr']}"]
    lc["diaSource_dia_object_lc"]["band"] = "lynx_lsst_" + lc["diaSource_dia_object_lc"]["band"]

    try:
        result, fitted_model = sncosmo.fit_lc(
            Table.from_pandas(lc["diaSource_dia_object_lc"]),
            model,
            modelpars,
            modelcov=modelcov,
            bounds=mpbounds.copy(),
            guess_z = True,
            guess_t0 = True,
            **kwargs,
        )

        res = flatten_result(result)
        res["id"] = lc["diaObjectId_dia_object_lc"]
        res["fit_error"] = None

        return pd.Series(res)
    except Exception as e:
        return pd.Series({"id": lc["diaObjectId_dia_object_lc"], "fit_error": repr(e)})
