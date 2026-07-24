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
