import numpy as np
import pandas as pd

def nightly_coadd(mjd, band, flux, fluxerr):
    """Inverse-variance weighted nightly coadd per band. Site: Rubin (Chile), mjd in UTC.

    Returns DataFrame with columns: mjd, band, flux, fluxerr, nobs
    """
    df = pd.DataFrame({
        "mjd": np.asarray(mjd, dtype=float),
        "band": np.asarray(band),
        "flux": np.asarray(flux, dtype=float),
        "fluxerr": np.asarray(fluxerr, dtype=float),
    })
    df["night"] = np.floor(df["mjd"] - 0.5).astype(int)  # night break at 12:00 UTC = Chilean morning
    df["w"] = 1.0 / df["fluxerr"] ** 2

    g = df.groupby(["night", "band"])
    out = pd.DataFrame({
        "mjd": g.apply(lambda x: np.average(x["mjd"], weights=x["w"]), include_groups=False),
        "band": g["band"].first(),
        "flux": g.apply(lambda x: np.average(x["flux"], weights=x["w"]), include_groups=False),
        "fluxerr": 1.0 / np.sqrt(g["w"].sum()),
        "nobs": g.size(),
    }).reset_index(drop=True)
    return out