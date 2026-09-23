"""Export compact diagnostics for every candidate entering the SALT fit.

Run after ``plot_candid_full.ipynb`` has written
``outputs/saltfit_on_good_sne_after_quality_cut``. The final sample is not
changed. Nested light curves are deliberately omitted from the export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ID = "diaObjectId_dia_object_lc"
SCALAR_COLUMNS = (
    ID,
    "ra_dia_object_lc",
    "dec_dia_object_lc",
    "objectId_object_lc",
    "refExtendedness_object_lc",
    "refSizeExtendedness_object_lc",
    "_dist_arcsec",
    "max_reliability",
    "max_snr",
    "ndet",
    "nbands",
    "n_g",
    "n_r",
    "n_i",
    "n_z",
    "dt",
    "median_flux_diff_ratio",
    "median_flux_diff_ratio_abs",
    "bazin_fit_reduced_chi2_g",
    "bazin_fit_reduced_chi2_r",
    "bazin_fit_reduced_chi2_i",
    "bazin_fit_reduced_chi2_z",
    "z_est",
    "success",
    "fit_error",
    "ncall",
    "chisq",
    "ndof",
    "z",
    "t0",
    "x0",
    "x1",
    "c",
    "mwebv",
)


def add_cut_flags(fits: pd.DataFrame, final_ids=None) -> pd.DataFrame:
    """Annotate saved SALT rows without discarding failed or boundary fits.

    ``passes_salt_parameters`` reproduces the notebook's first SALT cut.
    ``passes_reduced_chisq`` reproduces its later 0 < chi2/ndof < 20 cut.
    The notebook also applies a later forced-light-curve phase cut, which is
    represented only by ``in_final_sample`` when its output is supplied.
    """
    required = {ID, "success", "x1", "c", "mwebv", "chisq", "ndof"}
    missing = required - set(fits.columns)
    if missing:
        raise ValueError(f"Missing SALT columns: {sorted(missing)}")

    out = fits.copy()
    success = pd.to_numeric(out["success"], errors="coerce")
    x1 = pd.to_numeric(out["x1"], errors="coerce")
    color = pd.to_numeric(out["c"], errors="coerce")
    mwebv = pd.to_numeric(out["mwebv"], errors="coerce")
    chisq = pd.to_numeric(out["chisq"], errors="coerce")
    ndof = pd.to_numeric(out["ndof"], errors="coerce")

    out["passes_fit_success"] = success.eq(1)
    out["passes_x1"] = x1.gt(-3.99) & x1.lt(3.99)
    out["passes_color"] = color.gt(-0.399) & color.lt(0.799)
    out["passes_mwebv"] = mwebv.lt(0.25)
    out["reduced_chisq"] = chisq.div(ndof.where(ndof.gt(0)))
    out["passes_reduced_chisq"] = (
        out["reduced_chisq"].gt(0) & out["reduced_chisq"].lt(20)
    )
    parameter_flags = (
        "passes_fit_success",
        "passes_x1",
        "passes_color",
        "passes_mwebv",
    )
    out["passes_salt_parameters"] = out[list(parameter_flags)].all(axis=1)
    out["passes_salt_and_chisq"] = (
        out["passes_salt_parameters"] & out["passes_reduced_chisq"]
    )
    reason_labels = (
        ("passes_fit_success", "fit_failed"),
        ("passes_x1", "x1"),
        ("passes_color", "color"),
        ("passes_mwebv", "mwebv"),
        ("passes_reduced_chisq", "reduced_chisq"),
    )
    out["failed_cuts"] = [
        ",".join(label for column, label in reason_labels if not row[column])
        for _, row in out.iterrows()
    ]
    if final_ids is not None:
        out["in_final_sample"] = out[ID].isin(final_ids)
    return out


def read_scalar_catalog(path: Path, requested=SCALAR_COLUMNS) -> pd.DataFrame:
    """Project scalar HATS columns before computing the saved catalog."""
    import lsdb

    catalog = lsdb.open_catalog(path)
    available = set(catalog.columns)
    columns = [name for name in requested if name in available]
    if ID not in columns:
        raise ValueError(f"{path} does not contain {ID}")
    return catalog[columns].compute().reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-catalog",
        type=Path,
        default=Path("outputs/sncandid_w_bazin"),
        help="Optional saved catalog after the initial host/detection filters",
    )
    parser.add_argument(
        "--initial-output",
        type=Path,
        default=Path("outputs/initial_candidate_audit.parquet"),
        help="Scalar export of the initial candidate catalog",
    )
    parser.add_argument(
        "--salt-catalog",
        type=Path,
        default=Path("outputs/saltfit_on_good_sne_after_quality_cut"),
        help="Saved pre-cut SALT HATS catalog",
    )
    parser.add_argument(
        "--final-sample",
        type=Path,
        default=Path("outputs/final_sample.parquet"),
        help="Optional final Parquet file for final-sample membership",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/salt_candidate_audit.parquet"),
        help="Compact scalar output Parquet file",
    )
    args = parser.parse_args(argv)

    fits = read_scalar_catalog(args.salt_catalog)
    final_ids = None
    if args.final_sample.exists():
        final_ids = set(pd.read_parquet(args.final_sample, columns=[ID])[ID])
    audit = add_cut_flags(fits, final_ids=final_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.output, index=False)
    print(f"Wrote {len(audit)} pre-cut SALT rows to {args.output}")
    print(audit["failed_cuts"].value_counts(dropna=False).to_string())
    if args.initial_catalog.exists():
        initial = read_scalar_catalog(args.initial_catalog)
        args.initial_output.parent.mkdir(parents=True, exist_ok=True)
        initial.to_parquet(args.initial_output, index=False)
        print(f"Wrote {len(initial)} initial candidate rows to {args.initial_output}")


if __name__ == "__main__":
    main()
