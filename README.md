# rubin_dp2_sn
Filtering DP2 data to get sn candidates

## Export candidates before the final SALT cuts

The full-sample notebook saves a HATS catalog at
`outputs/saltfit_on_good_sne_after_quality_cut` before applying its SALT
parameter and reduced-chi-square cuts. To inspect candidates that did not
reach `final_sample.parquet`, run after the notebook has produced its saved
catalogs:

```bash
python candidate_audit.py
```

This writes `outputs/salt_candidate_audit.parquet`. It retains every row in
the saved pre-cut SALT catalog, including failed fits and boundary values,
and adds one Boolean column per notebook cut, `failed_cuts`, and
`in_final_sample` when the final Parquet file is present. It exports scalar
quantities only; nested light curves are not copied. If the saved
`sncandid_w_bazin` catalog exists, it also writes
`outputs/initial_candidate_audit.parquet` for the stage after the initial
host and detection filters. If the outputs are stored elsewhere, pass the
corresponding catalog and output path options.

This diagnostic does not change the selection. The initial export begins
after the host crossmatch and initial detection filters. The SALT export
begins after the first forced-light-curve phase cut. An object absent from
both may have failed an earlier stage or may not be in the underlying
catalog snapshot. The final-sample
membership flag also includes the later forced-light-curve phase cut, so
passing the listed SALT and chi-square flags does not imply inclusion.
Signed difference fluxes, including negative values, should be retained
when investigating failed SALT fits; the cut flags are descriptions of the
notebook's current selection, not classifications of SN quality.
