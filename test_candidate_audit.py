"""Tests for the pre-cut SALT candidate audit."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from candidate_audit import ID, add_cut_flags, read_scalar_catalog


class CandidateAuditTests(unittest.TestCase):
    def test_retains_failures_and_matches_notebook_boundaries(self):
        fits = pd.DataFrame(
            {
                ID: [11, 12, 13, 14, 15],
                "success": [1, 1, 1, 1, np.nan],
                "x1": [0.0, -4.0, 0.0, 0.0, np.nan],
                "c": [0.0, 0.0, 0.8, 0.0, np.nan],
                "mwebv": [0.1, 0.1, 0.1, 0.25, np.nan],
                "chisq": [30.0, 10.0, 10.0, 10.0, np.nan],
                "ndof": [10.0, 10.0, 10.0, 0.0, np.nan],
            }
        )
        audit = add_cut_flags(fits, final_ids={11})

        self.assertEqual(len(audit), len(fits))
        self.assertEqual(list(audit["passes_salt_parameters"]),
                         [True, False, False, False, False])
        self.assertEqual(list(audit["in_final_sample"]),
                         [True, False, False, False, False])
        self.assertEqual(audit.loc[0, "reduced_chisq"], 3.0)
        self.assertTrue(np.isnan(audit.loc[3, "reduced_chisq"]))
        self.assertEqual(audit.loc[1, "failed_cuts"], "x1")
        self.assertEqual(audit.loc[2, "failed_cuts"], "color")
        self.assertEqual(audit.loc[3, "failed_cuts"], "mwebv,reduced_chisq")
        self.assertEqual(
            audit.loc[4, "failed_cuts"],
            "fit_failed,x1,color,mwebv,reduced_chisq",
        )

    def test_preserves_multiple_host_matches_for_same_diaobject(self):
        fits = pd.DataFrame(
            {
                ID: [42, 42],
                "_dist_arcsec": [0.3, 1.2],
                "success": [1, 1],
                "x1": [0.0, 0.0],
                "c": [0.0, 0.0],
                "mwebv": [0.1, 0.1],
                "chisq": [15.0, 15.0],
                "ndof": [10.0, 10.0],
            }
        )
        audit = add_cut_flags(fits)
        self.assertEqual(len(audit), 2)
        self.assertEqual(list(audit["_dist_arcsec"]), [0.3, 1.2])
        self.assertNotIn("in_final_sample", audit.columns)

    def test_projects_scalar_columns_before_materializing_catalog(self):
        class FakeCatalog:
            def __init__(self):
                self.columns = [ID, "x1", "diaSource_dia_object_lc"]
                self.projected = None

            def __getitem__(self, columns):
                self.projected = columns
                return self

            def compute(self):
                return pd.DataFrame({ID: [42], "x1": [-4.0]})

        fake = FakeCatalog()
        with patch.dict("sys.modules", {"lsdb": SimpleNamespace(open_catalog=lambda _: fake)}):
            result = read_scalar_catalog(Path("saved_hats"))
        self.assertEqual(fake.projected, [ID, "x1"])
        self.assertEqual(list(result[ID]), [42])


if __name__ == "__main__":
    unittest.main()
