"""Explicit coverage of the upstream family registry, including inapplicable modes."""
import json
from pathlib import Path
import unittest

from heads import FAMILIES


class FamilyCoverageTests(unittest.TestCase):
    def test_every_family_has_an_honest_contract(self):
        manifest = json.loads(Path(__file__).with_name("families.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "colibri.strata.family-integration/v1")
        entries = manifest["families"]
        self.assertEqual(set(entries), {f.id for f in FAMILIES})
        for family in FAMILIES:
            row = entries[family.id]
            self.assertEqual(row["modality"], family.modality)
            if family.modality == "text":
                self.assertEqual(row["parallel_heads"], "pending")
                self.assertIn(row["verifier"], ("pending", "upstream", "upstream-guarded", "upstream-opt-in"))
            else:
                self.assertEqual(row["parallel_heads"], "not-applicable")
                self.assertEqual(row["verifier"], "not-applicable")


if __name__ == "__main__":
    unittest.main()
