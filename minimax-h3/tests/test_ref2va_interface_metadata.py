"""Dependency-free checks for the committed direct-ref2va public contract."""
import json
import unittest
from pathlib import Path


class Ref2vaInterfaceMetadataTest(unittest.TestCase):
    def test_named_reference_options(self) -> None:
        metadata = json.loads(
            (Path(__file__).parents[1] / "metadata" / "package-interface.json").read_text()
        )
        jobs = {job["name"]: job for job in metadata["jobs"]}
        expected = {"references", "style", "prepare_references", "caption_references"}
        for name in ("ref2va", "ref2va_turbo"):
            fields = {field["name"]: field for field in jobs[name]["request"]["fields"]}
            self.assertTrue(expected <= fields.keys())
            self.assertTrue(fields["references"]["type"]["list"]["fields"])
            self.assertEqual(fields["prepare_references"]["type"], "bool")
            self.assertEqual(fields["caption_references"]["type"], "bool")


if __name__ == "__main__":
    unittest.main()
