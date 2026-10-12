"""Dependency-free checks for the committed direct-ref2va public contract."""
import json
from pathlib import Path


def test_ref2va_metadata_exposes_named_reference_options() -> None:
    metadata = json.loads(
        (Path(__file__).parents[1] / "metadata" / "package-interface.json").read_text()
    )
    jobs = {job["name"]: job for job in metadata["jobs"]}
    expected = {"references", "style", "prepare_references", "caption_references"}
    for name in ("ref2va", "ref2va_turbo"):
        fields = {field["name"]: field for field in jobs[name]["request"]["fields"]}
        assert expected <= fields.keys()
        assert fields["references"]["type"]["list"]["fields"]
        assert fields["prepare_references"]["type"] == "bool"
        assert fields["caption_references"]["type"] == "bool"
