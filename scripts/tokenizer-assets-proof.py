#!/usr/bin/env python3
"""Exercise package tokenizers under Runtime's real admission capability fences."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("sdxl", "anima")]

import anima  # noqa: E402
import sdxl  # noqa: E402
from cozy_runtime.author import InvalidRequest  # noqa: E402
from cozy_runtime.author._loader import ModelAssets  # noqa: E402
from cozy_runtime.internal import sandbox  # noqa: E402


def assets(values: dict[str, bytes]) -> ModelAssets:
    return ModelAssets(values, lambda _operation: None)


def prove_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        prefix = "sdxl" if path.name.startswith("sdxl-") else "anima"
        paths = (
            ("clip_vocab.json", "clip_merges.txt")
            if prefix == "sdxl"
            else (
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                "t5_tokenizer/tokenizer.json", "t5_tokenizer/tokenizer_config.json",
            )
        )
        for relative in paths:
            member = f"{prefix}/{relative}"
            assert wheel.read(member) == (ROOT / prefix / member).read_bytes(), member
        print(f"PASS {path.name}: packaged tokenizer data matches source")


def prove_tokenizers() -> None:
    # Install after third-party imports, as Runtime does before author construction.
    sandbox.install()
    try:
        Path("tokenizer-must-not-be-written").write_text("forbidden")
    except sandbox.FenceViolation as error:
        assert error.fence == "no_filesystem_writes"
    else:
        raise AssertionError("the admission write fence is inactive")

    empty = assets({})
    first, second = (sdxl._tokenizer(empty, name) for name in ("tokenizer", "tokenizer_2"))
    expected = [49406, 320, 1125, 539, 320, 2368, 49407]
    assert first.encode("a photo of a cat") == expected
    assert second.encode("a photo of a cat") == expected
    assert first.pad_token_id == 49407 and second.pad_token_id == 0
    assert first.model_max_length == second.model_max_length == 77
    print("PASS standard SDXL token IDs, distinct padding, read-only fallback")

    root = ROOT / "sdxl" / "sdxl"
    vocab = json.loads((root / "clip_vocab.json").read_bytes())
    vocab["cat</w>"], vocab["dog</w>"] = vocab["dog</w>"], vocab["cat</w>"]
    override = {
        "tokenizer/vocab.json": json.dumps(vocab).encode(),
        "tokenizer/merges.txt": (root / "clip_merges.txt").read_bytes(),
    }
    changed = sdxl._tokenizer(assets(override), "tokenizer")
    assert changed.encode("cat")[1] == vocab["cat</w>"] != first.encode("cat")[1]
    assert sdxl._tokenizer(assets(override), "tokenizer_2").encode("cat") == second.encode("cat")
    print("PASS complete checkpoint override and independent second-tokenizer fallback")

    for missing in override:
        partial = {name: value for name, value in override.items() if name != missing}
        try:
            sdxl._tokenizer(assets(partial), "tokenizer")
        except InvalidRequest as error:
            assert error.code == "model_asset_missing" and missing in str(error)
        else:
            raise AssertionError("partial checkpoint override silently fell back")
    try:
        sdxl._tokenizer(assets({**override, "tokenizer/vocab.json": b"{"}), "tokenizer")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("corrupt checkpoint override silently fell back")
    print("PASS partial and corrupt checkpoint overrides preserve errors")

    for name in ("tokenizer", "t5_tokenizer"):
        tokenizer = anima._tokenizer(ROOT / "anima" / "anima" / name)
        assert tokenizer.encode("a photo of a cat")
        print(f"PASS Anima {name}: bundled data loads under read-only admission")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    args = parser.parse_args()
    for path in args.wheel:
        prove_wheel(path)
    prove_tokenizers()


if __name__ == "__main__":
    main()
