"""Private experiment: exact frozen source except its no-op lifecycle warm body."""

import ast
import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("package,class_name", [("sdxl", "SdxlModel"), ("anima", "AnimaModel")])
def test_only_warm_body_and_project_identity_change(package: str, class_name: str) -> None:
    directory = ROOT / "experiments" / f"{package}-no-discarded-warm"
    provenance = json.loads((directory / "NO-WARM-PROVENANCE.json").read_text())
    for name, digest in provenance["baseline_sha256"].items():
        frozen = subprocess.check_output(
            ["git", "show", f"{provenance['source_head']}:{provenance['source_directory']}/{name}"],
            cwd=ROOT,
        )
        assert hashlib.sha256(frozen).hexdigest() == digest
        candidate = (directory / name).read_bytes()
        if name == f"{package}/__init__.py":
            tree = ast.parse(frozen)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            warm = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "warm")
            lines = frozen.decode().splitlines(keepends=True)
            lines[warm.body[0].lineno - 1 : warm.end_lineno] = [
                "        ctx.raise_if_cancelled()\n",
                "        return\n",
            ]
            assert candidate == "".join(lines).encode()
        elif name in ("pyproject.toml", "uv.lock"):
            old = f"cozy-experiment-{package}-" + (
                "stable-vae" if package == "sdxl" else "stage-scopes"
            )
            new = f"cozy-experiment-{package}-no-discarded-warm"
            assert candidate == frozen.replace(old.encode(), new.encode())
        else:
            assert candidate == frozen, name


@pytest.mark.parametrize("package,class_name", [("sdxl", "SdxlModel"), ("anima", "AnimaModel")])
def test_actual_model_warm_only_checks_cancellation_without_scope(
    package: str, class_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cozy_runtime.author._model import component_use, undeclared_methods

    monkeypatch.syspath_prepend(str(ROOT / "experiments" / f"{package}-no-discarded-warm"))
    module = importlib.import_module(package)
    cls = getattr(module, class_name)

    def no_scope(*args: object, **kwargs: object) -> None:
        pytest.fail("lifecycle warm must not admit any model component")

    monkeypatch.setattr(cls, "_cozy_scope", no_scope)
    model = cls.__new__(cls)
    calls = []
    assert model.warm(SimpleNamespace(raise_if_cancelled=lambda: calls.append("checked"))) is None
    assert calls == ["checked"]
    assert "warm" not in component_use(cls) and "warm" not in undeclared_methods(cls)
    failure = ValueError("canceled")

    def cancelled() -> None:
        raise failure

    with pytest.raises(ValueError) as error:
        model.warm(SimpleNamespace(raise_if_cancelled=cancelled))
    assert error.value is failure

    from cozy_runtime.author._context import Device
    from cozy_runtime.internal.warm import warm_generation
    import torch

    before = torch.get_rng_state().clone()
    assert warm_generation(model, device=Device(), cancel=lambda: False) >= 0
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
