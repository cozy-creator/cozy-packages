"""Bounded source-key controls for SDXL/H3; no model execution or CLI qualification."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = '''import json,sys
sys.path[:] = [sys.argv[1], *json.loads(sys.argv[2])]
from cozy_runtime.author._calls import _export
from cozy_runtime.internal.memo_implementation import describe
from sdxl.normalization import normalize_component,assemble_normalized
from h3_tables.operations import assemble_full
from h3_tables.adaln_operations import select_adaln_weights,compute_adaln_tables,apply_adaln,retable_adaln
from h3_tables.turbo import turbo_lora
from h3_tables.job import lanes,retable
functions=(normalize_component,assemble_normalized,assemble_full,select_adaln_weights,
           compute_adaln_tables,apply_adaln,retable_adaln,turbo_lora,lanes,retable)
print(json.dumps({fn.__name__:describe(_export(fn).implementation) for fn in functions}))
'''


def identities(root: Path) -> dict[str, str]:
    rows = json.loads(subprocess.check_output(
        [sys.executable, '-I', '-B', '-c', PROBE, str(root), json.dumps(sys.path)], text=True))
    assert len(rows) == 10 and all(row.get('operation_identity') for row in rows.values()), rows
    return {name: row['operation_identity'] for name, row in rows.items()}


def edit_caller(path: Path, name: str) -> None:
    text = path.read_text()
    definition = next(node for node in ast.parse(text).body
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    lines = text.splitlines()
    lines.insert(definition.body[0].lineno - 1, '    print("caller-only edit")')
    path.write_text('\n'.join(lines) + '\n')


def changed_native_version(root: Path, name: str) -> None:
    """Shadow only installed distribution metadata; never copy or scan native binaries."""
    distribution = importlib.metadata.distribution(name)
    destination = root / f"{name}-memo_probe.dist-info"
    destination.mkdir()
    for member in ("METADATA", "WHEEL", "direct_url.json"):
        content = distribution.read_text(member)
        if content is not None:
            if member == "METADATA":
                content = content.replace(
                    f"Version: {distribution.version}\n", "Version: 999.0+memo.probe\n", 1
                )
            (destination / member).write_text(content)


def qualify(root: Path) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    for source, target in [('sdxl/sdxl', 'sdxl'), ('minimax-h3-tools/src/h3_tables', 'h3_tables')]:
        shutil.copytree(repository / source, root / target, ignore=shutil.ignore_patterns('__pycache__'))
    before = identities(root)
    edit_caller(root / 'sdxl/normalization.py', 'normalize')
    edit_caller(root / 'h3_tables/operations.py', 'precompute_adaln')
    assert identities(root) == before
    helper = root / 'sdxl/normalization.py'
    text = helper.read_text()
    assert '.T.copy().tobytes()' in text
    helper.write_text(text.replace('.T.copy().tobytes()', '.copy().tobytes()'))
    after_helper = identities(root)
    assert after_helper['normalize_component'] != before['normalize_component']
    plan = root / 'sdxl/normalization.json'
    data = json.loads(plan.read_text())
    data['configs'][next(iter(data['configs']))]['memo_dependency_probe'] = True
    plan.write_text(json.dumps(data))
    after_sdxl = identities(root)
    assert after_sdxl['normalize_component'] != after_helper['normalize_component']
    assert after_sdxl['assemble_normalized'] != after_helper['assemble_normalized']
    helper = root / 'h3_tables/kernel.py'
    text = helper.read_text()
    assert 'math.log(' in text
    helper.write_text(text.replace('math.log(', 'math.log(2.0 * ', 1))
    after_kernel = identities(root)
    for name in ('compute_adaln_tables', 'turbo_lora', 'lanes', 'retable'):
        assert after_kernel[name] != after_sdxl[name]
    plan = root / 'h3_tables/assets/timestep-plan.fl2va.json'
    data = json.loads(plan.read_text())
    data['video_shift'] = '0x1.0000000000000p+3'
    plan.write_text(json.dumps(data))
    after_plan = identities(root)
    for name in ('select_adaln_weights', 'compute_adaln_tables', 'apply_adaln', 'retable_adaln',
                 'lanes', 'retable'):
        assert after_plan[name] != after_kernel[name]
    assert after_plan['normalize_component'] == after_sdxl['normalize_component']
    changed_native_version(root, "tensorfs")
    after_native = identities(root)
    assert all(after_native[name] != after_plan[name] for name in before)
    return {'qualified': 'source-identity-only', 'operations': sorted(before),
            'caller_stable': True, 'sdxl_helper_invalidates': True,
            'sdxl_resource_invalidates': True, 'h3_kernel_invalidates': True,
            'h3_resource_invalidates': True, 'native_version_invalidates': True, 'regular_cli_matrix_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='model-memo-proof-') as temporary:
        result = qualify(Path(temporary))
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
