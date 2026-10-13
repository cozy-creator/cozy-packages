#!/usr/bin/env python3
"""Create NEW private benchmark clones from supplied immutable cohort files, CPU only."""
import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile

from prepare import HERE, SPEC, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--runtime-wheel', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--agent', type=Path, required=True)
    parser.add_argument('--tensorfs-wheel', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/fidika/cozy/.worktrees/packages/tracker-246-retry-current'))
    parser.add_argument('--spec', type=Path, default=SPEC)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9-]+', args.cohort): parser.error('cohort must be a simple unique lowercase label')
    root = HERE / 'cohorts' / args.cohort
    if root.exists(): parser.error('cohort exists; never overwrite earlier prepared artifacts')
    wheel, agent, tfs = (path.resolve(strict=True) for path in (args.runtime_wheel, args.agent, args.tensorfs_wheel))
    if sha(wheel) != args.runtime_sha256: parser.error('Runtime artifact SHA mismatch')
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        members = [name for name in archive.namelist() if name.endswith('.data/scripts/cozy-machine')]
        assert len(metadata) == len(members) == 1
        version = BytesParser().parsebytes(archive.read(metadata[0]))['Version']
        assert hashlib.sha256(archive.read(members[0])).hexdigest() == sha(agent), 'agent is not the wheel-owned executable'
    root.mkdir(parents=True)
    source = args.source.resolve(strict=True)
    manifest = {'owner': 'audit_rollout_completion', 'cohort': args.cohort,
                'created_at': datetime.now(timezone.utc).isoformat(), 'source_worktree': str(source),
                'source_head': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                'runtime_wheel': str(wheel), 'runtime_sha256': sha(wheel), 'runtime_version': version,
                'agent': str(agent), 'agent_sha256': sha(agent), 'tensorfs_wheel': str(tfs), 'tensorfs_sha256': sha(tfs),
                'public_memory_release_target': '0.18.89',
                'release_reservation': 'Public Runtime .88 belongs to boundary-only release; private benchmark clones are never publication metadata.',
                'packages': {}}
    for name in ('sdxl', 'anima'):
        folder = root / 'packages' / name
        shutil.copytree(source / name, folder, ignore=shutil.ignore_patterns('.venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'))
        original = {str(path.relative_to(folder)): sha(path) for path in folder.rglob('*') if path.is_file()}
        project = f'cozy-memory-bench-{name}-{args.cohort}'
        raw = (folder / 'pyproject.toml').read_text()
        raw, count = re.subn(r'^name = "' + name + '"$', 'name = "' + project + '"', raw, count=1, flags=re.M); assert count == 1
        raw = re.sub(r'^requires-python = .*$', 'requires-python = ">=3.12,<3.13"', raw, count=1, flags=re.M)
        raw, count = re.subn(r'"cozy-runtime\[media\][^"]+"', '"cozy-runtime[media]==' + version + '"', raw, count=1); assert count == 1
        raw = re.sub(r'"torch[<>=!~][^"]+"', '"torch==2.14.0+cu130"', raw)
        raw = re.sub(r'"torchvision[<>=!~][^"]+"', '"torchvision==0.29.0+cu130"', raw)
        assert '[tool.uv.sources]' not in raw, 'review new author source declarations before composing private pins'
        raw += '\n[tool.uv.sources]\ncozy-runtime = {path = ' + json.dumps(str(wheel)) + '}\ntensorfs = {path = ' + json.dumps(str(tfs)) + '}\ntorch = {index = "pytorch-cu130"}\n'
        if name == 'anima': raw += 'torchvision = {index = "pytorch-cu130"}\n'
        raw += '\n[[tool.uv.index]]\nname = "pytorch-cu130"\nurl = "https://download.pytorch.org/whl/cu130"\nexplicit = true\n'
        (folder / 'pyproject.toml').write_text(raw)
        subprocess.run(['uv', 'lock', '--project', str(folder)], check=True)
        manifest['packages'][name] = {'path': str(folder), 'target': f'local/{project}/generate',
            'source_files_sha256': original,
            'changes': 'private name/Python3.12/cohort and matching Torch wheel sources only; author modules unchanged'}
    (root / 'package-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([sys.executable, str(HERE / 'prepare.py'), '--root', str(root), '--spec', str(args.spec.resolve())], check=True)
    print(root / 'input-manifest.json')


if __name__ == '__main__':
    main()
