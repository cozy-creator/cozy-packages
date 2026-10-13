#!/usr/bin/env python3
"""Freeze payloads/provenance using AST and files only; never import either model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tomllib

HERE = Path(__file__).resolve().parent
SPEC = Path('/home/fidika/cozy/.worktrees/ComfyUI/memory-benchmark-20260929/benchmarks/memory_interleaved/spec.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plan(spec, suite, packages):
    rows = []
    occurrences = {}
    for index, model in enumerate(spec['suites'][suite]):
        selected = spec['models'][model]
        package = 'anima' if model == 'Anima' else 'sdxl'
        occurrence = occurrences.get(model, 0)
        occurrences[model] = occurrence + 1
        seeds = spec.get('seeds_by_model', {}).get(model)
        seed = seeds[occurrence] if seeds is not None else spec['base_seed'] + index
        payload = {'prompt': selected['prompt'], 'negative_prompt': selected['negative'],
                   'steps': selected['steps'], 'guidance': selected['cfg'],
                   'seed': seed, 'aspect_ratio': '1:1', 'megapixels': 1}
        if package == 'sdxl':
            payload['hidiffusion'] = False
            payload['vae_tile_size'] = None  # authored preflight choice; reported, never changed here
        else:
            payload.update(quality_prefix=selected['cozy_quality_prefix'],
                           cfg_interval_start=selected['cozy_cfg_interval'][0],
                           cfg_interval_stop=selected['cozy_cfg_interval'][1], first_block_cache=0.0)
        assert spec['width'] == spec['height'] == 1024
        rows.append({'index': index + 1, 'model': model, 'seed': payload['seed'],
                     'target': packages[package]['target'], 'model_ref': selected['cozy_model'],
                     'checkpoint': selected['checkpoint'], 'payload': payload})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--spec', type=Path, default=SPEC)
    args = parser.parse_args()
    root, spec_path = args.root.resolve(), args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    source = json.loads((root / 'package-provenance.json').read_text())
    for name, package in source['packages'].items():
        folder = Path(package['path'])
        module = folder / name / '__init__.py'
        tree = ast.parse(module.read_text())
        wanted = 'GenerateInput' if name == 'anima' else 'Txt2ImgInput'
        schema = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == wanted)
        package['payload_fields'] = sorted(node.target.id for node in schema.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name))
        package['frozen_files_sha256'] = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        lock = tomllib.loads((folder / 'uv.lock').read_text())
        package['locked_versions'] = {row['name']: row['version'] for row in lock['package'] if row['name'] in ('cozy-runtime', 'tensorfs', 'torch', 'torchvision', 'diffusers', 'transformers')}
        assert package['locked_versions']['cozy-runtime'] == source['runtime_version']
        assert package['locked_versions']['torch'] == '2.14.0+cu130'
        for relative, digest in package['source_files_sha256'].items():
            if relative not in ('pyproject.toml', 'uv.lock'):
                assert sha(folder / relative) == digest, f'author source changed: {relative}'
    rows = {suite: plan(spec, suite, source['packages']) for suite in spec['suites']}
    for jobs in rows.values():
        for job in jobs:
            package = source['packages']['anima' if job['model'] == 'Anima' else 'sdxl']
            assert set(job['payload']).issubset(package['payload_fields'])
    provenance = Path(spec['input_provenance'])
    matched = json.loads(provenance.read_text())
    for model, selected in spec['models'].items():
        assert matched['models'][model]['checkpoint'] == selected['checkpoint']
    manifest = {**source, 'benchmark_variant': spec.get('benchmark_variant', 'unspecified'), 'spec': str(spec_path), 'spec_sha256': sha(spec_path),
                'input_provenance': str(provenance), 'input_provenance_sha256': sha(provenance),
                'suites': rows, 'output_difference': 'Cozy author packages encode WebP; stock Comfy SaveImage writes PNG. Encoding/export cost remains in primary wall; no bitwise cross-engine image equality is claimed.',
                'timing': 'Cold engine + sequence: first ordinary cozy run (implicit startup) to final export file mtime; observer/RGB verification end and delay reported separately. Runtime source environment preparation stays inside cold total. Compare with Comfy startup_ms + ready-server queue; no invented Cozy readiness subtraction. Optional client registration separately timed.',
                'scheduler': 'one pending successor, submitted only after previous request has a recorded GPU lease; verify final lease order'}
    (root / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(root / 'input-manifest.json')


if __name__ == '__main__':
    main()
