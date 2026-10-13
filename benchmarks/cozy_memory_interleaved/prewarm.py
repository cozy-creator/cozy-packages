#!/usr/bin/env python3
"""Bounded public TensorFS reads; no GPU/model import, fetch, cache flush or retention write."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def read_objects(lease, buffer):
    verified = []
    view = memoryview(buffer)
    for identity, length in lease.objects:
        digest = hashlib.sha256()
        offset = 0
        while offset < length:
            size = min(len(buffer), length - offset)
            target = view[:size]
            lease.read_into(identity, length, offset, size, target)
            digest.update(target)
            offset += size
        if digest.hexdigest() != identity.removeprefix('sha256:'):
            raise RuntimeError('native payload SHA256 mismatch: ' + identity)
        verified.append({'id': identity, 'length': length, 'sha256_verified': True})
    return verified


def prewarm(store_path, selected, buffer_bytes=8 << 20):
    import tensorfs
    assert 'torch' not in sys.modules, 'prewarm must remain metadata/host-buffer only'
    began = time.perf_counter_ns()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    store = tensorfs.Store.open(store_path)  # Never ensure/create/materialize a user's store.
    buffer = bytearray(buffer_bytes)
    models = []
    for name, checkpoint in selected:
        started = time.perf_counter_ns()
        row = store.manifest(checkpoint)
        assert hashlib.sha256(row['manifest']).hexdigest() == checkpoint.removeprefix('sha256:')
        assert row['header'] is not None, 'checkpoint is not a native tensor payload'
        # Public verified read lease protects against GC, and is always released.
        with store.acquire_cozytensors(checkpoint) as lease:
            objects = read_objects(lease, buffer)
        models.append({'model': name, 'checkpoint': checkpoint,
                       'header_sha256': hashlib.sha256(row['header']).hexdigest(),
                       'objects': objects, 'verified_bytes': sum(o['length'] for o in objects),
                       'duration_ms': (time.perf_counter_ns() - started) / 1e6})
    assert 'torch' not in sys.modules
    return {'tensorfs_version': tensorfs.__version__, 'store': str(store_path), 'models': models,
            'duration_ms': (time.perf_counter_ns() - began) / 1e6, 'client_buffer_bytes': buffer_bytes,
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'verified_bytes': sum(row['verified_bytes'] for row in models),
            'cpu_user_s': resource.getrusage(resource.RUSAGE_SELF).ru_utime - usage_before.ru_utime,
            'cpu_system_s': resource.getrusage(resource.RUSAGE_SELF).ru_stime - usage_before.ru_stime,
            'filesystem_input_blocks': resource.getrusage(resource.RUSAGE_SELF).ru_inblock - usage_before.ru_inblock,
            'cache_regime': 'warm intent: sequential verified reads of native objects used by this suite; pages are not pinned and residency is not guaranteed under shared host RAM pressure',
            'torch_imported': False, 'gpu_or_machine_acquired': False,
            'mutation': 'no imports/materialization/GC/ref changes; only transient public read leases, released before engine startup'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--read-weights', action='store_true', help='must only run before the owned Cozy cell; bulk reads can evict Comfy cache')
    args = p.parse_args()
    if not args.read_weights: p.error('bulk read intentionally gated by --read-weights')
    if args.out.exists(): p.error('preserve prior prewarm evidence; use a fresh path')
    manifest = json.loads(args.manifest.read_text())
    spec = json.loads(Path(manifest['spec']).read_text())
    assert hashlib.sha256(Path(manifest['spec']).read_bytes()).hexdigest() == manifest['spec_sha256']
    provenance = json.loads(Path(manifest['input_provenance']).read_text())
    wanted = {job['model'] for job in manifest['suites'][args.suite]}
    # Match Comfy's source-group traversal, excluding checkpoints unused by a suite.
    order = list(dict.fromkeys(key.split('_', 1)[0] for key in spec['sources'] if key.split('_', 1)[0] in wanted))
    assert set(order) == wanted
    selected = [(name, spec['models'][name]['checkpoint']) for name in order]
    for name, checkpoint in selected: assert provenance['models'][name]['checkpoint'] == checkpoint
    result = prewarm(Path(provenance['store']), selected)
    result.update(suite=args.suite, spec_sha256=manifest['spec_sha256'], manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out: json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}))


if __name__ == '__main__': main()
