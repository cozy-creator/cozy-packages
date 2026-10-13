"""CPU-only control test: mocked CLI and sampler, never an engine or GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import run


def verify(suite):
    manifest_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else run.HERE / 'input-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    plan = manifest['suites'][suite]
    calls = []
    accepted = []
    env = {'pid': os.getpid(), 'start_ticks': 1, 'cgroup': '/user.slice/app.slice/test.scope',
           'effective_limits': {'memory_max_bytes': None}, 'affinity': [0]}
    class Sampler:
        def __init__(self, pid, gpu, path): self.pid = pid
        def sample(self): return {'compute_processes': [], 'gpu_free_bytes': 8 << 30}
        def start(self): pass
        def stop(self): pass
    def command(argv, **kwargs):
        calls.append(argv)
        if len(argv) > 1 and argv[1].endswith('/prewarm.py'):
            Path(argv[argv.index('--out') + 1]).write_text(json.dumps({'cache_regime': 'CPU mock; no real weights read'}))
            return SimpleNamespace(returncode=0, stdout='{}', stderr='')
        assert argv[0] == 'test-only-cozy'
        args = argv[1:]
        if args[:2] == ['machine', 'show']:
            data = {'running': bool(accepted), 'pid': os.getpid(), 'agent': {'sha256': manifest['agent_sha256']}, 'runtime': {'version': manifest['runtime_version']}}
        elif args[0] == 'run' and args[1] != 'show':
            index = len(accepted)
            if index: assert accepted[-1]['lease_seen'], 'successor submitted before predecessor lease'
            payload = json.loads(Path(next(a.split('=', 1)[1] for a in args if a.startswith('--input='))).read_text())
            assert payload == plan[index]['payload']
            assert args[1] == plan[index]['target']
            assert 'model.model=' + plan[index]['model_ref'] in args
            assert not any(a.startswith('--rental') or a.startswith('--rent-new') for a in args)
            destination = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--out=')))
            accepted.append({'polls': 0, 'lease_seen': False, 'destination': destination, 'index': index})
            data = {'run': index + 100, 'request_id': f'test-{index}'}
        else:
            assert args[:2] == ['run', 'show']
            index = int(args[2]) - 100; state = accepted[index]; state['polls'] += 1
            if state['polls'] >= 2: state['lease_seen'] = True
            done = state['polls'] >= 4 or '--full' in args
            data = {'status': 'completed' if done else 'running', 'machine': 'local', 'request_id': f'test-{index}',
                    'stages': [{'kind': 'gpu', 'start_unix_ms': 1000 + index}] if state['lease_seen'] else [],
                    'calls': [], 'runtime': manifest['runtime_version'],
                    'resolved': {'models': [{'manifest': {'digest': plan[index]['checkpoint']}}]},
                    'triage': {'measurements': {'ledger': {'closed': True}}}, 'result': {}}
            if done:
                path = state['destination'] / 'image.webp'
                if not path.exists():
                    path.parent.mkdir(parents=True); Image.new('RGB', (1024, 1024)).save(path, format='WEBP')
                data['output'] = [{'path': str(path), 'status': 'completed', 'length': path.stat().st_size,
                                   'sha256': 'sha256:' + run.sha(path), 'media_type': 'image/webp'}]
                data['gpus'] = [{'pid': os.getpid()}]
        return SimpleNamespace(returncode=0, stdout=json.dumps(data), stderr='')
    with tempfile.TemporaryDirectory(prefix='cozy-benchmark-control-') as temp:
        root = Path(temp); peer = root / 'peer.json'; peer.write_text(json.dumps(env))
        argv = ['run.py', '--manifest', str(manifest_path), '--suite', suite, '--budget-label', 'uncapped', '--out', str(root / 'cell'),
                '--cozy', 'test-only-cozy', '--execute-gpu', '--ownership-handoff', 'CPU-mock-only',
                '--peer-environment', str(peer), '--poll-seconds', '.001']
        with patch.object(sys, 'argv', argv), patch.object(run.subprocess, 'run', command), \
             patch.object(run.subprocess, 'check_output', return_value='test-cli'), \
             patch.object(run, 'process_environment', return_value=env), \
             patch.dict(sys.modules, {'telemetry': SimpleNamespace(Sampler=Sampler)}), \
             contextlib.redirect_stdout(io.StringIO()):
            run.main()
        result = json.loads((root / 'cell/results.json').read_text())
        assert result['order'] == [j['model'] for j in plan]
        assert len(result['jobs']) == len(plan)
        assert result['cold_engine_first_run_to_last_export_mtime_ms'] >= 0
        assert result['verification_and_observer_delay_ms'] >= 0
        assert result['ready_service_queue_ms'] is None
        assert 'total_first_submit_to_last_saved_ms' not in result
        assert all(j['outputs'] for j in result['jobs'])
        assert not any('cancel' in a or 'stop' in a for a in calls)
    print(suite, 'CPU mock control passed', len(plan), 'owned submissions')


if __name__ == '__main__':
    verify('alternating')
    verify('reuse')
