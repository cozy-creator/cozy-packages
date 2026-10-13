#!/usr/bin/env python3
"""Ordinary default-home Cozy benchmark; execution requires explicit GPU ownership."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from environment import comparable, process_environment
from prepare import sha

HERE = Path(__file__).resolve().parent
TERMINAL_FAILURE = {'failed', 'canceled', 'paused', 'refused', 'blocked'}


def write(path, value):
    scratch = path.with_suffix(path.suffix + '.new')
    scratch.write_text(json.dumps(value, indent=2) + '\n')
    scratch.replace(path)


def gpu_starts(report):
    scopes = [report, *report.get('calls', [])]
    return [stage['start_unix_ms'] for scope in scopes for stage in scope.get('stages', [])
            if stage.get('kind') == 'gpu' and stage.get('start_unix_ms', 0) > 0]


def verify_outputs(report, destination):
    from PIL import Image
    outputs = report.get('output', [])
    if not outputs or any(row.get('status') != 'completed' or not row.get('path') for row in outputs):
        return None
    verified = []
    for row in outputs:
        path = Path(row['path']).resolve()
        if not path.is_relative_to(destination.resolve()) or not path.is_file():
            raise RuntimeError('ordinary export escaped its fresh owned destination or is absent')
        digest = sha(path)
        if digest != row['sha256'].removeprefix('sha256:') or path.stat().st_size != row['length']:
            raise RuntimeError('ordinary output bytes disagree with receipt')
        with Image.open(path) as picture:
            assert picture.size == (1024, 1024), picture.size
            rgb_sha = hashlib.sha256(picture.convert('RGB').tobytes()).hexdigest()
        verified.append({**row, 'verified_sha256': digest, 'rgb_sha256': rgb_sha,
                         'export_mtime_ns': path.stat().st_mtime_ns})
    return verified


def receipt_ready(report):
    """Completion, accounting and authored result are independently observed fields."""
    if report.get('status') in TERMINAL_FAILURE:
        raise RuntimeError('owned run ended ' + report['status'] + '; no cancellation issued')
    ledger = report.get('triage', {}).get('measurements', {}).get('ledger', {})
    return report.get('status') == 'completed' and ledger.get('closed') is True and 'result' in report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--budget-label', choices=['uncapped', '6GiB', '3GiB', '1GiB'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-state', choices=['cold', 'ready'], default='cold',
                        help='cold starts the machine implicitly; ready measures from an already running empty machine')
    parser.add_argument('--cozy', default='cozy')
    parser.add_argument('--ballast-receipt', type=Path)
    parser.add_argument('--prewarm-python', type=Path, default=Path('/home/fidika/cozy_v2/outputs/comfy-cozy-memory-20260929/cozy/prewarm-venv/bin/python'))
    parser.add_argument('--peer-environment', type=Path, help='matched Comfy server process_environment JSON')
    parser.add_argument('--register-packages', action='store_true')
    parser.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--ownership-handoff', default='')
    parser.add_argument('--max-seconds', type=float, default=0, help='explicit observer budget only; never a cancellation deadline')
    parser.add_argument('--poll-seconds', type=float, default=.25)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    if args.suite not in manifest['suites']:
        parser.error('suite is absent from selected manifest')
    plan = manifest['suites'][args.suite]
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    if not args.execute_gpu or not args.ownership_handoff:
        parser.error('explicit exclusive GPU handoff and --execute-gpu required')
    if os.environ.get('COZY_HOME') not in (None, str(Path.home() / '.cozy')):
        parser.error('this comparison uses the ordinary default home, not a fixture override')
    if args.budget_label != 'uncapped' and args.ballast_receipt is None:
        parser.error('physical-pressure cell requires the shared ballast receipt')
    if args.poll_seconds <= 0:
        parser.error('positive polling interval required')
    assert sha(manifest['runtime_wheel']) == manifest['runtime_sha256']
    assert sha(manifest['spec']) == manifest['spec_sha256']
    assert sha(manifest['input_provenance']) == manifest['input_provenance_sha256']
    for package in manifest['packages'].values():
        for relative, digest in package['frozen_files_sha256'].items():
            assert sha(Path(package['path']) / relative) == digest, relative
    import yaml
    policy_path = Path.home() / '.cozy/machine/root/etc/cozy/runtime.yaml'
    policy = yaml.safe_load(policy_path.read_text()) if policy_path.exists() else {}
    budget = (policy or {}).get('gpu', {}).get('budget')
    if budget not in (None, 0, '0', 'off'):
        raise RuntimeError('Cozy gpu.budget must be off; this benchmark uses physical ballast')
    args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'input-manifest.json', manifest)
    metadata = {'input_manifest': str(args.manifest.resolve()), 'input_manifest_sha256': sha(args.manifest), 'suite': args.suite, 'budget_label': args.budget_label, 'start_state': args.start_state, 'ownership_handoff': args.ownership_handoff,
                'gpu_budget_policy': policy, 'driver_environment': process_environment(os.getpid()),
                'observer_poll_seconds': args.poll_seconds, 'output_difference': manifest['output_difference'],
                'timing': manifest['timing'], 'invocations': []}
    write(args.out / 'metadata.json', metadata)
    def cli(argv, label):
        began = time.perf_counter_ns()
        result = subprocess.run([args.cozy, *argv], text=True, capture_output=True)
        (args.out / (label + '.stdout')).write_text(result.stdout)
        (args.out / (label + '.stderr')).write_text(result.stderr)
        metadata['invocations'].append({'argv': [args.cozy, *argv], 'began_ns': began,
                                        'ended_ns': time.perf_counter_ns(), 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError(f'ordinary CLI failed: {label}; see saved stdout/stderr; do not blindly resubmit')
        return json.loads(result.stdout)
    rows, monitor = [], None
    first_submit = None; first_unix = None
    cell_start = time.perf_counter_ns()
    try:
        initial = cli(['machine', 'show', '--json'], 'machine-before')
        if args.start_state == 'cold' and initial.get('running'):
            raise RuntimeError('fresh-engine cell requires stopped default machine; driver never stops it')
        if args.start_state == 'ready' and not initial.get('running'):
            raise RuntimeError('ready-service cell requires an already running empty machine')
        if '/agents.slice/' in metadata['driver_environment']['cgroup']:
            raise RuntimeError('scored driver must run in its owned user scope outside agents.slice')
        metadata['machine_before'] = initial
        metadata['cli_version'] = subprocess.check_output([args.cozy, '--version'], text=True).strip()
        if args.ballast_receipt:
            ballast = json.loads(args.ballast_receipt.read_text())
            expected = {'6GiB': 6, '3GiB': 3, '1GiB': 1}.get(args.budget_label)
            if expected is None or ballast.get('gpu') != 0 or ballast['target_free_bytes'] != expected * (1 << 30):
                raise RuntimeError('ballast receipt is not this pressure cell')
            metadata['ballast'] = {**ballast, 'environment': process_environment(ballast['pid'])}
            write(args.out / 'ballast.json', metadata['ballast'])
        if args.register_packages:
            began = time.perf_counter_ns()
            for name, package in manifest['packages'].items():
                cli(['package', 'install', package['path'], '--editable', '--json'], 'register-' + name)
            metadata['client_package_registration_ms'] = (time.perf_counter_ns() - began) / 1e6
            if args.start_state == 'cold' and cli(['machine', 'show', '--json'], 'machine-after-registration').get('running'):
                raise RuntimeError('package registration unexpectedly started the machine; preserve it and review before scoring')
        prewarm_command = [str(args.prewarm_python), str(HERE / 'prewarm.py'), '--manifest', str(args.manifest.resolve()),
                           '--suite', args.suite, '--out', str(args.out / 'prewarm.json'), '--read-weights']
        started = time.perf_counter_ns()
        prewarmed = subprocess.run(prewarm_command, text=True, capture_output=True)
        (args.out / 'prewarm.stdout').write_text(prewarmed.stdout)
        (args.out / 'prewarm.stderr').write_text(prewarmed.stderr)
        if prewarmed.returncode: raise RuntimeError('public TensorFS prewarm failed; no benchmark run submitted')
        metadata['prewarm'] = json.loads((args.out / 'prewarm.json').read_text())
        metadata['prewarm_wall_ms'] = (time.perf_counter_ns() - started) / 1e6
        from telemetry import Sampler
        monitor = Sampler(os.getpid(), 0, args.out / 'telemetry.jsonl')
        metadata['before_first_submit_gpu'] = monitor.sample()
        ballast_pid = metadata.get('ballast', {}).get('pid')
        other = [row for row in metadata['before_first_submit_gpu'].get('compute_processes', []) if row['pid'] != ballast_pid]
        if other: raise RuntimeError('exclusive GPU window contains another compute process; nothing was submitted')
        if args.ballast_receipt:
            metadata['pre_engine_free_delta_from_calibration'] = metadata['before_first_submit_gpu']['gpu_free_bytes'] - metadata['ballast']['observed_free_bytes']
        monitor.start()
        token = uuid.uuid4().hex
        def submit(job):
            nonlocal first_submit, first_unix
            label = f"{job['index']:02d}-{job['model']}"
            payload = args.out / (label + '-input.json'); write(payload, job['payload'])
            destination = args.out / 'images' / label
            key = f"memory-benchmark-{token}-{job['index']}"
            argv = ['run', job['target'], 'model.model=' + job['model_ref'], '--input=' + str(payload),
                    '--out=' + str(destination), '--idempotency-key=' + key, '--json']
            write(args.out / (label + '-intent.json'), {'argv': [args.cozy, *argv], 'job': job, 'idempotency_key': key})
            began = time.perf_counter_ns(); unix = time.time_ns()
            if first_submit is None: first_submit, first_unix = began, unix
            receipt = cli(argv, label + '-submit')
            number = int(receipt['run'])
            row = {**job, 'number': number, 'submit_ns': began, 'submit_unix_ns': unix,
                   'idempotency_key': key, 'receipt': receipt, 'destination': str(destination)}
            rows.append(row); write(args.out / 'owned-runs.json', rows)
        submit(plan[0]); sequence = 1; observation = 0
        while any('outputs_verified_ns' not in row for row in rows) or sequence < len(plan):
            if args.max_seconds and (time.perf_counter_ns() - first_submit) / 1e9 > args.max_seconds:
                raise TimeoutError('explicit observation budget exceeded; accepted runs remain untouched')
            observation += 1
            for row in list(rows):
                if 'outputs_verified_ns' in row: continue
                report = cli(['run', 'show', str(row['number']), '--json'], f"{row['index']:02d}-observe-{observation}")
                write(args.out / f"run-{row['number']}.json", report)
                row['request_id'] = report['request_id']
                if report.get('machine') not in ('local', 'this computer'):
                    raise RuntimeError('ordinary default run did not select local machine')
                starts = gpu_starts(report)
                if starts and 'lease_start_unix_ms' not in row:
                    row['lease_start_unix_ms'] = min(starts)
                    row['lease_observed_ns'] = time.perf_counter_ns()
                if report['status'] in TERMINAL_FAILURE:
                    raise RuntimeError(f"owned run {row['number']} ended {report['status']}; no cancellation issued")
                if report['status'] == 'completed':
                    outputs = verify_outputs(report, Path(row['destination']))
                    if outputs is not None:
                        row['outputs'] = outputs; row['outputs_verified_ns'] = time.perf_counter_ns(); row['outputs_verified_unix_ns'] = time.time_ns()
                if 'machine_environment' not in metadata:
                    state = cli(['machine', 'show', '--json'], f'machine-observe-{observation}')
                    if state.get('running') and state.get('pid'):
                        monitor.pid = int(state['pid'])
                        metadata['machine_environment'] = process_environment(monitor.pid)
                        metadata['machine_software'] = state
                        if args.peer_environment:
                            peer = json.loads(args.peer_environment.read_text())
                            metadata['peer_environment'] = peer
                            metadata['cgroup_matched'] = comparable(metadata['machine_environment'], peer)
                # At most one successor waits behind a GPU-owning predecessor.
                if row is rows[-1] and sequence < len(plan) and 'lease_start_unix_ms' in row:
                    submit(plan[sequence]); sequence += 1
            write(args.out / 'owned-runs.json', rows); write(args.out / 'metadata.json', metadata)
            if any('outputs_verified_ns' not in row for row in rows): time.sleep(args.poll_seconds)
        leases = [row['lease_start_unix_ms'] for row in rows]
        if leases != sorted(leases): raise RuntimeError('actual GPU lease order differs from requested interleaving')
        verified_end = max(row['outputs_verified_ns'] for row in rows)
        verified_wall_end = max(row['outputs_verified_unix_ns'] for row in rows)
        export_wall_end = max(output['export_mtime_ns'] for row in rows for output in row['outputs'])
        if export_wall_end < first_unix or export_wall_end > verified_wall_end:
            raise RuntimeError('export filesystem times fall outside the observed cell; cannot score wall endpoint')
        for row in rows:
            revision = 0
            while True:
                revision += 1
                full = cli(['run', 'show', str(row['number']), '--full', '--json'], f"{row['index']:02d}-full-receipt-{revision}")
                if receipt_ready(full): break
                if args.max_seconds and (time.perf_counter_ns() - first_submit) / 1e9 > args.max_seconds:
                    raise TimeoutError('explicit receipt observation budget exceeded; no cancellation issued')
                time.sleep(args.poll_seconds)
            write(args.out / f"run-{row['number']}-full.json", full)
            models = full['resolved']['models']
            if not any(model.get('manifest', {}).get('digest') == row['checkpoint'] for model in models):
                raise RuntimeError('executed model checkpoint differs from matched input manifest')
            assert manifest['runtime_version'] in full['runtime'], full['runtime']
            measurements = full['triage']['measurements']
            assert measurements['ledger']['closed'], 'owned Runtime work did not close'
            row['measurements'] = measurements
            row['source_environment_setup_stages'] = [s for s in full.get('stages', []) if s['kind'] == 'setup']
            row['gpus'] = [g for scope in [full, *full.get('calls', [])] for g in scope.get('gpus', [])]
            row['executor_pids'] = sorted({g['pid'] for g in row['gpus'] if g.get('pid')})
            row['result'] = full['result']
        final_machine = cli(['machine', 'show', '--json'], 'machine-after')
        metadata['machine_software_final'] = final_machine
        if final_machine.get('agent', {}).get('sha256') != manifest['agent_sha256']:
            raise RuntimeError('observed request-plane agent differs from selected benchmark cohort')
        observed_runtime = final_machine.get('runtime', {})
        observed_runtime = observed_runtime.get('version') if isinstance(observed_runtime, dict) else observed_runtime
        if observed_runtime != manifest['runtime_version']:
            raise RuntimeError('live machine Runtime differs from selected benchmark cohort')
        result = {'suite': args.suite, 'budget_label': args.budget_label, 'start_state': args.start_state, 'order': [row['model'] for row in rows],
                  'scored': metadata.get('cgroup_matched', False),
                  'cold_engine_first_run_to_last_export_mtime_ms': (export_wall_end - first_unix) / 1e6 if args.start_state == 'cold' else None,
                  'cold_engine_first_run_to_verified_exports_ms': (verified_end - first_submit) / 1e6 if args.start_state == 'cold' else None,
                  'verification_and_observer_delay_ms': (verified_wall_end - export_wall_end) / 1e6,
                  'export_end_unix_ns': export_wall_end,
                  'verified_end_unix_ns': verified_wall_end,
                  'ready_service_queue_ms': (export_wall_end - first_unix) / 1e6 if args.start_state == 'ready' else None,
                  'ready_service_queue_reason': 'Observed machine already running before the first request; no startup subtraction.' if args.start_state == 'ready' else 'First cozy run includes implicit startup; no startup subtraction.',
                  'comparison': 'Compare ready-service first-request-to-export with Comfy ready-server first-submit-to-saved; exclude startup for both.' if args.start_state == 'ready' else 'Compare cold Cozy first-run-to-export with Comfy startup_ms plus ready-server first-submit-to-saved.',
                  'including_optional_client_registration_to_verification_ms': (verified_end - cell_start) / 1e6,
                  'receipts_verified_ms': (time.perf_counter_ns() - first_submit) / 1e6,
                  'first_submit_unix_ns': first_unix, 'metadata': metadata, 'jobs': rows,
                  'output_difference': manifest['output_difference']}
        write(args.out / 'results.json', result); print(json.dumps(result))
    except BaseException as exc:
        write(args.out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
              'owned_runs': rows, 'policy': 'observer failure leaves accepted work intact; reconcile owned IDs/intents, never blindly resubmit'})
        raise
    finally:
        if monitor: monitor.stop()
        write(args.out / 'metadata.json', metadata)
        # No run cancel, daemon stop, package removal, or default-machine mutation.


if __name__ == '__main__':
    main()
