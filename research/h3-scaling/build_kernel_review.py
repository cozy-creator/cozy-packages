#!/usr/bin/env python3
"""Join the archived kernel table to verified original videos; no inference or encoding."""
import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def link_kernel_study(review):
    """Add navigation without rebuilding or otherwise replacing an existing gallery."""
    index = review / 'index.html'
    if not index.exists() or not (review / 'kernels.html').exists():
        return
    before = index.read_text()
    if 'href="kernels.html"' in before:
        return
    navigation = ('<nav aria-label="Benchmark studies" style="display:flex;gap:12px;'
                  'flex-wrap:wrap;margin:0 0 24px">'
                  '<strong style="padding:12px 18px;border:1px solid #53647d;'
                  'border-radius:8px;background:#214267" aria-current="page">'
                  'GPU scaling &amp; transport</strong>'
                  '<a href="kernels.html" style="padding:12px 18px;border:2px solid #94c7ff;'
                  'border-radius:8px;font-weight:700;text-decoration:none">'
                  'Attention kernel comparison · all 4 kernels · 16 videos →</a></nav>')
    if '<main>' not in before:
        raise ValueError('review index has no main element for kernel navigation')
    index.write_text(before.replace('<main>', '<main>\n' + navigation, 1))


def build(archive, source_index, output, template):
    matrix = json.loads(archive.read_text())['single_5090_attention_matrix']
    source = json.loads(source_index.read_text())
    indexed = {run['run_number']: run for run in source['runs']}
    records = []
    seen = set()
    input_hashes = {}
    for row in matrix['rows']:
        times = []
        for sample in row['samples']:
            run = indexed[sample['run']]
            key = (row['kernel'], row['schedule'], sample['seed'])
            if key in seen:
                raise ValueError(f'duplicate cell: {key}')
            seen.add(key)
            if (run['kernel'], run['schedule'], run['seed']) != key:
                raise ValueError(f'identity mismatch: {key}')
            video, input_file = Path(run['video']), Path(run['input'])
            sha = digest(video)
            if sha != sample['video_sha256'] or sha != run['sha256']:
                raise ValueError(f'video hash mismatch: {video}')
            input_sha = digest(input_file)
            if input_sha != sample['input_sha256']:
                raise ValueError(f'input hash mismatch: {input_file}')
            previous = input_hashes.setdefault(sample['seed'], input_sha)
            if previous != input_sha:
                raise ValueError('different inputs within a matched seed')
            payload = json.loads(input_file.read_text())
            if payload['seed'] != sample['seed'] or payload['duration_s'] != 15:
                raise ValueError('unexpected request geometry or seed')
            if not run['verified'] or not all(run['checks'].values()):
                raise ValueError(f'prior validation failed: {key}')
            qa_path = source_index.parent / run['qa']
            qa = json.loads(qa_path.read_text())
            if qa['sha256'] != sha or not all(qa['checks'].values()):
                raise ValueError(f'decode validation mismatch: {qa_path}')
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)
            ]))
            v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            a = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            if (v['width'], v['height'], int(v['nb_frames']), v['avg_frame_rate']) != (1344, 768, 362, '24/1'):
                raise ValueError(f'unexpected video metadata: {video}')
            if (int(a['sample_rate']), a['channels']) != (32000, 2):
                raise ValueError(f'unexpected audio metadata: {video}')
            seconds = sample['execution_seconds']
            if not math.isclose(seconds, run['execution_seconds'], abs_tol=1e-6):
                raise ValueError(f'timing mismatch: {key}')
            times.append(seconds)
            records.append(dict(
                id=run['id'], kernel=row['kernel'], schedule=row['schedule'],
                seed=sample['seed'], input='A' if sample['seed'] == 581204 else 'B',
                prompt=payload['prompt'], run=sample['run'], seconds=seconds,
                mean_seconds=row['mean_seconds'], note=run['arithmetic_note'],
                video=quote(os.path.relpath(video, output)), sha256=sha,
                bytes=video.stat().st_size, input_sha256=input_sha,
                metadata=dict(width=v['width'], height=v['height'], frames=int(v['nb_frames']),
                              fps=v['avg_frame_rate'], duration=float(v['duration']),
                              audio_sample_rate=int(a['sample_rate']), audio_channels=a['channels']),
                prior_full_av_decode=run['checks']['full_av_decode'],
                qa=quote(os.path.relpath(qa_path, output)),
            ))
        if len(times) != 2 or not math.isclose(sum(times)/2, row['mean_seconds'], abs_tol=1e-6):
            raise ValueError('incomplete pair or incorrect mean')
    expected = {(k, s, seed) for k in ('kitchen-int8','sage3-nvfp4','fp8-fp16','bf16-fp32')
                for s in ('dense4-sol4','dense8') for seed in (581204,581205)}
    if seen != expected:
        raise ValueError('kernel matrix does not contain all sixteen expected cells')
    result = dict(title='MiniMax H3 attention kernel comparison',
                  verified_at=datetime.now(timezone.utc).isoformat(),
                  archive_sha256=digest(archive), source_index_sha256=digest(source_index),
                  hardware=matrix['hardware'], runtime=matrix['runtime'],
                  linear_variant=matrix['linear_variant'], limits=matrix['limits'], runs=records)
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    (output / 'kernel-artifact-index.json').write_text(encoded + '\n')
    html = template.read_text().replace('__KERNEL_DATA__', encoded.replace('<', '\\u003c'))
    (output / 'kernels.html').write_text(html)
    link_kernel_study(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--source-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=Path(__file__).with_name('kernel_review.html'))
    args = parser.parse_args()
    result = build(args.archive.resolve(), args.source_index.resolve(), args.output.resolve(), args.template)
    print(json.dumps({'verified_videos': len(result['runs']), 'page': str(args.output / 'kernels.html')}))


if __name__ == '__main__':
    main()
