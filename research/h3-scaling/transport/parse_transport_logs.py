"""Validate observed four-rank NCCL channel routes against a completed ordinaryCLI run.

Use the warmup receipt and its actual leader fd1/fd2 logs. Timed requests must reuse
those four PIDs; this checks setup routes, not physical byte counters or wire timing.
"""
from pathlib import Path
import argparse
import json
import re
from transport_control import COMMON, PREFIX, SEALED

PID = re.compile(r'\S+:(\d+):\d+\s+\[[^]]+\]\s+NCCL INFO')
CHANNEL = re.compile(r'Channel\s+\S+\s*:\s*(\d+)\[(\d+)\]\s*->\s*(\d+)\[(\d+)\](?:\s+\[[^]]+\])?\s+via\s+(\S+)')
VERSION = re.compile(r'NCCL version\s+(\d+\.\d+\.\d+)')


def collect(show, paths, mode):
    if mode not in ('peer','host') or show.get('degree') != 4 or show.get('status') != 'completed':
        raise ValueError('expected peer/host and a completed four-GPU run')
    pids={g['pid'] for g in show['gpus']}
    physical_gpus={g['gpu'] for g in show['gpus']}
    if len(physical_gpus)!=4: raise ValueError('Receipt must identify four physical GPU indices')
    if len(pids)!=4: raise ValueError('receipt must identify four distinct rank PIDs')
    controls=[];routes=[];issues=[];versions=set()
    for path in paths:
        with Path(path).open(errors='replace') as stream:
            for n,line in enumerate(stream,1):
                if PREFIX in line:
                    try: row=json.loads(line.split(PREFIX,1)[1])
                    except json.JSONDecodeError:
                        issues.append(f'malformed control receipt at {path}:{n}');continue
                    if row.get('pid') in pids:
                        controls.append({**row,'source':str(path),'line':n})
                pidmatch=PID.search(line)
                if not pidmatch or int(pidmatch[1]) not in pids: continue
                version=VERSION.search(line)
                if version: versions.add(version[1])
                channel=CHANNEL.search(line)
                if channel:
                    ar,a,br,b,route=channel.groups();ar,a,br,b=map(int,(ar,a,br,b))
                    if a==b: continue
                    routes.append({'pid':int(pidmatch[1]),'from_nccl_rank':ar,'to_nccl_rank':br,
                                   'from_gpu':a,'to_gpu':b,
                                   'transport':route,'source':str(path),'line':n})
    expected={**COMMON,'NCCL_P2P_DISABLE':'1' if mode=='host' else '0'}
    if sorted(r['pid'] for r in controls)!=sorted(pids): issues.append('missing or repeated before-communicator control receipts')
    for r in controls:
        if r.get('mode')!=mode or r.get('environment')!=expected or r.get('unchanged_seal')!=SEALED or not r.get('before_process_group'):
            issues.append(f"control receipt mismatch for PID {r['pid']}")
    if versions!={'2.30.7'}: issues.append(f'expected NCCL2.30.7, observed {sorted(versions)}')
    required={(a,b) for a in physical_gpus for b in physical_gpus if a!=b}
    observed={(r['from_gpu'],r['to_gpu']) for r in routes}
    if observed!=required: issues.append(f'missing/unexpected channel links: missing={sorted(required-observed)}, extra={sorted(observed-required)}')
    wanted='P2P/' if mode=='peer' else 'SHM/'
    wrong=[r for r in routes if not r['transport'].startswith(wanted) or 'MNNVL' in r['transport']]
    if wrong: issues.append('observed another transport; requested arm is not established')
    if {r['pid'] for r in routes}!=pids: issues.append('route logs do not cover every assigned process')
    return {'run':show['number'],'mode':mode,'complete':not issues,'issues':issues,
            'pids':sorted(pids),'versions':sorted(versions),'control_receipts':controls,
            'route_counts':{t:sum(r['transport']==t for r in routes) for t in sorted({r['transport'] for r in routes})},
            'directed_physical_gpu_links':sorted(observed),'routes':routes,
            'link_identity':'Bracketed NCCL nvmlDev indices, mapped to the CLI physical GPU set; communicator-local rank labels are retained separately.',
            'limits':['NCCL connection-route evidence, not physical byte or transfer-time measurement.',
                      'Use completed full-H3 receipt separately for successful 8-step execution and quality.',
                      'Timed cases must reuse this exact PID group; a fresh group needs its own route evidence.',
                      'Native controls affect all NCCL group traffic, including VAE results; Comfy switch is DiT-only.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--show',type=Path,required=True);p.add_argument('--log',type=Path,action='append',required=True)
    p.add_argument('--mode',choices=('peer','host'),required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=collect(json.loads(a.show.read_text()),a.log,a.mode)
    a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('run','mode','complete','issues','route_counts')}))
    raise SystemExit(0 if r['complete'] else 1)

if __name__=='__main__':main()
