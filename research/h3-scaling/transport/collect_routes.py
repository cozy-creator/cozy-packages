"""Read the completed warmup group's captured logs and validate its NCCL routes."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile
from parse_transport_logs import collect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-dir',type=Path,required=True)
    p.add_argument('--mode',choices=('peer','host'),required=True)
    p.add_argument('--case',choices=('warmup-0','timed-1','timed-2'),default='warmup-0')
    a=p.parse_args();host=a.host_dir.resolve();folder=host/f'native-{a.mode}'/a.case
    show=json.loads((folder/'show-final.json').read_text())
    if show['status']!='completed': raise SystemExit('The selected request must be complete')
    ssh=json.loads((host/'ssh-command.json').read_text())
    pids=[g['pid'] for g in show['gpus']]
    leader=pids[0]  # CLI execution evidence lists leader then followers, verified below by argv.
    remote='''import hashlib,io,json,pathlib,sys,tarfile,time
pids=PIDS
leader=LEADER
identities=[]
for pid in pids:
 root=pathlib.Path('/proc')/str(pid)
 cmd=[v.decode() for v in (root/'cmdline').read_bytes().split(bytes([0])) if v]
 if not any('cozy_runtime.internal.executor' in v for v in cmd):raise RuntimeError('PID is not the expected Runtime executor')
 fields=(root/'stat').read_text().rsplit(')',1)[1].split()
 identities.append({'pid':pid,'start_ticks':int(fields[19]),'argv':cmd})
leader_identity=identities[0]
if '--rank' in leader_identity['argv'] and leader_identity['argv'][leader_identity['argv'].index('--rank')+1]!='0':raise RuntimeError('First receipt process is not rank zero')
rows={};payload={}
for fd,name in [('1','stdout.log'),('2','stderr.log')]:
 path=(pathlib.Path('/proc')/str(leader)/'fd'/fd).resolve()
 if not path.is_file() or not str(path).startswith('/var/lib/cozy/'):raise RuntimeError('Executor log is not a retained regular Cozy file')
 size=path.stat().st_size
 if size>32*1024*1024:raise RuntimeError('Log exceeds bounded collection; investigate before copying')
 data=path.read_bytes()
 if len(data)>32*1024*1024:raise RuntimeError('Growing log exceeds collection bound')
 payload[name]=data;rows[name]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
manifest={'captured_unix_ns':time.time_ns(),'leader_pid':leader,'processes':identities,'files':rows}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,data in {**payload,'manifest.json':json.dumps(manifest,indent=2).encode()}.items():
  info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
'''.replace('PIDS',repr(pids)).replace('LEADER',repr(leader))
    (folder/'collect-routes-remote.py').write_text(remote)
    archive_path=folder/'transport-logs.tar.gz'
    with archive_path.open('wb') as out:
        result=subprocess.run(ssh+['python3 -'],input=remote.encode(),stdout=out,stderr=subprocess.PIPE,timeout=60)
    (folder/'collect-routes.stderr').write_bytes(result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
    with tarfile.open(archive_path,'r:gz') as archive:
        manifest=json.load(archive.extractfile('manifest.json'))
        for name,expected in manifest['files'].items():
            if name not in ('stdout.log','stderr.log'):raise ValueError('Unexpected archive member')
            data=archive.extractfile(name).read()
            if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:
                raise ValueError('Transferred log custody mismatch')
            (folder/name).write_bytes(data)
    (folder/'transport-logs-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=collect(show,[folder/'stdout.log',folder/'stderr.log'],a.mode)
    (folder/'transport-proof.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('run','mode','complete','issues','route_counts')},indent=2))
    raise SystemExit(0 if report['complete'] else 1)

if __name__=='__main__':main()
