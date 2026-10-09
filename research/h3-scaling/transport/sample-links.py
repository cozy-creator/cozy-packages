"""Read-only NVLink counter sampler. Never resets or configures a counter."""
import argparse,json,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=21600);p.add_argument('--interval',type=float,default=2)
    p.add_argument('--kinds',nargs='+',choices=('d','r'),required=True);a=p.parse_args()
    until=time.monotonic()+a.seconds;a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'nvlink.jsonl').open('a',buffering=1) as log:
        while time.monotonic()<until and not (a.out/'stop').exists():
            begin=time.monotonic()
            for kind in a.kinds:
                started=time.time_ns();command=['nvidia-smi','nvlink','-gt',kind]
                try:
                    result=subprocess.run(command,text=True,capture_output=True,timeout=10)
                    row={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
                except subprocess.TimeoutExpired:
                    row={'error':'read-only query timeout'}
                row.update({'kind':kind,'start_unix_ns':started,'end_unix_ns':time.time_ns(),
                            'command':command,'documented_unit':'KiB','semantics':'Raw NVIDIA counter read; do not assume a rate or total without verifying the installed driver semantics.'})
                log.write(json.dumps(row,separators=(',',':'))+'\n')
            time.sleep(max(0,a.interval-(time.monotonic()-begin)))

if __name__=='__main__':main()
