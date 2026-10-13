"""Read-only process/cgroup provenance; importing this module starts nothing."""
from pathlib import Path
import os


def process_environment(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    group = next(line.split('::', 1)[1] for line in (proc / 'cgroup').read_text().splitlines() if line.startswith('0::'))
    current = Path('/sys/fs/cgroup') / group.lstrip('/')
    hierarchy = []
    memory, high, pids, quotas = [], [], [], []
    while current.is_relative_to('/sys/fs/cgroup'):
        row = {'path': str(current)}
        for name in ('memory.max', 'memory.high', 'memory.current', 'cpu.max', 'pids.max', 'cpuset.cpus.effective'):
            if (current / name).exists():
                row[name] = (current / name).read_text().strip()
        for name, found in [('memory.max', memory), ('memory.high', high), ('pids.max', pids)]:
            if row.get(name, 'max') != 'max':
                found.append(int(row[name]))
        quota = row.get('cpu.max', 'max 100000').split()
        if quota[0] != 'max':
            quotas.append(int(quota[0]) / int(quota[1]))
        hierarchy.append(row)
        if current == Path('/sys/fs/cgroup'):
            break
        current = current.parent
    return {'pid': pid, 'start_ticks': int(stat[19]), 'cgroup': group,
            'affinity': sorted(os.sched_getaffinity(pid)), 'hierarchy': hierarchy,
            'effective_limits': {'memory_max_bytes': min(memory) if memory else None,
                                 'memory_high_bytes': min(high) if high else None,
                                 'pids_max': min(pids) if pids else None,
                                 'cpu_quota_cores': min(quotas) if quotas else None}}


def comparable(ours, peer):
    return (ours['effective_limits'] == peer['effective_limits'] and
            ours['affinity'] == peer['affinity'] and
            '/agents.slice/' not in ours['cgroup'] and '/agents.slice/' not in peer['cgroup'])
