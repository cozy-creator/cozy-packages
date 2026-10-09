"""Research-only NCCL control, called by package warmup on every rank before join.

Does not change sealed values, model math, collectives or the installed Runtime.
Only valid for a fresh four-rank group; route evidence remains a separate check.
"""
from __future__ import annotations

import json
import os
import time

PREFIX = 'COZY_H3_TRANSPORT_V1 '
COMMON = {
    'NCCL_SHM_DISABLE': '0',
    'NCCL_NET': 'Socket',
    'NCCL_NET_GDR_LEVEL': 'LOC',
    'NCCL_NET_GDR_C2C': '0',
    'NCCL_MNNVL_ENABLE': '0',
    'NCCL_DEBUG': 'INFO',
    'NCCL_DEBUG_SUBSYS': 'INIT,GRAPH,P2P,SHM,NET,ENV,NVLS',
    # NCCL2.30.7 debug.cc: empty means retain stdout; no fopen on a shared fd.
    'NCCL_DEBUG_FILE': '',
}
SEALED = {'NCCL_NVLS_ENABLE': '0', 'NCCL_P2P_LEVEL': 'NVL'}


def configure(mode: str) -> dict:
    import torch

    if mode not in ('peer', 'host'):
        raise ValueError('transport control must be peer or host')
    if torch.distributed.is_initialized():
        raise RuntimeError('transport controls must precede communicator initialization')
    if any(os.environ.get(k) != v for k, v in SEALED.items()):
        raise RuntimeError('expected the ordinary group seal NVLS=0 and P2P_LEVEL=NVL')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 4 or any(not d.strip() for d in devices):
        raise RuntimeError('transport comparison requires a fresh four-GPU group')
    configured = {**COMMON, 'NCCL_P2P_DISABLE': '1' if mode == 'host' else '0'}
    os.environ.update(configured)
    receipt = {
        'mode': mode, 'pid': os.getpid(), 'unix_ns': time.time_ns(),
        'before_process_group': True, 'environment': configured,
        'unchanged_seal': {k: os.environ[k] for k in SEALED},
        'visible_devices': devices,
        'evidence_limit': 'Requested settings only; accept measured arm only after NCCL channel-route logs agree.',
    }
    os.write(2, ('\n' + PREFIX + json.dumps(receipt, separators=(',', ':')) + '\n').encode())
    return receipt
