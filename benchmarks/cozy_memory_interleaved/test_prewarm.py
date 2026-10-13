"""Real public TensorFS tiny-store check; never reads the default store."""
import hashlib
import io
from pathlib import Path
import tempfile

import tensorfs
from prewarm import prewarm

payload = bytes(range(256)) * 8192
plain = next(value for name, value in tensorfs.seed_digests() if name == 'plain/1')
with tempfile.TemporaryDirectory(prefix='cozy-prewarm-public-read-') as folder:
    root = Path(folder) / 'store'
    store = tensorfs.Store.init(root)
    writer = store.begin_derived('sha256:' + '83' * 32, 1, {}, {'model': {'drop': [], 'add': {
        'weight': {'logical_dtype': 'u8', 'shape': [len(payload)], 'encoding': plain,
                   'parts': {'value': {'dtype': 'u8', 'shape': [len(payload)]}}}}}},
        {}, [('model', 'weight')], len(payload) + (1 << 20), work_fingerprint='sha256:' + '84' * 32)
    writer.add_part('model', 'weight', 'value', io.BytesIO(payload))
    receipt = writer.commit()
    result = prewarm(root, [('test-only', 'sha256:' + receipt['manifest']['sha256'])], buffer_bytes=8192)
    assert result['client_buffer_bytes'] == 8192
    assert any(row['id'].removeprefix('sha256:') == hashlib.sha256(payload).hexdigest() for row in result['models'][0]['objects'])
    assert result['verified_bytes'] >= len(payload)
    assert not result['torch_imported'] and not result['gpu_or_machine_acquired']
    print('public TensorFS prewarm tiny-store check passed; 8KiB buffer, no default store/GPU')
