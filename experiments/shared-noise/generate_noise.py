"""Generate two canonical CPU-only quality-control inputs; never initializes CUDA."""
import hashlib
import json
from pathlib import Path
import sys
import torch

DEST = Path(sys.argv[1])
DEST.mkdir(parents=True, exist_ok=True)
rows = []
for model, seed, shape, axes in [('sdxl',1005,(1,4,128,128),'NCHW'),('anima',1006,(1,16,1,128,128),'NCTHW')]:
    generator = torch.Generator(device='cpu').manual_seed(seed)
    before = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    noise = torch.randn(shape, dtype=torch.float32, device='cpu', generator=generator)
    assert not torch.cuda.is_initialized()
    data = memoryview(noise.numpy()).cast('B')
    name = f'{model}-{seed}-f32le.bin'
    (DEST/name).write_bytes(data)
    rows.append({'model':model,'seed':seed,'shape':list(shape),'axes':axes,'dtype':'float32','byteorder':'little','file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'generator_state_before_sha256':before,'generator_state_after_sha256':hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()})
(DEST/'noise-manifest.json').write_text(json.dumps({'purpose':'private shared raw-noise quality diagnostic; not weights','generator':'torch.Generator(cpu).manual_seed(payload_seed); torch.randn(F32,contiguous)','torch_version':str(torch.__version__),'torch_git_version':torch.version.git_version,'rows':rows},indent=2)+'\n')
