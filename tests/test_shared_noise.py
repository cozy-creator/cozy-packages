import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'experiments/shared-noise/shared_noise.py'
spec = importlib.util.spec_from_file_location('shared_noise', HELPER)
noise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noise)


def test_canonical_artifacts_real_cpu_roundtrip_and_global_rng_unchanged():
    before = torch.get_rng_state().clone()
    manifest = json.loads((noise.ARTIFACTS/'noise-manifest.json').read_text())
    for row in manifest['rows']:
        value = noise.raw_noise(row['model'],row['seed'],tuple(row['shape']),device='cpu',dtype=torch.float32)
        expected = torch.randn(row['shape'], generator=torch.Generator(device='cpu').manual_seed(row['seed']),dtype=torch.float32)
        assert torch.equal(value,expected)
        assert hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()==row['sha256']
        assert value.is_contiguous()
    assert torch.equal(before,torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_explicit_anima_axis_mapping_and_sdxl_intended_rounding():
    five = noise.comfy_noise(torch.empty(1,16,1,128,128),1006)
    four = noise.comfy_noise(torch.empty(1,16,128,128),1006)
    assert torch.equal(five.squeeze(2),four)
    assert hashlib.sha256(five.numpy().tobytes()).digest()==hashlib.sha256(four.numpy().tobytes()).digest()
    source = noise.raw_noise('sdxl',1005,(1,4,128,128),device='cpu',dtype=torch.float32)
    rounded = noise.raw_noise('sdxl',1005,(1,4,128,128),device='cpu',dtype=torch.float16)
    assert torch.equal(rounded,source.half())
    assert not torch.equal(rounded.float(),source)  # first-denoiser equality is NOT promised
    for shape in [(1,16,128,1,128),(1,4,64,256)]:
        with pytest.raises(ValueError):noise.comfy_noise(torch.empty(shape),1006)
    with pytest.raises(ValueError):noise.comfy_noise(torch.empty(1,4,128,128),1005,[0])
    with pytest.raises(ValueError):noise.raw_noise('sdxl',1007,(1,4,128,128),device='cpu',dtype=torch.float32)


def test_actual_anima_prepare_latents_accepts_exact_shared_values_without_rng_draw():
    from diffusers.modular_pipelines.anima.before_denoise import AnimaPrepareLatentsStep
    value = noise.raw_noise('anima',1006,(1,16,1,128,128),device='cpu',dtype=torch.float32)
    generator=torch.Generator(device='cpu').manual_seed(1006);before=generator.get_state().clone()
    result=AnimaPrepareLatentsStep.prepare_latents(1,16,1024,1024,8,torch.float32,torch.device('cpu'),generator,value)
    assert torch.equal(result,value) and torch.equal(before,generator.get_state())


def test_comfy_function_body_changes_only_noise_provider():
    provenance=json.loads((ROOT/'experiments/shared-noise/SOURCE.json').read_text())
    original=Path(provenance['comfy_nodes_path']);assert hashlib.sha256(original.read_bytes()).hexdigest()==provenance['comfy_nodes_sha256']
    a=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='common_ksampler')
    b=next(n for n in ast.parse((ROOT/'experiments/shared-noise/comfy_node/__init__.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='common_ksampler')
    class Undo(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.parse('comfy.sample.prepare_noise',mode='eval').body if node.id=='comfy_noise'else node
    assert ast.dump(a)==ast.dump(Undo().visit(b))


def test_comfy_actual_prepare_noise_side_effect_is_explicitly_not_preserved():
    source=Path('/home/fidika/cozy/.worktrees/ComfyUI/memory-benchmark-20260929/comfy/sample.py')
    tree=ast.parse(source.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('prepare_noise_inner','prepare_noise')]
    namespace={'torch':torch};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);before=torch.get_rng_state().clone()
        expected=namespace['prepare_noise'](torch.empty(1,4,128,128),1005)
        after=torch.get_rng_state().clone();assert not torch.equal(before,after)
        supplied=noise.comfy_noise(torch.empty(1,4,128,128),1005)
        assert torch.equal(expected,supplied) and torch.equal(after,torch.get_rng_state())


@pytest.mark.parametrize('model', ['sdxl','anima'])
def test_frozen_assets_and_only_author_injection_delta(model):
    provenance=json.loads((ROOT/'experiments/shared-noise/SOURCE.json').read_text())['controls'][model]
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',provenance['commit'],provenance['path']],cwd=ROOT).decode().splitlines()
    for path in names:
        relative=Path(path).relative_to(provenance['path']);original=subprocess.check_output(['git','show',provenance['commit']+':'+path],cwd=ROOT)
        candidate=(ROOT/'experiments'/f'{model}-shared-noise'/relative).read_bytes()
        if str(relative)in ('pyproject.toml','uv.lock'):
            suffix='stable-vae'if model=='sdxl'else'stage-scopes'
            assert candidate==original.replace(f'cozy-experiment-{model}-{suffix}'.encode(),f'cozy-experiment-{model}-shared-noise'.encode());continue
        if str(relative)!=f'{model}/__init__.py':assert candidate==original;continue
        a=ast.parse(original);b=ast.parse(candidate)
        class Undo(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                return None if node.module=='shared_noise'else node
            def visit_FunctionDef(self,node):
                if node.name=='render_request':
                    node.args.args=[x for x in node.args.args if x.arg!='initial_noise'];node.args.defaults=[]
                return self.generic_visit(node)
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name)and node.func.id=='raw_noise'and model=='sdxl':
                    return ast.parse('torch.randn(1,4,height//8,width//8,generator=generator,device=device,dtype=torch.float16)',mode='eval').body
                node.keywords=[k for k in node.keywords if k.arg not in ('initial_noise',)and not(k.arg=='latents'and isinstance(k.value,ast.Name)and k.value.id=='initial_noise')]
                return self.generic_visit(node)
        assert ast.dump(a)==ast.dump(Undo().visit(b))


def test_corrupt_noise_fails_before_return(tmp_path,monkeypatch):
    data=(noise.ARTIFACTS/'noise-manifest.json').read_bytes();(tmp_path/'noise-manifest.json').write_bytes(data)
    row=json.loads(data)['rows'][0];(tmp_path/row['file']).write_bytes(b'corrupt')
    monkeypatch.setattr(noise,'ARTIFACTS',tmp_path)
    with pytest.raises(RuntimeError,match='payload changed'):
        noise.raw_noise('sdxl',1005,(1,4,128,128),device='cpu',dtype=torch.float32)
