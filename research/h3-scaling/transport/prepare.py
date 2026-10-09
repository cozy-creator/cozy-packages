"""Copy the existing exact-four-GPU H3 capture and add only transport warmup."""
from pathlib import Path
import argparse,ast,shutil,tomllib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-package',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();source=a.base_package.resolve();destination=a.campaign.resolve()/'native-transport-control'
    project=tomllib.loads((source/'pyproject.toml').read_text())
    if project['project']['entry-points']['cozy.application']['default']!='h3:app':
        raise ValueError('Expected the reviewed H3 application export')
    tree=ast.parse((source/'h3.py').read_text())
    if any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='warmup' for n in tree.body):
        raise ValueError('This H3 source already has a warmup; review composition instead of replacing it')
    destination.mkdir(parents=True,exist_ok=True)
    for mode in ('peer','host'):
        target=destination/f'package-{mode}'
        if target.exists():raise FileExistsError(f'Preserve existing capture: {target}')
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('.venv','.git','__pycache__'))
        shutil.copy2(Path(__file__).with_name('transport_control.py'),target/'transport_control.py')
        h3=target/'h3.py'
        h3.write_text(h3.read_text()+f'\n\ndef warmup() -> None:\n    from transport_control import configure\n    configure({mode!r})\n')
        metadata=target/'pyproject.toml';text=metadata.read_text()
        text=text.replace(f'name = "{project["project"]["name"]}"',f'name = "h3-transport-{mode}"',1)
        marker='only-include = [\n'
        if marker not in text:raise ValueError('Expected the reviewed wheel include list')
        text=text.replace(marker,marker+'    "transport_control.py",\n',1);metadata.write_text(text)
        print(target)

if __name__=='__main__':main()
