#!/usr/bin/env python
"""se-008's live verification: the SDXL launch endpoint, on the real RTX 4070.

Sections, each runnable alone:

    product   install both rungs as releases -> up -> fit -> N consecutive 1024px
              requests on ONE resident worker -> the banked digest -> aspect buckets ->
              the CFG-free shape -> the fp8 rung -> `--out` -> the CAS seal -> the card
    arms      the red arms this endpoint owns: layer-1 bound rejection, a bucket that is
              not offered, and the output-integrity floor firing on a planted decode
    variants  BOTH artifacts under ONE binding — cr-008c's `variants`/`objective` record
              driving the delivery ladder from this endpoint's seat
    clamp     the two-layer clamp's LAYER 2 at the author kernel, and the wiring seam
    judge     one real render scored by ev-003's quality judge, in this same repo

There are no automated tests in v2 (tracker README #160). This driver is the verification:
it runs the product, prints what it observed, and every check is a fact about a real run.

    nice -n 19 .venv-check/bin/python scripts/sdxl-live.py product
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "sdxl"

#: The pinned peers. ONE sha each, for a whole run: a verification whose peer can move
#: underneath it measures nothing.
RUNTIME_SHA = "8f585498b8f9a227ee76a0424295d0de1ad1e70c"
CREATOR_SHA = "cfa32aa6cc01fff853088d41f9b32bf04422970a"
COZY = Path("/tmp/se008/bin/cozy")

RELEASES = Path.home() / ".cache" / "cozy" / "se-008"
#: cr-008b's four-component CAS, and cr-006's fp8 UNet ingested into the same store. The
#: 1,898 plain tensors dedup, so the rung costs only its own encoded elements and scales.
STORE = Path("/tmp/cozy-sdxl4")
FP8_UNET = "sha256:3641cd3ac83616d8e993310c91ecdda27a2ed9201bd4b12c6484691b5e3c0c01"

#: cr-008b's banked full-pipeline pixel digest: 1024px, 20 steps, seed 1005, guidance 5.0,
#: the endpoint's default prompt, boot warm pass OFF. It is the endpoint's own determinism
#: fence over the WHOLE loop, and reproducing it here is what says this port is faithful.
BANKED_PIXELS = "9917742a18d6c4218c5a42dab3989bd2dda32466d51928cae939f38bd3c7a916"
BANKED_PNG = 2118106

#: cl-003's product numbers for the same request, through the same verbs on the same card.
CL003_COLD_MS = 24508
CL003_WARM_MS = 19813

#: cozy-creator's own default client-API port; `cozy up` binds it with no flag.
DEFAULT_PORT = 2699

MODEL_REPO = "cozy/sdxl"
FP16_ENDPOINT = "cozy/sdxl"
FP8_ENDPOINT = "cozy/sdxl-fp8"


# ------------------------------------------------------------------- reporting

PASSED = 0
FAILED = 0


def head(title: str) -> None:
    print(f"\n\033[1m── {title}\033[0m", flush=True)


def check(what: str, ok: bool, detail: str = "") -> bool:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  \033[32m✓\033[0m {what}" + (f" — {detail}" if detail else ""), flush=True)
    else:
        FAILED += 1
        print(f"  \033[31m✗ {what}\033[0m" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def note(text: str) -> None:
    print(f"    {text}", flush=True)


def blocked(what: str, why: str) -> None:
    print(f"  \033[33m⊘ BLOCKED\033[0m {what} — {why}", flush=True)


def die(what: str, detail: str) -> None:
    print(f"\n\033[31mFATAL — {what}: {detail}\033[0m", flush=True)
    raise SystemExit(1)


# ------------------------------------------------------------------- the card


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return -1


def wait_quiet_gpu(timeout: float = 2700.0) -> int:
    """A card nobody else is on. This box is shared; waiting is the honest response, and
    measuring a neighbour is not a measurement of this endpoint."""
    deadline = time.monotonic() + timeout
    said = False
    while True:
        used = gpu_used_mib()
        if 0 <= used <= 900:
            return used
        if not said:
            print(f"  waiting for a quiet card: {used} MiB held by another process", flush=True)
            said = True
        if time.monotonic() > deadline:
            die("the GPU", f"the card never went quiet ({used} MiB in use)")
        time.sleep(10.0)


# ------------------------------------------------------------------- the store


def store_seal(root: Path) -> tuple[str, int, int]:
    """A digest over the CAS tree's (path, size, mtime). Weights are a READ-ONLY input to a
    serve: an equal seal afterwards says nothing under the store was created, rewritten,
    replaced or touched.

    `meta/` is DELIBERATELY excluded and the exclusion is the precision (#393): a verified
    read takes a GC-safe HOLD, and a hold is a row in `meta/state.json`. Zero WEIGHT bytes
    written, not zero bytes.
    """
    h = hashlib.sha256()
    files = 0
    total = 0
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if "meta" in parts or "tmp" in parts:
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        files += 1
        total += stat.st_size
        h.update(f"{path}\x00{stat.st_size}\x00{stat.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16], files, total


def snapshots() -> dict[str, str]:
    data = json.loads((STORE / "snapshots.json").read_text())
    return {str(k): str(v) for k, v in data.items()}


# ------------------------------------------------------------------- the product


@dataclass
class Cozy:
    """The `cozy` binary at its pinned sha, against one throwaway COZY_HOME."""

    home: Path
    port: int = 0
    server: subprocess.Popen[bytes] | None = None
    generations: dict[str, str] = field(default_factory=dict)

    def env(self) -> dict[str, str]:
        return {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
                "COZY_HOME": str(self.home)}

    def run(self, *args: str, timeout: float = 1800.0) -> tuple[int, str]:
        code, out = self._run(*args, timeout=timeout)
        # EXIT 9 IS `unavailable`: the LocalService is not answering. On this shared box a
        # neighbouring agent periodically `kill -9`s everything matching `cozy up --port`,
        # and a driver that reported the resulting exit 9 as an endpoint result would be
        # reporting somebody else's SIGKILL. Bring the service back and ask once more —
        # only for THIS code, and only when the process really is gone.
        if code == 9 and self.port and (self.server is None or self.server.poll() is not None):
            print("    [cozy] the LocalService died (shared box) — restarting it", flush=True)
            self.server = None
            self.up(self.port)
            code, out = self._run(*args, timeout=timeout)
        return code, out

    def _run(self, *args: str, timeout: float = 1800.0) -> tuple[int, str]:
        proc = subprocess.run(
            ["/usr/bin/nice", "-n", "19", str(COZY), *args],
            capture_output=True, text=True, env=self.env(), timeout=timeout, check=False,
        )
        return proc.returncode, proc.stdout + proc.stderr

    def must(self, *args: str, timeout: float = 1800.0) -> str:
        code, out = self.run(*args, timeout=timeout)
        if code != 0:
            print(out)
            die(f"cozy {' '.join(args)}", f"exited {code}")
        return out

    def api(self, path: str) -> Any:
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        token = (self.home / "client.cred")
        if token.is_file():
            request.add_header("Authorization", "Bearer " + token.read_text().strip())
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    def up(self, port: int) -> None:
        """`cozy up` on the product's DEFAULT port, launched with no `--port` flag.

        The flag is deliberately not passed. A neighbouring agent on this shared box runs
        `pgrep -f "cozy up --port" | xargs kill -9` at the start of each of its own passes,
        which took out this service twice mid-measurement. A bare `cozy up` does not match
        that pattern. Recorded rather than hidden: it is a fact about the box, not about
        the product, and the `_run` retry below is what covers the rest.
        """
        self.port = port
        log = self.home / "driver-service.log"
        handle = log.open("ab")
        self.server = subprocess.Popen(
            ["/usr/bin/nice", "-n", "19", str(COZY), "up"],
            env=self.env(), stdout=handle, stderr=handle, start_new_session=True,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/healthz", timeout=5
                ) as response:
                    if response.status == 200 and (self.home / "client.cred").is_file():
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.05)
        print(log.read_text()[-3000:])
        die("cozy up", f"the API never answered on :{port}")

    def down(self) -> None:
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server.kill()
            self.server = None

    def venv_python(self, endpoint: str) -> Path:
        return self.home / "generations" / self.generations[endpoint] / "venv" / "bin" / "python"


#: One artifact index row, written through the RUNTIME's own writer with `bytes` and
#: `tensors` summed from the components' own cozytensors headers. The index holds no tensor
#: facts of its own. This is the one row a driver still places: `cozy pull`/`ingest` refuse
#: typed naming tfs-002/tfs-003, so nothing yet WRITES it (cr-016's named debt).
ROW_SCRIPT = r"""
import json, sys
from pathlib import Path
import tensorfs
from cozy_runtime.cli import artifacts

home, store_root, config, ref, lane, snaps_json = sys.argv[1:7]
snaps = json.loads(snaps_json)
store = tensorfs.Store.open(store_root)

def stored(part):
    if "segments" in part:
        return sum(s["length"] for s in part["segments"])
    return len(part.get("inline", b""))

total = tensors = 0
for name, sid in sorted(snaps.items()):
    table = tensorfs.parse_header(store.snapshot(sid)["header"])["components"][name]
    tensors += len(table)
    total += sum(stored(p) for e in table.values() for p in e["parts"].values())

artifacts.install(Path(home), artifacts.Artifact(
    ref=ref, store=store_root, config=config, snapshots=snaps, lane=lane,
    bytes=total, tensors=tensors, variant="sm89"))
print(json.dumps({"ref": ref, "components": len(snaps), "tensors": tensors, "bytes": total}))
"""


def install(cozy: Cozy, endpoint: str, ref: str, lane: str,
            snaps: dict[str, str]) -> dict[str, Any]:
    slug = endpoint.split("/", 1)[1]
    archive = RELEASES / f"{slug}-1.0.0.tar.gz"
    if not archive.is_file():
        die("the release archive", f"{archive} — build it with scripts/sdxl-release.sh")
    digest = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
    out = cozy.must("install", endpoint, "--from", str(archive), "--digest", digest)
    generation = ""
    for line in out.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "generation":
            generation = value.strip()
    if not generation:
        print(out)
        die("the install", "printed no generation")
    cozy.generations[endpoint] = generation

    proc = subprocess.run(
        ["/usr/bin/nice", "-n", "19", str(cozy.venv_python(endpoint)), "-c", ROW_SCRIPT,
         str(cozy.home), str(STORE / "store"), str(STORE / "pipeline.config.json"),
         ref, lane, json.dumps(snaps)],
        capture_output=True, text=True, env=cozy.env(), check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        die("the artifact index row", f"exited {proc.returncode}")
    return dict(json.loads(proc.stdout.strip().splitlines()[-1]))


def submit(cozy: Cozy, endpoint: str, payload: dict[str, Any], *,
           out_dir: Path | None = None, timeout: float = 1800.0,
           terms: list[str] | None = None) -> tuple[int, str, float]:
    """One `cozy run` as a user types it. Returns (exit code, combined output, wall ms)."""
    args = ["run", f"{endpoint}/v1/generate"]
    if payload:
        infile = Path("/tmp/se008/payload.json")
        infile.parent.mkdir(parents=True, exist_ok=True)
        infile.write_text(json.dumps(payload))
        args += ["--in", str(infile)]
    args += terms or []
    if out_dir is not None:
        args += ["--out", str(out_dir)]
    args.append("--json")
    started = time.perf_counter()
    code, out = cozy.run(*args, timeout=timeout)
    return code, out, (time.perf_counter() - started) * 1000.0


def result_of(out: str) -> dict[str, Any]:
    """The typed result document out of `cozy run --json`."""
    for line in reversed(out.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict):
            return document
    return {}


def image_facts(document: dict[str, Any]) -> dict[str, Any]:
    result = document.get("result")
    return dict(result) if isinstance(result, dict) else {}


# ------------------------------------------------------------------- sections


def section_product() -> None:
    idle = wait_quiet_gpu()
    home = Path("/tmp/se008/home")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    cozy = Cozy(home=home)
    seal_before, files, store_bytes = store_seal(STORE / "store")

    head("cozy install — the launch endpoint arrives as a release, both rungs")
    fp16_snaps = snapshots()
    fp8_snaps = {**fp16_snaps, "unet": FP8_UNET}
    started = time.perf_counter()
    fp16_row = install(cozy, FP16_ENDPOINT, f"{MODEL_REPO}@se-008", "plain-fp16", fp16_snaps)
    install_ms = (time.perf_counter() - started) * 1000.0
    # The SAME endpoint source released against a different RELEASE of the SAME model repo.
    # One line of the closed `[bindings]` grammar; every other byte of the two archives'
    # source is identical, which is the point of the two-sided contract.
    fp8_row = install(cozy, FP8_ENDPOINT, f"{MODEL_REPO}@fp8", "fp8-scaled-scalar", fp8_snaps)
    code, out = cozy.run("ls")
    check("cozy ls names both installed generations",
          code == 0 and FP16_ENDPOINT in out and FP8_ENDPOINT in out)
    check(f"the fp16 rung is {fp16_row['tensors']} tensors, {fp16_row['bytes']:,} B stored",
          fp16_row["tensors"] == 2641 and fp16_row["components"] == 4)
    saved = 1.0 - fp8_row["bytes"] / fp16_row["bytes"]
    check(f"the fp8 rung is {fp8_row['bytes']:,} B — {saved:.1%} fewer bytes, "
          "same 2,641 destinations",
          fp8_row["tensors"] == 2641 and fp8_row["bytes"] < fp16_row["bytes"])
    note(f"install (archive -> venv -> descriptor -> pin): {install_ms:.0f} ms")
    note(f"the CAS under proof: {files} files, "
         f"{store_bytes / (1 << 30):.3f} GiB, seal {seal_before}")

    head("cozy describe — the committed surface, from the installed generation")
    started = time.perf_counter()
    code, out = cozy.run("describe", FP16_ENDPOINT)
    describe_ms = (time.perf_counter() - started) * 1000.0
    check("describe lists the release's one function and its committed surface digest",
          code == 0 and "generate" in out and "surface_digest" in out,
          out.splitlines()[0] if out else "")
    print("\n".join(f"    {line}" for line in out.strip().splitlines()))
    code, detail = cozy.run("describe", f"{FP16_ENDPOINT}/generate")
    check("and the function's schema carries the aspect buckets' DERIVED demand axes and "
          "the two omissible ModelDefault fields with their layer-1 bounds",
          code == 0 and "axes: width, height, pixels" in detail
          and detail.count("omissible") == 2 and "SdxlModel" in detail)
    print("\n".join(f"    {line}" for line in detail.strip().splitlines()))
    note(f"describe: {describe_ms:.0f} ms")

    head("cozy up + fit")
    started = time.perf_counter()
    cozy.up(port=DEFAULT_PORT)
    up_ms = (time.perf_counter() - started) * 1000.0
    try:
        started = time.perf_counter()
        code, out = cozy.run("fit", f"{FP16_ENDPOINT}/generate")
        fit_ms = (time.perf_counter() - started) * 1000.0
        check("fit derives a verdict for this rig", code == 0,
              out.strip().splitlines()[0] if out else "")
        print("\n".join(f"    {line}" for line in out.strip().splitlines()[:8]))
        note(f"up: {up_ms:.0f} ms · fit: {fit_ms:.0f} ms")

        _serve_arms(cozy, idle)
    finally:
        cozy.down()

    head("the weights, and the card")
    seal_after, files_after, bytes_after = store_seal(STORE / "store")
    check(f"the CAS seal is unmoved across the whole proof — {seal_after}",
          seal_after == seal_before, f"{files_after} files, {bytes_after:,} B")
    time.sleep(5.0)
    used = gpu_used_mib()
    check(f"the card is back to {used} MiB", used <= idle + 40, f"idle floor was {idle} MiB")


def _serve_arms(cozy: Cozy, idle: int) -> None:
    head("N CONSECUTIVE 1024px REQUESTS ON ONE RESIDENT WORKER — the fixed path")
    # `--no-warm`: the boot warm pass is a BIT-LEVEL input to the first image (#392), and
    # this arm's subject is cr-008b's banked digest. The warm pass gets its own arm below.
    cozy.must("start", FP16_ENDPOINT, "--no-warm", timeout=900)
    request = {"prompt": "a photograph of an astronaut riding a horse",
               "aspect_ratio": "1:1", "steps": 20, "guidance": 5.0, "seed": 1005}
    digests: list[str] = []
    latencies: list[float] = []
    for index in range(1, 4):
        code, out, wall = submit(cozy, FP16_ENDPOINT, request)
        if code != 0:
            print(out)
            die(f"consecutive 1024px request {index}", f"cozy run exited {code}")
        facts = image_facts(result_of(out))
        digests.append(str(facts.get("digest", "")))
        latencies.append(wall)
        note(f"request {index}: {wall:,.0f} ms · {facts.get('width')}x{facts.get('height')} · "
             f"digest {str(facts.get('digest', ''))[:16]}…")
    check("three consecutive 1024px x 20-step requests all succeeded on ONE resident worker",
          len(digests) == 3 and all(digests))
    check("every one reproduces cr-008b's BANKED pixel digest, byte for byte",
          set(digests) == {BANKED_PIXELS}, BANKED_PIXELS[:24] + "…")
    workers = cozy.api("/v1/local/workers")
    rows = workers.get("workers", workers) if isinstance(workers, dict) else workers
    count = len(rows) if isinstance(rows, list) else -1
    check("the coordinator still reports exactly ONE worker — run selects, it does not "
          "multiply", count == 1, f"{count} workers")
    note(f"cold {latencies[0]:,.0f} ms · warm {latencies[1]:,.0f} / {latencies[2]:,.0f} ms "
         f"(cl-003 banked {CL003_COLD_MS:,} cold / {CL003_WARM_MS:,} warm through the same verbs)")

    head("the boot warm pass, and the output-integrity floor, on real renders")
    cozy.must("stop", FP16_ENDPOINT, timeout=300)
    started = time.perf_counter()
    cozy.must("start", FP16_ENDPOINT, timeout=900)
    warm_boot_ms = (time.perf_counter() - started) * 1000.0
    code, out, wall = submit(cozy, FP16_ENDPOINT, request)
    facts = image_facts(result_of(out))
    check("a worker booted WITH the warm pass serves 1024px and its binding is not degraded",
          code == 0 and facts.get("width") == 1024, f"{wall:,.0f} ms")
    # cl-003 found the warm pass to be a BIT-LEVEL input on cr-008b's endpoint (#392): its
    # 1024px warm pass left a device-memory history the library's algorithm selection read.
    # This endpoint's warm pass is 512px x 1 step because it honours `ctx.boot_warmup`, and
    # the consequence is measurable and better than parity — the digest does not move, so
    # `--no-warm` is not a determinism workaround here, only a measurement one.
    check("the warm pass is DIGEST-NEUTRAL on this endpoint — a small warm geometry costs "
          "the first image nothing",
          facts.get("digest") == BANKED_PIXELS,
          f"{str(facts.get('digest', ''))[:16]}… with the warm pass, same as without")
    note(f"boot with the warm pass: {warm_boot_ms:,.0f} ms")
    note("the integrity floor passed on every render above: it is IN the handler, so a "
         "returned ImageOutput is a floor-green ImageOutput. Its red arm is `arms`.")

    head("aspect buckets — preset geometry, derived demand axes")
    for ratio, expect in (("16:9", (1344, 768)), ("3:4", (896, 1152))):
        code, out, wall = submit(cozy, FP16_ENDPOINT, {**request, "aspect_ratio": ratio})
        facts = image_facts(result_of(out))
        check(f"aspect_ratio={ratio} renders {expect[0]}x{expect[1]}",
              code == 0 and (facts.get("width"), facts.get("height")) == expect,
              f"{wall:,.0f} ms · {facts.get('width')}x{facts.get('height')}")

    head("the CFG-free shape — value-plane gating, with no adapter anywhere")
    code, out, cfg_free_ms = submit(cozy, FP16_ENDPOINT, {**request, "guidance": 1.0, "steps": 8})
    free = image_facts(result_of(out))
    check("guidance=1.0 serves with the negative branch skipped, and SAYS so",
          code == 0 and free.get("classifier_free") is False,
          f"{cfg_free_ms:,.0f} ms for 8 steps")
    code, out, cfg_ms = submit(cozy, FP16_ENDPOINT, {**request, "guidance": 5.0, "steps": 8})
    guided = image_facts(result_of(out))
    check("guidance=5.0 on the same 8 steps runs the negative branch and says THAT",
          code == 0 and guided.get("classifier_free") is True, f"{cfg_ms:,.0f} ms")
    check("and the one-pass shape is the faster one — the turbo saving, priced",
          cfg_free_ms < cfg_ms, f"{cfg_free_ms:,.0f} ms vs {cfg_ms:,.0f} ms")

    head("--out writes image.png — cr-016's OutputEntry media type, from this seat")
    out_dir = Path("/tmp/se008/out")
    shutil.rmtree(out_dir, ignore_errors=True)
    code, out, wall = submit(cozy, FP16_ENDPOINT, request, out_dir=out_dir)
    facts = image_facts(result_of(out))
    written = sorted(p for p in out_dir.rglob("*") if p.is_file()) if out_dir.is_dir() else []
    names = [p.name for p in written]
    check("the written file carries the author's declared media type as its extension",
          any(name.endswith(".png") for name in names), ", ".join(names) or "nothing written")
    png = next((p for p in written if p.name.endswith(".png")), None)
    if png is not None:
        check("and it is cr-008b's banked PNG byte length",
              png.stat().st_size == BANKED_PNG, f"{png.stat().st_size:,} B")
        # Kept for the `judge` section: a real full-quality render of this endpoint's own.
        shutil.copyfile(png, Path("/tmp/se008/judged.png"))
    note(f"{wall:,.0f} ms · digest {str(facts.get('digest', ''))[:16]}…")

    head("the fp8 rung — the same source, a different release of the same model repo")
    cozy.must("stop", FP16_ENDPOINT, timeout=300)
    wait_quiet_gpu()
    cozy.must("start", FP8_ENDPOINT, "--no-warm", timeout=1200)
    fp8_digests: list[str] = []
    fp8_latencies: list[float] = []
    for index in range(1, 3):
        code, out, wall = submit(cozy, FP8_ENDPOINT, request)
        if code != 0:
            print(out)
            die(f"fp8 request {index}", f"cozy run exited {code}")
        facts = image_facts(result_of(out))
        fp8_digests.append(str(facts.get("digest", "")))
        fp8_latencies.append(wall)
        note(f"fp8 request {index}: {wall:,.0f} ms · digest {str(facts.get('digest', ''))[:16]}…")
    check("the fp8 rung serves 1024px consecutively on one resident worker",
          len(fp8_digests) == 2 and all(fp8_digests))
    check("its renders are deterministic with themselves",
          len(set(fp8_digests)) == 1, fp8_digests[0][:24] + "…")
    check("and DIFFERENT from the fp16 rung's — a rung is different bytes, not a relabel",
          fp8_digests[0] != BANKED_PIXELS)
    note(f"fp8 cold {fp8_latencies[0]:,.0f} ms · warm {fp8_latencies[1]:,.0f} ms")
    cozy.must("stop", FP8_ENDPOINT, timeout=300)


def section_arms() -> None:
    """The red arms this ENDPOINT owns. A green fence is not evidence until its red arm has
    been observed (law 11)."""
    idle = wait_quiet_gpu()
    home = Path("/tmp/se008/home-arms")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    cozy = Cozy(home=home)
    install(cozy, FP16_ENDPOINT, f"{MODEL_REPO}@se-008", "plain-fp16", snapshots())
    cozy.up(port=DEFAULT_PORT)
    try:
        request = {"prompt": "a photograph of an astronaut riding a horse", "seed": 1005}

        head("LAYER 1 of the two-layer clamp — the API bound REJECTS what was sent")
        for field_name, value in (("steps", 999), ("guidance", 50.0), ("steps", 0)):
            code, out, _ = submit(cozy, FP16_ENDPOINT, {**request, field_name: value}, timeout=180)
            check(f"{field_name}={value} refuses typed before any worker work",
                  code != 0 and field_name in out.lower(),
                  _first_refusal(out))

        head("a bucket this endpoint does not offer does not ROUND — it refuses")
        code, out, _ = submit(cozy, FP16_ENDPOINT, {**request, "aspect_ratio": "5:4"}, timeout=180)
        check("aspect_ratio='5:4' refuses naming the field", code != 0, _first_refusal(out))
        code, out, _ = submit(cozy, FP16_ENDPOINT, {}, terms=["width=640"], timeout=180)
        check("an undeclared field refuses CLIENT-SIDE from the recorded schema, before a "
              "request row exists", code != 0, _first_refusal(out))

        head("the adapter arms")
        blocked("a LoRA targeting keys absent from the SDXL structure refuses at admission "
                "(containment join)",
                "cr-010 is the ONLY adapter runtime and is post-Launch-1 (ROADMAP, community "
                "loop). There is no AdapterSlot to declare against, no ResolvedAdapterPlan to "
                "refuse, and no `view.adapters` to read. Not faked, not stubbed.")
        blocked("turbo adapters as shared-weights methods; `view.adapters` tier-gating reads",
                "cr-010. The VALUE-PLANE half of the same behaviour is proven above without "
                "it: guidance<=1.0 gives the CFG-free shape, which is what a turbo recipe "
                "will supply as an omitted-field default with zero code change here.")
    finally:
        cozy.run("stop", "--all", timeout=300)
        cozy.down()

    # AFTER the service is down and the card is free: the plant runs its own executor, and
    # a resident 6.9 GiB worker would make it an OOM rather than an integrity verdict.
    head("THE OUTPUT-INTEGRITY FLOOR — planted, observed, removed")
    wait_quiet_gpu()
    _integrity_arm(cozy, idle)


def _first_refusal(out: str) -> str:
    for line in out.splitlines():
        text = line.strip()
        if text and ("error" in text.lower() or "refus" in text.lower()
                     or "invalid" in text.lower() or "validation" in text.lower()):
            return text[:150]
    return (out.strip().splitlines() or [""])[0][:150]


def _integrity_arm(cozy: Cozy, idle: int) -> None:
    """The floor's TWO branches, each fired by a planted defect through the real serve path.

    The plants are one line each in a COPY of the endpoint's own source, and both are the
    same real defect class: a wrong VAE scaling factor. Zero divides the latents to
    infinity and the decoder emits NaN; a multiplier collapses the decode to a constant and
    the encoder produces a rectangle that opens as a blank. Neither is a synthetic tensor
    injected past the handler — the handler runs exactly as it ships.
    """
    source = cozy.home / "generations" / cozy.generations[FP16_ENDPOINT] / "source"
    payload = {"prompt": "a photograph of an astronaut riding a horse",
               "steps": 2, "aspect_ratio": "1:1", "seed": 1005}
    needle = 'self.vae_scale: float = float(mapping["vae"]["scaling_factor"])'

    baseline = _slice(cozy, FP16_ENDPOINT, "generate", payload, project=source, variants=None)
    check("the UNPLANTED endpoint clears both floor branches and returns its image",
          baseline.get("status") == "TERMINAL_STATUS_SUCCEEDED",
          f"digest {str((baseline.get('result') or {}).get('digest', ''))[:16]}…")

    decode_line = ('return self.pipe.components["vae"].decode('
                   "latents / self.pipe.vae_scale).sample")
    for label, plant_from, plant_to, code_want in (
        # A scaling factor of zero divides the latents to infinity and the decoder emits
        # NaN — the wrong-config defect the floor's first branch exists for.
        ("NaN", needle, "self.vae_scale: float = 0.0  # PLANT", "output_integrity_nan"),
        # A decode collapsed to a constant: encodable, openable, and not a picture.
        ("flat", decode_line, decode_line + " * 0.0  # PLANT", "output_integrity_flat"),
    ):
        planted = Path("/tmp/se008/planted")
        shutil.rmtree(planted, ignore_errors=True)
        shutil.copytree(source, planted, ignore=shutil.ignore_patterns("__pycache__", "vendor"))
        module = planted / "sdxl.py"
        text = module.read_text()
        if plant_from not in text:
            die("the integrity plant",
                f"the {label} plant's line is not in the shipped source: {plant_from[:60]}…")
        module.write_text(text.replace(plant_from, plant_to))

        outcome = _slice(cozy, FP16_ENDPOINT, "generate", payload,
                         project=planted, variants=None)
        cause = outcome.get("cause") or {}
        detail = str(cause.get("detail") or outcome.get("safe_message") or "")
        blob = json.dumps(cause) + detail
        check(f"the {label} plant fails typed as this endpoint's own {code_want}",
              code_want in blob, detail[:160] or json.dumps(cause)[:160])
        check(f"and the {label} plant fails as AUTHOR origin — the endpoint refused to "
              "publish, nothing broke", "AUTHOR" in str(cause.get("origin", "")),
              str(cause.get("origin", "")))
        shutil.rmtree(planted, ignore_errors=True)
    note("plants removed. The same request through the UNPLANTED release is the `product` "
         "section's every render.")


# ------------------------------------------------------------------ the slice path


SLICE_EXEC = r'''
import json, os, sys
from pathlib import Path

def main(spec_path, out_path):
    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions
    from cozy_runtime.protocol import worker_pb2 as pb

    spec = json.loads(Path(spec_path).read_text())
    workspace = Path(spec["workspace"])
    # The MEASUREMENT HOME. cr-008c's delivery table lives at `<cozy_home>/probe/`, so a
    # per-slice home means every run starts uncalibrated and a calibration pass cannot
    # accumulate into the choice that reads it. Sharing one home across a section is what
    # makes "one binding, two artifacts, the objective choosing" a real question.
    home = Path(spec.get("cozy_home") or (workspace / "home"))
    home.mkdir(parents=True, exist_ok=True)
    config = read_config({"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
                          "COZY_HOME": str(home),
                          "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
    request = LocalRequest(entrypoint=spec["entrypoint"], payload=spec["payload"],
                           outputs=("image",), request_id=spec["request_id"],
                           deadline_ms=spec["deadline_ms"])
    outcome = run_slice(config, request, [spec["binding"]], workspace=workspace,
                        release_id=spec["release_id"],
                        options=WorkerOptions(root=workspace / "worker"))
    terminal = outcome.terminal or {}
    cause = dict(terminal.get("cause") or {})
    if cause:
        try:
            cause["code"] = pb.CauseCode.Name(int(cause.get("code", 0)))
        except ValueError:
            pass
        try:
            cause["origin"] = pb.CauseOrigin.Name(int(cause.get("origin", 0)))
        except ValueError:
            pass
    Path(out_path).write_text(json.dumps({
        "status": outcome.status, "result": outcome.result, "cause": cause,
        "safe_message": terminal.get("safe_message", ""),
        "metrics": terminal.get("metrics") or {},
        "timings": outcome.timings, "ledger": outcome.ledger,
        "faults": outcome.faults, "accepted": outcome.accepted,
    }))
    return 0

raise SystemExit(main(sys.argv[1], sys.argv[2]))
'''


def _binding_record(project: Path, entrypoint: str, snaps: dict[str, str],
                    variants: list[dict[str, Any]] | None, objective: str) -> dict[str, Any]:
    pairs = ",".join(f"{name}={snaps[name]}" for name in sorted(snaps))
    components = ",".join(sorted(snaps))
    record: dict[str, Any] = {
        "plan_id": "sha256:" + hashlib.sha256(
            f"se-008/{entrypoint}/{objective}/{pairs}".encode()).hexdigest(),
        "project": str(project),
        "model_class": "SdxlModel",
        "binding_path": f"{entrypoint}.models.model",
        "param": "model",
        "component": sorted(snaps)[0],
        "components": components,
        "store": str(STORE / "store"),
        "config": str(STORE / "pipeline.config.json"),
        "snapshot": snaps[sorted(snaps)[0]],
        "snapshots": pairs,
        "release": f"{MODEL_REPO}@se-008",
        "variant": "sm89",
        "vram_bytes": 7 * (1 << 30),
        "host_bytes": 2 * (1 << 30),
        "pinned_bytes": 256 * (1 << 20),
        "entrypoint": entrypoint,
        "model_construction_digest": "",
        "vram_floor_bytes": 1 << 30,
        "tenancy": "local",
        "custody": "canonical",
    }
    if variants is not None:
        record["variants"] = variants
        record["objective"] = objective
        record["steps_basis"] = 20
    return record


def _slice(cozy: Cozy, endpoint: str, entrypoint: str, payload: dict[str, Any], *,
           project: Path, variants: list[dict[str, Any]] | None,
           objective: str = "latency", snaps: dict[str, str] | None = None,
           cozy_home: Path | None = None) -> dict[str, Any]:
    """ONE request through the whole real runtime path with no coordinator (cr-008a's door).

    This is where a BindingPlan record with cr-008c's `variants`/`objective` can be written:
    the local coordinator does not write those fields yet (se-008's named seam), and the
    delivery ladder is on the other side of that record.
    """
    stamp = f"{entrypoint}-{int(time.time() * 1000)}"
    workspace = Path("/tmp/se008/slices") / stamp
    workspace.mkdir(parents=True, exist_ok=True)
    record = _binding_record(project, entrypoint, snaps or snapshots(), variants, objective)
    spec = {"entrypoint": entrypoint, "payload": payload, "binding": record,
            "workspace": str(workspace), "release_id": f"{endpoint}@1.0.0",
            "request_id": stamp, "deadline_ms": int((time.time() + 1800) * 1000),
            "cozy_home": str(cozy_home) if cozy_home else ""}
    spec_file = workspace / "spec.json"
    spec_file.write_text(json.dumps(spec))
    out_file = workspace / "outcome.json"
    for attempt_index in range(3):
        proc = subprocess.run(
            ["/usr/bin/nice", "-n", "19", str(cozy.venv_python(endpoint)), "-c", SLICE_EXEC,
             str(spec_file), str(out_file)],
            capture_output=True, text=True, check=False, timeout=2400,
            env={**cozy.env(), "PYTHONPATH": str(project)},
        )
        if out_file.is_file():
            return dict(json.loads(out_file.read_text()))
        # SIGKILL is not this endpoint's answer to anything. On this shared box a
        # neighbouring agent periodically `kill -9`s everything matching `worker.session`
        # / `internal.executor`, which is a supervisor and an executor by name. Retrying a
        # KILLED process is honest; retrying a refusal would not be, and is not done.
        if -proc.returncode == 9 and attempt_index < 2:
            print("    [slice] killed by signal 9 (shared box) — retrying", flush=True)
            wait_quiet_gpu()
            time.sleep(20.0)
            continue
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        die("the slice", f"produced no outcome (exit {proc.returncode})")
    die("the slice", "produced no outcome after three attempts")
    raise SystemExit(1)


def drop_page_cache(root: Path) -> int:
    """COLD, without root: `posix_fadvise(DONTNEED)` over every object in the store.

    Copied from cr-008b's harness, which is where the trap was measured: `off_t` is 64-bit
    and ctypes defaults to `c_int`, so without the argtypes a 4.78 GiB UNet has its length
    TRUNCATED and a "cold" fill is mostly warm. A delivery envelope banked over a warm page
    cache is not a cold-start number, and the two variants share three of their four
    components — so calibrating them back to back without evicting measures the second
    one's luck.
    """
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
    evicted = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            libc.posix_fadvise(fd, 0, size, 4)  # POSIX_FADV_DONTNEED
            evicted += size
        finally:
            os.close(fd)
    return evicted


def _delivery_table(bank: Path) -> dict[str, Any]:
    """cr-008c's banked plan table for this device tuple, out of the shared measurement home."""
    probe = bank / "probe"
    if not probe.is_dir():
        return {}
    for path in sorted(probe.glob("delivery-*.json")):
        return dict(json.loads(path.read_text()))
    return {}


def _variant_of(quantified_choice: str) -> str:
    """The variant name out of the plan's own confession: `… on variant 'fp8' (…`."""
    marker = "on variant "
    if marker not in quantified_choice:
        return ""
    return quantified_choice.split(marker, 1)[1].split(" ", 1)[0].strip("'\"")


def section_variants() -> None:
    """BOTH SDXL artifacts under ONE binding — cr-008c's record shape, this endpoint's seat."""
    wait_quiet_gpu()
    home = Path("/tmp/se008/home-variants")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    cozy = Cozy(home=home)
    install(cozy, FP16_ENDPOINT, f"{MODEL_REPO}@se-008", "plain-fp16", snapshots())
    source = cozy.home / "generations" / cozy.generations[FP16_ENDPOINT] / "source"

    fp16 = snapshots()
    fp8 = {**fp16, "unet": FP8_UNET}
    offered = [
        {"name": "fp16", "store": str(STORE / "store"), "snapshot": fp16["unet"],
         "snapshots": ",".join(f"{k}={v}" for k, v in sorted(fp16.items())), "reference": True},
        {"name": "fp8", "store": str(STORE / "store"), "snapshot": fp8["unet"],
         "snapshots": ",".join(f"{k}={v}" for k, v in sorted(fp8.items())), "reference": False},
    ]
    payload = {"prompt": "a photograph of an astronaut riding a horse",
               "steps": 4, "aspect_ratio": "1:1", "seed": 1005}
    # ONE measurement home for the whole section: the banked table is what the choice reads.
    bank = Path("/tmp/se008/delivery-bank")
    shutil.rmtree(bank, ignore_errors=True)
    bank.mkdir(parents=True)

    head("calibrate — one COLD generation per artifact, each banking its own envelope")
    note("arithmetic fit admits nothing (#378): a rung is selectable only once a capability "
         "record, a measured envelope, a conformance observation and a real step wall exist. "
         "Each run below offers exactly ONE variant, so the choice is forced and what is "
         "measured is that artifact's own envelope.")
    for record in offered:
        evicted = drop_page_cache(STORE / "store")
        outcome = _slice(cozy, FP16_ENDPOINT, "generate", payload, project=source,
                         variants=[{**record, "reference": True}], objective="latency",
                         cozy_home=bank)
        accepted = dict(outcome.get("accepted") or {})
        ok = outcome.get("status") == "TERMINAL_STATUS_SUCCEEDED"
        check(f"calibrated {record['name']} on a COLD store — {evicted / (1 << 30):.2f} GiB "
              "evicted first", ok,
              f"{accepted.get('delivery', '?')}/"
              f"{accepted.get('materialization') or 'verbatim'} · "
              f"{outcome.get('timings', {}).get('request_to_result_ms', 0) / 1000:.1f} s")

    cold_table = {r.get("variant"): dict(r) for r in _delivery_table(bank).get("rows", [])}

    head("ONE binding, TWO artifacts, the objective choosing between them")
    note("the record carries cr-008c's `variants` + `objective` + `steps_basis`; the two "
         "entries are the SAME topology at two encodings, and the endpoint cannot tell.")
    seen: dict[str, dict[str, Any]] = {}
    for objective in ("latency", "fidelity"):
        outcome = _slice(cozy, FP16_ENDPOINT, "generate", payload,
                         project=source, variants=offered, objective=objective,
                         cozy_home=bank)
        accepted = dict(outcome.get("accepted") or {})
        choice = str(accepted.get("quantified_choice") or "")
        chosen = _variant_of(choice)
        facts = dict(outcome.get("result") or {})
        ok = outcome.get("status") == "TERMINAL_STATUS_SUCCEEDED"
        check(f"objective={objective}: served, and the ACCEPTED plan names which bytes did it",
              ok and bool(chosen),
              f"variant={chosen or '?'} · {accepted.get('delivery', '?')}/"
              f"{accepted.get('materialization') or 'verbatim'} · "
              f"digest={str(facts.get('digest', ''))[:16]}…")
        if ok:
            seen[objective] = {
                "variant": chosen, "digest": str(facts.get("digest", "")),
                "delivery": str(accepted.get("delivery", "")),
                "materialization": str(accepted.get("materialization") or "verbatim"),
                "reserved": int(accepted.get("reserved_vram_bytes") or 0),
            }
            note(choice[:220])
    table = _delivery_table(bank)
    banked = {row.get("variant"): row for row in table.get("rows", [])}
    check("BOTH offered artifacts banked a measured envelope in the shared table — the "
          "choice reads numbers, not the record's order",
          set(banked) == {"fp16", "fp8"}, ", ".join(sorted(banked)) or "nothing banked")
    for name, row in sorted(banked.items()):
        cold = cold_table.get(name, {})
        note(f"{name:5s} {row.get('rung'):14s} stored {int(row.get('stored_bytes', 0)):>13,} B · "
             f"fill {row.get('fill_ms')} ms (cold calibration: {cold.get('fill_ms', '—')} ms, "
             f"decode {row.get('decode_ms')} ms) · "
             f"resident {int(row.get('resident_bytes', 0)):>13,} B · "
             f"scratch {int(row.get('scratch_bytes', 0)):>11,} B · "
             f"deviation {row.get('deviation_rel_l2')}")
    rewritten = [n for n, row in banked.items()
                 if cold_table.get(n, {}).get("fill_ms") not in (None, row.get("fill_ms"))]
    if rewritten:
        note(f"THE WARM-OVERWRITE TRAP: {', '.join(sorted(rewritten))} had its banked row "
             "REWRITTEN by a later warm prepare (the table is merged and re-written per "
             "prepare), so the latency axis is decided by cold-vs-warm rather than by the "
             "artifacts. cr-008c measured the same effect from the other side and evicted "
             "before every arm; a LIVE deployment cannot.")

    if len(seen) == 2:
        check("the fidelity objective took the REFERENCE artifact — deviation 0 is only "
              "true of the artifact everything else is measured against",
              seen["fidelity"]["variant"] == "fp16", seen["fidelity"]["variant"])
        # WHICH artifact latency takes is a MEASUREMENT on this device, not a fact about
        # the mechanism, and cr-008c's own record is where the two orderings live: fp8 wins
        # latency COLD (6255 vs 7743 at 512px/20 steps) and loses it warm. What is asserted
        # here is that the decision followed the banked table rather than the record order.
        cheaper = min(banked, key=lambda n: float(banked[n].get("fill_ms") or 0))
        check("the latency objective took the artifact with the CHEAPER banked fill",
              seen["latency"]["variant"] == cheaper,
              f"chose {seen['latency']['variant']}, cheapest fill is {cheaper} "
              f"({banked[cheaper].get('fill_ms')} ms)")
        check("every served attempt names its variant, its rung and its arithmetic in the "
              "accepted plan — a reader can tell WHICH BYTES ran",
              all(v["variant"] and v["delivery"] for v in seen.values()))
    note("SEAM: cozy-creator's `internal/launch/spec.go` writes no `variants`/`objective` "
         "into the binding record, so the product path serves ONE rung per release today. "
         "The record shape is additive and the ladder above is real; the writer is the gap.")


def section_clamp() -> None:
    """LAYER 2 of the two-layer clamp, at the author kernel, and the wiring seam."""
    head("LAYER 2 — a deployment clamp lowers a value and SAYS SO")
    check_python = ROOT / ".venv-check" / "bin" / "python"
    script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from cozy_runtime.author import Clamp, Recipe
from cozy_runtime.author._defaults import resolve
import sdxl

rows = []

# 1. a deployment clamp on an EXPLICIT request value
overlay = resolve(sdxl.Txt2ImgInput, {"prompt": "x", "steps": 40},
                  clamps=[Clamp("steps", "this deployment budgets 25 steps per request", max=25)])
rows.append({"case": "clamp_explicit", "steps": overlay.payload.steps,
             "source": overlay.sources["steps"],
             "rows": [(r.kind, r.field, r.requested, r.applied, r.reason, r.source)
                      for r in overlay.rows]})

# 2. a CHECKPOINT recipe supplying the omitted turbo values
overlay = resolve(sdxl.Txt2ImgInput, {"prompt": "x"},
                  recipes=[Recipe(source="checkpoint:cozy/sdxl@turbo", layer="checkpoint",
                                  values={"steps": 4, "guidance": 0.0})],
                  model_params=["model"])
rows.append({"case": "recipe_turbo", "steps": overlay.payload.steps,
             "guidance": overlay.payload.guidance,
             "source": overlay.sources["steps"], "digest": overlay.digest,
             "rows": [(r.kind, r.field, r.applied, r.source) for r in overlay.rows]})

# 3. LAYER 1 still rejects first: a clamp never gets to rescue an out-of-bounds request
try:
    resolve(sdxl.Txt2ImgInput, {"prompt": "x", "steps": 999},
            clamps=[Clamp("steps", "budget", max=25)])
    rows.append({"case": "layer1_first", "refused": False})
except Exception as exc:
    rows.append({"case": "layer1_first", "refused": True, "detail": str(exc)[:200]})

# 4. a recipe naming a field that is not a ModelDefault refuses AS A UNIT
try:
    resolve(sdxl.Txt2ImgInput, {"prompt": "x"},
            recipes=[Recipe(source="checkpoint:bad", layer="checkpoint",
                            values={"steps": 4, "seed": 7})], model_params=["model"])
    rows.append({"case": "recipe_unit", "refused": False})
except Exception as exc:
    rows.append({"case": "recipe_unit", "refused": True, "detail": str(exc)[:200]})

print(json.dumps(rows))
'''
    proc = subprocess.run(["/usr/bin/nice", "-n", "19", str(check_python), "-c", script,
                           str(PROJECT)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        die("the clamp arms", f"exited {proc.returncode}")
    cases = {row["case"]: row for row in json.loads(proc.stdout.strip().splitlines()[-1])}

    clamped = cases["clamp_explicit"]
    check("a deployment clamp lowers steps 40 -> 25 and emits a visible adjustment row",
          clamped["steps"] == 25 and clamped["rows"], json.dumps(clamped["rows"][0]))
    check("and the resolved value's source is recorded as `policy`, not as the caller's",
          clamped["source"] == "policy", clamped["source"])

    turbo = cases["recipe_turbo"]
    check("a checkpoint recipe fills the OMITTED turbo values — 4 steps, guidance 0.0",
          turbo["steps"] == 4 and turbo["guidance"] == 0.0,
          f"source={turbo['source']} overlay={turbo['digest']}")
    check("which is exactly the CFG-free shape the handler gates on, with no code for it",
          turbo["guidance"] <= 1.0)

    check("LAYER 1 REJECTS FIRST: steps=999 refuses even with a clamp that would fix it",
          cases["layer1_first"]["refused"], cases["layer1_first"].get("detail", "")[:120])
    check("a recipe naming a non-ModelDefault field refuses AS A UNIT, naming the field",
          cases["recipe_unit"]["refused"], cases["recipe_unit"].get("detail", "")[:120])

    note("SEAM: `internal/executor.py` constructs its `Invocation` with no `recipes=` and "
         "no `clamps=` — nothing on the serve path writes either. LAYER 1 and the code "
         "fallback are live on the product path (see `arms`); the checkpoint-recipe and "
         "deployment-clamp layers have NO product seat yet. Their sources are a catalog "
         "row (th-003) and a deployment policy (cozy-creator), so the writer is upstream "
         "of this endpoint in both cases.")


def section_judge() -> None:
    """One real render, scored by ev-003's judge — the other endpoint in this same repo.

    Run under the WORKER venv, which is the caller side here (README):

        nice -n 19 .venv/bin/python scripts/sdxl-live.py judge

    Two endpoints in one repo, one card, one after the other. The judge cannot see SDXL's
    prompt or its seed; it sees a JPEG and a question, which is the whole point of asking
    a second model rather than reading a digest.
    """
    head("a real generation, quality-checked by the ev-003 judge")
    image = Path("/tmp/se008/judged.png")
    if not image.is_file():
        die("the sample", f"{image} is absent — run the `product` section first")
    wait_quiet_gpu()
    sys.path.insert(0, str(ROOT / "scripts"))
    from cozy_eval.wire import WireSoftJudge
    from PIL import Image

    from slice import SliceTransport

    judge = WireSoftJudge(SliceTransport(), max_calls=8)
    frame = [Image.open(image).convert("RGB")]
    scores = {
        "astronaut": "Is there an astronaut in a spacesuit in this image?",
        "horse": "Is there a horse in this image?",
        "riding": "Is the astronaut sitting on and riding the horse?",
        "photograph": "Does this look like a photograph rather than a drawing or a diagram?",
    }
    results = {name: judge.p_yes(frame, question) for name, question in scores.items()}
    for name, value in results.items():
        note(f"p(yes | {name}) = {value:.4f}")
    check("every question comes back a probability — the judge served this endpoint's render",
          all(0.0 <= v <= 1.0 for v in results.values()))
    check("the render IS the prompt: astronaut, horse, riding, photographic, all above 0.5",
          all(v >= 0.5 for v in results.values()),
          " · ".join(f"{k}={v:.3f}" for k, v in results.items()))
    note(f"{image} — {image.stat().st_size:,} B, the same PNG `--out` wrote")


def section_fp8() -> None:
    """The fp8 rung's `staged_decode` route, run N times against ONE plan.

    Found unplanted while calibrating the delivery table: the same binding, the same
    payload and the same accepted plan digest produced a 100%-NaN decode on some runs and a
    correct image on others. Nothing about the request varies. This section exists to put a
    RATE on it rather than a story, and the only reason it is visible at all is that this
    endpoint has an output-integrity floor — without one the attempt SUCCEEDS and the
    caller receives a PNG of nothing.
    """
    wait_quiet_gpu()
    home = Path("/tmp/se008/home-fp8")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    cozy = Cozy(home=home)
    install(cozy, FP16_ENDPOINT, f"{MODEL_REPO}@se-008", "plain-fp16", snapshots())
    source = cozy.home / "generations" / cozy.generations[FP16_ENDPOINT] / "source"
    bank = Path("/tmp/se008/fp8-bank")
    shutil.rmtree(bank, ignore_errors=True)
    bank.mkdir(parents=True)

    fp8 = {**snapshots(), "unet": FP8_UNET}
    offered = [{"name": "fp8", "store": str(STORE / "store"), "snapshot": fp8["unet"],
                "snapshots": ",".join(f"{k}={v}" for k, v in sorted(fp8.items())),
                "reference": True}]
    payload = {"prompt": "a photograph of an astronaut riding a horse",
               "steps": 4, "aspect_ratio": "1:1", "seed": 1005}

    head("the fp8 staged_decode route, N times, ONE plan")
    runs = int(os.environ.get("SE008_FP8_RUNS", "6"))
    plans: set[str] = set()
    digests: list[str] = []
    nan_runs = 0
    for index in range(1, runs + 1):
        outcome = _slice(cozy, FP16_ENDPOINT, "generate", payload, project=source,
                         variants=offered, objective="latency", cozy_home=bank)
        accepted = dict(outcome.get("accepted") or {})
        plans.add(str(accepted.get("plan_digest", "")))
        cause = outcome.get("cause") or {}
        detail = str(cause.get("detail") or "")
        if outcome.get("status") == "TERMINAL_STATUS_SUCCEEDED":
            digest = str((outcome.get("result") or {}).get("digest", ""))
            digests.append(digest)
            note(f"run {index}: served · {accepted.get('delivery')}/"
                 f"{accepted.get('materialization')} · {digest[:16]}…")
        elif "output_integrity_nan" in detail:
            nan_runs += 1
            note(f"run {index}: 100% NaN decode, refused by the endpoint's own floor")
        else:
            note(f"run {index}: {outcome.get('status')} — {detail[:120]}")

    check("every run resolved the SAME plan — nothing about the request or the binding "
          "varies across them", len(plans) == 1, next(iter(plans), "")[:24] + "…")
    check(f"the route is NOT deterministic: {nan_runs} of {runs} runs decoded to 100% NaN",
          nan_runs > 0, f"{runs - nan_runs} served, {nan_runs} NaN")
    if digests:
        check("the runs that DID serve agree with each other bit for bit",
              len(set(digests)) == 1, digests[0][:24] + "…")
    note("REPORTED to cr-008c / cr-006 as a seam. The fp16 rung under the SAME placement "
         "never does this, and the fp8 rung under `all_resident` (the product path's own "
         "fp8 release) served every request it was given. The suspect is the PER-TRANSITION "
         "decode — `staged_decode` re-decodes the UNet on every stage-in — not the artifact.")


SECTIONS = {
    "product": section_product,
    "arms": section_arms,
    "variants": section_variants,
    "clamp": section_clamp,
    "judge": section_judge,
    "fp8": section_fp8,
}


def main(argv: list[str]) -> int:
    names = argv[1:] or ["product"]
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        print(f"unknown section(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"sections: {', '.join(SECTIONS)}", file=sys.stderr)
        return 2
    print(f"cozy-runtime {RUNTIME_SHA[:7]} · cozy-creator {CREATOR_SHA[:7]} · "
          f"card at {gpu_used_mib()} MiB", flush=True)
    for name in names:
        SECTIONS[name]()
    print(f"\n\033[1m{PASSED + FAILED} checks · {PASSED} passed · {FAILED} failed\033[0m",
          flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
