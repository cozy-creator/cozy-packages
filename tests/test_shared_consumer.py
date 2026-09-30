import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "consumer", ROOT / "experiments/shared-noise/compare_observations.py"
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
manifest = json.loads(
    Path(
        "/home/fidika/.cozy/outputs/comfy-cozy-memory-20260929/analysis-shared-initial-noise/artifacts/noise-manifest.json"
    ).read_text()
)


def fixture(engine):
    counter = [0]
    allcopies = []

    def tensor(shape, dtype="torch.float16", device="cuda:0"):
        counter[0] += 1
        size = c.SIZES[dtype]
        for x in shape:
            size *= x
        row = {
            "copy_index": counter[0],
            "sha256": "a" * 64,
            "shape": shape,
            "stride": [1] * len(shape),
            "dtype": dtype,
            "logical_bytes": size,
            "storage_offset": 0,
            "device": device,
            "allocator_before": {} if device == "cpu" else {k: 0 for k in c.ALLOC},
            "allocator_after": {} if device == "cpu" else {k: 0 for k in c.ALLOC},
        }
        allcopies.append(row)
        return row

    tree = {"type": "dict", "items": {"in_channels": {"type": "int", "value": 4}}}
    config = {"typed": tree, "sha256": hashlib.sha256(c.encode(tree)).hexdigest()}
    source = manifest["rows"][0]
    provider = {
        "event": "provider_return",
        "native_tid": 1,
        "source_sha256": source["sha256"],
        "source_shape": source["shape"],
        "returned": tensor(
            source["shape"],
            "torch.float16" if engine == "cozy" else "torch.float32",
            "cuda:0" if engine == "cozy" else "cpu",
        ),
    }
    initial = {
        "event": "initial_sampler_state",
        "native_tid": 1,
        "state": tensor(source["shape"]),
        "rng": {
            "cpu": tensor([5], "torch.uint8", "cpu"),
            "device": tensor([5], "torch.uint8", "cpu"),
        },
    }
    extra = []
    if engine == "cozy":
        initial["schedule"] = {
            "config": config,
            "timesteps": tensor([20], "torch.float32"),
            "sigmas": tensor([21], "torch.float32", "cpu"),
        }
    else:
        extra = [
            {"event": "sampler_schedule", "native_tid": 1, "sigmas": tensor([21], "torch.float32")},
            {"event": "execution_config", "native_tid": 1, "unet": config, "sampling": config},
        ]
    inputs = {}
    outputs = {}
    for label in ["negative", "positive"]:
        inputs[label] = {
            ("sample" if engine == "cozy" else "x"): tensor([1, 4, 128, 128]),
            ("encoder_hidden_states" if engine == "cozy" else "context"): tensor([1, 77, 2048]),
            ("timestep" if engine == "cozy" else "timesteps"): tensor([1], "torch.float32"),
        }
        outputs[label] = {"prediction": tensor([1, 4, 128, 128])}
    net = {
        "event": "network",
        "native_tid": 1,
        "complete": True,
        "ordinal": 1,
        "output_kind": "epsilon",
        "training": False,
        "branches": ["negative", "positive"],
        "branch_inputs": inputs,
        "branch_outputs": outputs,
        "config": config,
    }
    return {
        "model": "sdxl",
        "request": {
            "seed": 1005,
            "steps": 20,
            "guidance": 7,
            "sampler": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "prompt_sha256": "a" * 64,
            "negative_prompt_sha256": "b" * 64,
        },
        "seed": 1005,
        "engine": engine,
        "complete": True,
        "exception": None,
        "native_tid": 1,
        "max_tensor_bytes": c.MAX_TENSOR,
        "max_copied_bytes": c.MAX_COPIED,
        "copied_bytes": sum(x["logical_bytes"] for x in allcopies),
        "settings": {
            "torch": "2.14",
            "cuda_build": "13",
            "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "cudnn_allow_tf32": True,
            "matmul_allow_tf32": False,
            "deterministic_algorithms": False,
        },
        "events": [provider, *extra, initial, net],
    }


def test_comparison_never_rounds_rawnoise_into_false_equality():
    result = c.compare(fixture("cozy"), fixture("comfy"), manifest)
    assert result["complete"] and result["canonical_raw_source_equal"]
    assert not result["returned_noise_logical_equal"]
    assert not result["full_network_argument_equivalence_established"]


def test_partial_branch_retry_modes_and_counters_fail_closed():
    edits = [
        lambda d: d.update(complete=False),
        lambda d: d.pop("settings"),
        lambda d: d["events"][-1].update(complete=False),
        lambda d: d["events"][-1].update(ordinal=2),
        lambda d: d["events"][-1].update(branches=["unknown"]),
        lambda d: d["events"][-1]["branch_inputs"].pop("negative"),
        lambda d: d["events"][0].update(source_sha256="f" * 64),
        lambda d: d.update(copied_bytes=0),
        lambda d: d["events"][-1]["branch_outputs"]["positive"]["prediction"][
            "allocator_after"
        ].update(num_ooms=1),
    ]
    for edit in edits:
        doc = fixture("cozy")
        edit(doc)
        assert not c.compare(doc, fixture("comfy"), manifest)["complete"]


def test_explicit_comfy_dtype_is_not_an_arbitrary_string_coercion():
    assert c.typed_valid({"type": "torch_dtype", "value": "torch.float16"})
    assert not c.typed_valid({"type": "torch_dtype", "value": "torch.unknown"})
    assert not c.typed_valid({"type": "torch_dtype", "value": None})
