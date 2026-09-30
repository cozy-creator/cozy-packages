"""CPU evidence gates; never coerce dtype/shape/branch identity into equality."""

import hashlib
import json
import math
from pathlib import Path

MAX_TENSOR = 2 * 512 * 2048 * 2
MAX_COPIED = 16 * MAX_TENSOR
SIZES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.uint8": 1,
    "torch.bool": 1,
}
ALLOC = {
    "allocated_bytes.all.current",
    "reserved_bytes.all.current",
    "num_ooms",
    "num_alloc_retries",
}


def integer(x):
    return type(x) is int and x >= 0


def checksum(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def encode(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def typed_valid(x):
    if not isinstance(x, dict):
        return False
    t = x.get("type")
    if t == "torch_dtype":
        return set(x) == {"type", "value"} and x["value"] in set(SIZES) | {
            "torch.int16",
            "torch.int8",
        }
    if t == "null":
        return set(x) == {"type"}
    if t in ("bool", "int", "str"):
        return (
            set(x) == {"type", "value"}
            and type(x["value"]) is {"bool": bool, "int": int, "str": str}[t]
        )
    if t == "float":
        try:
            return (
                set(x) == {"type", "hex"}
                and math.isfinite(float.fromhex(x["hex"]))
                and float.fromhex(x["hex"]).hex() == x["hex"]
            )
        except (ValueError, TypeError):
            return False
    if t in ("tuple", "list"):
        return (
            set(x) == {"type", "items"}
            and isinstance(x["items"], list)
            and all(typed_valid(v) for v in x["items"])
        )
    if t == "dict":
        return (
            set(x) == {"type", "items"}
            and isinstance(x["items"], dict)
            and all(type(k) is str and typed_valid(v) for k, v in x["items"].items())
        )
    return False


def config_valid(x):
    return (
        isinstance(x, dict)
        and typed_valid(x.get("typed"))
        and hashlib.sha256(encode(x["typed"])).hexdigest() == x.get("sha256")
        and len(encode(x)) < 66000
    )


def digest_valid(x):
    if not isinstance(x, dict) or not checksum(x.get("sha256")):
        return False
    shape = x.get("shape")
    stride = x.get("stride")
    size = SIZES.get(x.get("dtype"))
    if (
        size is None
        or not isinstance(shape, list)
        or not all(integer(n) for n in shape)
        or not isinstance(stride, list)
        or len(stride) != len(shape)
        or not all(integer(n) for n in stride)
    ):
        return False
    n = math.prod(shape) * size
    if not integer(x.get("storage_offset")) or x.get("logical_bytes") != n or n > MAX_TENSOR:
        return False
    if not isinstance(x.get("device"), str) or not (
        x["device"] == "cpu" or x["device"].startswith("cuda:")
    ):
        return False
    if not type(x.get("copy_index")) is int or x["copy_index"] <= 0:
        return False
    if x["device"] == "cpu":
        return x.get("allocator_before") == {} and x.get("allocator_after") == {}
    for k in ["allocator_before", "allocator_after"]:
        a = x.get(k)
        if not isinstance(a, dict) or set(a) != ALLOC or not all(integer(v) for v in a.values()):
            return False
    return x["allocator_before"] == x["allocator_after"]


def logical(x):
    return {k: x[k] for k in ["sha256", "shape", "dtype", "logical_bytes"]}


def tensors(x):
    if isinstance(x, dict):
        if "sha256" in x and "shape" in x:
            yield x
        else:
            for v in x.values():
                yield from tensors(v)
    elif isinstance(x, list):
        for v in x:
            yield from tensors(v)


def validate(doc, manifest):
    reasons = []
    events = doc.get("events", [])
    model = doc.get("model")
    seed = doc.get("seed")
    if (model, seed) not in [("sdxl", 1005), ("anima", 1006)] or doc.get("engine") not in (
        "cozy",
        "comfy",
    ):
        reasons.append("request_identity")
    request = doc.get("request", {})
    if (
        request.get("seed") != seed
        or type(request.get("steps")) is not int
        or request["steps"] != (20 if model == "sdxl" else 30)
        or request.get("guidance") != (7 if model == "sdxl" else 4.5)
    ):
        reasons.append("request_modes")
    if doc.get("engine") == "comfy" and (
        request.get("sampler") != "euler"
        or request.get("scheduler") != ("normal" if model == "sdxl" else "simple")
        or request.get("denoise") != 1.0
    ):
        reasons.append("sampler_modes")
    if doc.get("engine") == "cozy" and not all(
        checksum(request.get(k + "_sha256")) for k in ("prompt", "negative_prompt")
    ):
        reasons.append("prompt_identity")
    if doc.get("complete") is not True or doc.get("exception") is not None:
        reasons.append("failed_request")
    if (
        not integer(doc.get("copied_bytes"))
        or doc["copied_bytes"] > MAX_COPIED
        or doc.get("max_tensor_bytes") != MAX_TENSOR
        or doc.get("max_copied_bytes") != MAX_COPIED
    ):
        reasons.append("copy_bound")
    if not isinstance(events, list) or len(events) > 24:
        reasons.append("event_bound")
        events = []
    if (
        not type(doc.get("native_tid")) is int
        or doc["native_tid"] <= 0
        or any(e.get("native_tid") != doc["native_tid"] for e in events)
    ):
        reasons.append("tid_association")
    settings = doc.get("settings", {})
    if (
        not isinstance(settings, dict)
        or not isinstance(settings.get("torch"), str)
        or not isinstance(settings.get("cuda_build"), str)
        or any(
            type(settings.get(k)) is not bool
            for k in [
                "cudnn_benchmark",
                "cudnn_deterministic",
                "cudnn_allow_tf32",
                "matmul_allow_tf32",
                "deterministic_algorithms",
            ]
        )
    ):
        reasons.append("missing_modes")
    provider = [e for e in events if e.get("event") == "provider_return"]
    initial = [e for e in events if e.get("event") == "initial_sampler_state"]
    network = [e for e in events if e.get("event") == "network"]
    if len(provider) != 1 or len(initial) != 1 or not 1 <= len(network) <= 2:
        reasons.append("missing_or_duplicate_boundary")
    copies = {}
    for t in tensors(events):
        if not digest_valid(t):
            reasons.append("invalid_or_nonreadonly_tensor_record")
            continue
        idx = t["copy_index"]
        if idx in copies and copies[idx] != t:
            reasons.append("contradictory_copy_record")
        copies[idx] = t
    if set(copies) != set(range(1, len(copies) + 1)) or sum(
        t["logical_bytes"] for t in copies.values()
    ) != doc.get("copied_bytes"):
        reasons.append("partial_copy_ledger")
    if not reasons:
        if not (events.index(provider[0]) < events.index(initial[0]) < events.index(network[0])):
            reasons.append("boundary_order")
        expected = next(
            (x for x in manifest["rows"] if x["model"] == model and x["seed"] == seed), None
        )
        if (
            expected is None
            or provider[0].get("source_sha256") != expected["sha256"]
            or provider[0].get("source_shape") != expected["shape"]
        ):
            reasons.append("wrong_source")
        if not digest_valid(provider[0].get("returned")) or not digest_valid(
            initial[0].get("state")
        ):
            reasons.append("missing_initial_values")
        else:
            returned = provider[0]["returned"]
            allowed = [expected["shape"]]
            if model == "anima" and doc["engine"] == "comfy":
                allowed.append([1, 16, 128, 128])
            if returned["shape"] not in allowed or returned["dtype"] != (
                "torch.float16" if model == "sdxl" and doc["engine"] == "cozy" else "torch.float32"
            ):
                reasons.append("unexpected_provider_cast_or_axis")
            if (
                returned["shape"] != expected["shape"]
                and provider[0].get("axis_mapping") != "NCTHW_to_NCHW_squeeze_temporal_2"
            ):
                reasons.append("unknown_axis_mapping")
        rng = initial[0].get("rng", {})
        if not digest_valid(rng.get("device")):
            reasons.append("missing_device_rng")
        # CPU RNG records have device=cpu; these remain captured but are not compared asCUDA inputs.
        if not isinstance(rng.get("cpu"), dict) or not checksum(rng["cpu"].get("sha256")):
            reasons.append("missing_cpu_rng")
        if doc["engine"] == "cozy":
            schedule = initial[0].get("schedule", {})
            if (
                not config_valid(schedule.get("config"))
                or not digest_valid(schedule.get("timesteps"))
                or not digest_valid(schedule.get("sigmas"))
            ):
                reasons.append("missing_schedule")
        else:
            schedules = [e for e in events if e["event"] == "sampler_schedule"]
            configs = [e for e in events if e["event"] == "execution_config"]
            if len(schedules) != 1 or not digest_valid(schedules[0].get("sigmas")):
                reasons.append("missing_schedule")
            if (
                len(configs) != 1
                or not config_valid(configs[0].get("unet"))
                or not config_valid(configs[0].get("sampling"))
            ):
                reasons.append("missing_execution_config")
    branches = {}
    for i, n in enumerate(network, 1):
        if (
            n.get("complete") is not True
            or n.get("ordinal") != i
            or n.get("output_kind") != ("epsilon" if model == "sdxl" else "flow_velocity")
            or type(n.get("training")) is not bool
            or "exception" in n
        ):
            reasons.append("failed_or_wrong_prediction")
        labels = n.get("branches", [])
        if (
            not labels
            or len(set(labels)) != len(labels)
            or any(x not in ("positive", "negative") for x in labels)
        ):
            reasons.append("unknown_branch")
            continue
        if set(n.get("branch_inputs", {})) != set(labels) or set(
            n.get("branch_outputs", {})
        ) != set(labels):
            reasons.append("partial_branches")
            continue
        if doc.get("engine") == "cozy" and not config_valid(n.get("config")):
            reasons.append("missing_network_config")
        for label in labels:
            if label in branches:
                reasons.append("repeated_branch")
            inputs = n["branch_inputs"][label]
            prediction = n["branch_outputs"][label].get("prediction")
            keys = (
                ("sample", "encoder_hidden_states", "timestep")
                if model == "sdxl" and doc["engine"] == "cozy"
                else ("hidden_states", "encoder_hidden_states", "timestep")
                if doc["engine"] == "cozy"
                else ("x", "context", "timesteps")
            )
            if not all(digest_valid(inputs.get(k)) for k in keys) or not digest_valid(prediction):
                reasons.append("partial_network_values")
                continue
            expected_shape = [1, 4, 128, 128] if model == "sdxl" else [1, 16, 1, 128, 128]
            expected_dtype = "torch.float16" if model == "sdxl" else "torch.bfloat16"
            if (
                inputs[keys[0]]["shape"] != expected_shape
                or prediction["shape"] != expected_shape
                or inputs[keys[0]]["dtype"] != expected_dtype
                or prediction["dtype"] != expected_dtype
            ):
                reasons.append("unexpected_network_geometry")
            context_shape = inputs[keys[1]]["shape"]
            if (
                len(context_shape) != 3
                or context_shape[0] != 1
                or not 0 < context_shape[1] <= 512
                or context_shape[2] != (2048 if model == "sdxl" else 1024)
            ):
                reasons.append("unexpected_conditioning_geometry")
            branches[label] = {
                "input": inputs[keys[0]],
                "context": inputs[keys[1]],
                "timestep": inputs[keys[2]],
                "prediction": prediction,
            }
    if set(branches) != {"positive", "negative"}:
        reasons.append("missing_branch_coverage")
    return {
        "complete": not reasons,
        "reasons": sorted(set(reasons)),
        "branches": branches,
        "provider": provider[0] if len(provider) == 1 else None,
    }


def compare(a, b, manifest):
    x, y = validate(a, manifest), validate(b, manifest)
    out = {
        "complete": x["complete"] and y["complete"],
        "left": x,
        "right": y,
        "full_network_argument_equivalence_established": False,
    }
    if not out["complete"]:
        return out
    if a["model"] != b["model"] or a["seed"] != b["seed"] or a["engine"] == b["engine"]:
        out["complete"] = False
        out["reasons"] = ["pair_identity"]
        return out
    out["canonical_raw_source_equal"] = (
        x["provider"]["source_sha256"] == y["provider"]["source_sha256"]
    )
    out["returned_noise_logical_equal"] = logical(x["provider"]["returned"]) == logical(
        y["provider"]["returned"]
    )
    out["branch_comparison"] = {
        label: {
            key: logical(x["branches"][label][key]) == logical(y["branches"][label][key])
            for key in ["input", "context", "timestep", "prediction"]
        }
        for label in ["positive", "negative"]
    }
    out["scope"] = (
        "Noquietcasts/reshape/scale; extra conditioning schemas and scheduler parameterizations remain separate. This is quality evidence, not numericalparity."
    )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    for name in ("left", "right", "manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "9dea16ec10e3582af052bde91336f9e8c8a6ab62d64c7444ea6336f8ecee4875"
    ):
        raise ValueError("unreviewed canonical noise manifest")
    result = compare(
        json.loads(args.left.read_bytes()), json.loads(args.right.read_bytes()), json.loads(raw)
    )
    result["scored"] = False
    result["artifact_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.left, args.right, args.manifest)
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
