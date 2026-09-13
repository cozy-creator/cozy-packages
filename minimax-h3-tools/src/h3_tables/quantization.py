"""The reviewed MiniMax-H3 policy supplied to generic quantization machinery."""

from importlib.resources import files


def h3_quantization_plan() -> bytes:
    """Return the exact pruned-DiT plan owned by this model package."""
    return files(__package__).joinpath("assets/h3-dit-quantization-plan.json").read_bytes()
