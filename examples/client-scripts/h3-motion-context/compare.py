# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["minimax-h3"]
# [tool.uv.sources]
# minimax-h3 = {path = "./candidate", editable = true}
# ///
"""Run next to the prepared private candidate directory using cozy run compare.py."""

from h3_motion_context import Comparison, ComparisonInput, StoryReference, StorySegment, compare


async def main() -> Comparison:
    return await compare(  # type: ignore[call-arg]
        payload=ComparisonInput(
            references=[
                StoryReference(
                    name="courier",
                    kind="character",
                    seed=1701,
                    description=(
                        "An adult female courier with short dark hair, a mustard jacket, navy "
                        "trousers and brown boots."
                    ),
                ),
                StoryReference(
                    name="depot",
                    kind="scene",
                    seed=1702,
                    description=(
                        "A sunlit brick railway depot courtyard with a blue cargo cart and "
                        "cobblestones."
                    ),
                ),
            ],
            shared=(
                "A continuous eye-level tracking shot, natural daylight. Footsteps and "
                "soft outdoor wind, no music, no voices."
            ),
            predecessor=StorySegment(
                summary="[reference generation] <courier> approaches the cart in <depot>.",
                detailed_description=(
                    "[Shot 1] <courier> walks steadily across <depot> toward the blue cargo cart. "
                    "The camera tracks smoothly beside her."
                ),
                overall_soundscape="Footsteps and soft outdoor wind.",
                non_diegetic_music="N/A",
            ),
            continuation=StorySegment(
                summary="[reference generation] <courier> reaches the cart in <depot>.",
                detailed_description=(
                    "[Shot 1] <courier> keeps walking across <depot>, reaches the blue cargo cart "
                    "and rests one hand on its handle. The camera keeps tracking smoothly."
                ),
                overall_soundscape="Footsteps and soft outdoor wind.",
                non_diegetic_music="N/A",
            ),
            seed=41001,
        )
    )
