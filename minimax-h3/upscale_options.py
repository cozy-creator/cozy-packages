"""Closed output choices shared by the H3 generation and upscaling workflows."""

from enum import Enum


class Resolution(Enum):
    NATIVE = "768p"
    FULL_HD = "1080p"
    QHD = "1440p"
    UHD = "2160p"


class UpscaleMethod(Enum):
    SEEDVR2 = "seedvr2"
    H3_LATENT = "h3_latent"


OUTPUT_SIZES = {
    Resolution.NATIVE: (1344, 768),
    Resolution.FULL_HD: (1920, 1080),
    Resolution.QHD: (2560, 1440),
    Resolution.UHD: (3840, 2160),
}

# H3's learned refinement stays on the 32-pixel token grid, at the original 7:4
# aspect ratio. Standard media fitting converts it to the selected delivery size.
REFINEMENT_SIZES = {
    Resolution.NATIVE: (1344, 768),
    Resolution.FULL_HD: (2016, 1152),
    Resolution.QHD: (2688, 1536),
    Resolution.UHD: (3808, 2176),
}
