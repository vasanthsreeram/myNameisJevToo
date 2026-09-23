"""myNameisJevToo - convert any causal language model into a Jev-style
(System One) decision model, with no retraining and no new weights.

The name is a tease and a tribute. It nods to TypeSafe AI's Jev, the model that
made "typed decisions with calibrated probabilities" a category worth building
in, and it is honest about what this is: not a reimplementation of Jev, and not
a claim to match it, but the same interface shape applied to weights anyone can
download.

    from jevtoo import convert

    jev = convert("openbmb/MiniCPM5-2B-MLX")
    jev.noul(state, "Is this claim fraudulent?")        # -> 0.83
    jev.choice(state, "Route this ticket.", LABELS)     # -> distribution
"""
from .decide import DecisionModel, convert, convert_gguf
from .calibrate import abstain_curve, chance_corrected, ece, fit_platt, noise_floor
from .readout import Distribution, LETTERS, render_options, render_state

__version__ = "0.1.0"
__all__ = [
    "convert", "convert_gguf", "DecisionModel", "Distribution", "LETTERS",
    "render_state", "render_options",
    "ece", "fit_platt", "abstain_curve", "noise_floor", "chance_corrected",
]
