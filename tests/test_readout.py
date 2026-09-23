"""Tests for the pure logic - no model weights, no MLX needed."""
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jevtoo import Distribution, LETTERS, chance_corrected, ece  # noqa: E402
from jevtoo.readout import peakedness, render_options, render_state, softmax  # noqa: E402


def test_share_renormalises():
    d = Distribution(labels=["a", "b"], probs=[0.1, 0.1], raw=[0.1, 0.1])
    assert abs(sum(d.share) - 1.0) < 1e-9
    assert abs(d.share[0] - 0.5) < 1e-9


def test_choice_and_confidence():
    d = Distribution(labels=["a", "b", "c"], probs=[0.2, 0.6, 0.2], raw=[0.2, 0.6, 0.2])
    assert d.choice == "b"
    assert abs(d.confidence - 0.6) < 1e-9


def test_noul_reads_the_positive_label():
    d = Distribution(labels=["no", "yes"], probs=[0.25, 0.75], raw=[0.25, 0.75])
    assert abs(d.noul() - 0.75) < 1e-9
    assert abs(d.noul(positive="no") - 0.25) < 1e-9


def test_noul_works_when_yes_is_first():
    """Order must not change the reported probability. If this ever fails for a
    real model, that is a finding about the model, not this function."""
    d = Distribution(labels=["yes", "no"], probs=[0.75, 0.25], raw=[0.75, 0.25])
    assert abs(d.noul() - 0.75) < 1e-9


def test_score_expected_value():
    d = Distribution(labels=["0", "1", "2", "3"], probs=[0, 0, 0, 1], raw=[0, 0, 0, 1])
    assert abs(d.score() - 3.0) < 1e-9


def test_peakedness_matches_the_published_three_way_examples():
    # (3 * 0.95 - 1) / 2 = 0.925, printed as 0.92 on the confidence page.
    assert abs(peakedness([0.0, 0.95, 0.05]) - 0.925) < 1e-12
    # uniform is zero, a sure option is one
    assert peakedness([1 / 3, 1 / 3, 1 / 3]) == 0.0
    assert peakedness([1.0, 0.0, 0.0]) == 1.0


def test_softmax_stable_on_large_inputs():
    out = softmax([1e4, 1e4 + 1])
    assert abs(sum(out) - 1.0) < 1e-9
    assert out[1] > out[0]


def test_render_options_includes_criteria():
    text = render_options(["yes", "no"], {"true": "It is permitted.", "false": "It is not."})
    assert "A. yes - It is permitted." in text
    assert "B. no - It is not." in text


def test_render_state_ends_with_answer():
    text = render_state("STATE", "QUESTION", ["a", "b"])
    assert text.startswith("STATE")
    assert text.rstrip().endswith("Answer:")


def test_ece_zero_for_perfect_calibration_ish():
    pairs = [(1.0, True)] * 10 + [(0.0, False)] * 10
    val, bins = ece(pairs)
    assert val < 0.05, val


def test_ece_high_when_confidently_wrong():
    pairs = [(0.99, False)] * 20
    val, _ = ece(pairs)
    assert val > 0.9, val


def test_chance_corrected_clips():
    assert chance_corrected(0.5, 0.25) == 100 * (0.5 - 0.25) / 0.75
    assert chance_corrected(0.1, 0.25) == 0.0
    assert chance_corrected(1.0, 0.25) == 100.0


def test_letters_are_expected_set():
    assert LETTERS[0] == "A" and LETTERS[25] == "Z"
