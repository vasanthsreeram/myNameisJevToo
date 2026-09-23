#!/usr/bin/env python3
"""Is a binary answer reading the content, or the label position?

The aggregate accuracy in stateblind.py cannot tell you. Swapping two options can:
a model that always picks position A scores ~50% on binary items in ANY order,
so stable accuracy across orders is a symptom, not reassurance.

This is the test that caught it.
"""
import json
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from jevtoo import convert  # noqa: E402

STATE = (
    "Support ticket 44821. Customer Priya is on the Pro annual plan. She writes: "
    "'I was charged twice this month, on the 3rd and again on the 5th, both for 49 "
    "dollars. I have emailed twice and nobody replied. I need this refunded today or "
    "I am cancelling and disputing the charge with my bank.' Account history: plan "
    "upgraded in June, two successful payments in July and August, one failed payment "
    "attempt on the 4th that was retried and succeeded on the 5th. No prior refunds."
)
QUESTION = "Did the customer explicitly ask for a refund?"
# The state says "I need this refunded today". The honest answer is yes.
ORDER_A = ["no", "yes"]
ORDER_B = ["yes", "no"]

out = {}
jev = convert(os.environ.get("JEV_MODEL", "openbmb/MiniCPM5-2B-MLX"))

print(f"question: {QUESTION}")
print("state says: 'I need this refunded today'  -> the answer is yes\n")
for tag, state in (("with state", STATE), ("state withheld", "[no state provided]")):
    for order in (ORDER_A, ORDER_B):
        d = jev.options(state, order, QUESTION)
        dist = "  ".join(f"{l}={s*100:5.1f}%" for l, s in zip(d.labels, d.share))
        print(f"  {tag:<16} labels {str(order):<16} -> {dist}")
        out[f"{tag}|{','.join(order)}"] = dict(zip(d.labels, d.share))

flip = out["with state|no,yes"]["no"] > 0.5 and out["with state|yes,no"]["yes"] > 0.5
print(f"\nanswer flips purely from option order: {flip}")
print("verdict: " + ("POSITION-PICKING - this readout carries no content signal on"
                     " 2-option questions" if flip else "reading content"))
json.dump(out, open("benchmarks/results/binary_flip.json", "w"), indent=1)
