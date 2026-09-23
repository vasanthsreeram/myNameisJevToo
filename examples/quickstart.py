#!/usr/bin/env python3
"""End-to-end check that the packaged converter works, not just the scratch scripts.

    python examples/quickstart.py [model_path_or_hf_id]
"""
import sys

from jevtoo import abstain_curve, convert

STATE = (
    "Support ticket 44821. Customer Priya is on the Pro annual plan. She writes: "
    "'I was charged twice this month, on the 3rd and again on the 5th, both for 49 "
    "dollars. I have emailed twice and nobody replied. I need this refunded today or "
    "I am cancelling and disputing the charge with my bank.' Account history: plan "
    "upgraded in June, two successful payments in July and August, one failed payment "
    "attempt on the 4th that was retried and succeeded on the 5th. No prior refunds."
)

ROUTES = ["billing", "refund", "account", "feature"]
ADDRESSES_THE_REFUND = "Did the customer explicitly ask for a refund?"
URGENCY = ["low", "medium", "high"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openbmb/MiniCPM5-2B-MLX"
    jev = convert(model)

    print(f"model: {model}\n")

    d = jev.choice(STATE, "Route this support ticket to a team.", ROUTES)
    print("CHOICE  (pick-one)")
    for lab, s in zip(d.labels, d.share):
        print(f"   {lab:<10} {s*100:6.2f}%  {'<-' if lab == d.choice else ''}")
    print(f"   -> {d.choice}  confidence {d.confidence:.3f}  ({d.latency_s*1000:.0f} ms)\n")

    p = jev.noul(STATE, ADDRESSES_THE_REFUND)
    print("NOUL    (yes/no gate)")
    print(f"   {ADDRESSES_THE_REFUND}\n   -> P(yes) = {p:.3f}\n")

    ev, d2 = jev.score(STATE, "How urgent is this ticket?", URGENCY)
    print("SCORE   (rate on a scale)")
    for lab, s in zip(d2.labels, d2.share):
        print(f"   {lab:<8} {s*100:6.2f}%")
    print(f"   -> expected value {ev:.2f}\n")

    print("typed output, no prose generated:")
    print("  ", d.as_dict())

    # abstain curve shape, on a trivially small illustrative set
    pairs = [(x.confidence, x.choice == "refund") for x in [d]]
    print("\nabstain curve (illustrative, n=1):")
    for row in abstain_curve(pairs):
        print(f"   th>={row['threshold']:.2f}  coverage {row['coverage']*100:5.1f}%  "
              f"accuracy {row['accuracy'] if row['accuracy'] is None else round(row['accuracy']*100,1)}")

    total = (d.latency_s) * 3
    print(f"\nthree decisions = ~{total*1000:.0f} ms of forward passes total")


if __name__ == "__main__":
    main()
