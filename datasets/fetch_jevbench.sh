#!/bin/bash
# Fetch JevBench's frozen PUBLIC task items.
#
# Task data is fstandhartinger/jevbench's, not ours, so it is fetched rather than
# vendored. Read their repository's terms before redistributing the item text.
# The router / judge / held-out splits are not public, so the judge tier cannot
# be run locally - see benchmarks/jevbench.py.
set -euo pipefail

DEST="${1:-datasets/jevbench}"
BASE="https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public"
mkdir -p "$DEST"

for f in easy.jsonl original.jsonl hard.jsonl; do
  curl -sSL "$BASE/$f" -o "$DEST/$f"
  printf '%9d bytes  %s\n' "$(wc -c < "$DEST/$f")" "$DEST/$f"
done

# Scoring code, so the numbers can be checked against their formulas
curl -sSL "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/jevbench/composite_v13.py" \
  -o "$DEST/../jevbench_composite_v13.py"
curl -sSL "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/jevbench/metrics.py" \
  -o "$DEST/../jevbench_metrics.py"

echo
echo "tiers available locally:"
echo "  easy     <- easy.jsonl      (48 public of 72)"
echo "  standard <- original.jsonl  (72 public of 96)"
echo "  hard     <- hard.jsonl      (111 public of 220)"
echo
echo "not available (not public): judge tier (router+judge splits), held-out halves."
