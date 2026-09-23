#!/usr/bin/env python3
"""Dependency-free test runner (the tests are plain pytest-style functions).

    python tests/run_tests.py
"""
import importlib.util
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

spec = importlib.util.spec_from_file_location(
    "test_readout", pathlib.Path(__file__).parent / "test_readout.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

tests = sorted((n, getattr(mod, n)) for n in dir(mod) if n.startswith("test_"))
passed = failed = 0
for name, fn in tests:
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception:  # noqa: BLE001
        failed += 1
        print(f"  FAIL  {name}")
        traceback.print_exc()

print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
sys.exit(1 if failed else 0)
