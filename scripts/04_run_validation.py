"""Run all three validation experiments and print pass/fail.

The project is not shippable until this prints "ALL PASS".
"""
from __future__ import annotations


def main():
    results = {}
    try:
        from src.validation import test_tripod_gait, test_stop_command, test_turning

        for name, mod in [
            ("tripod_gait", test_tripod_gait),
            ("stop_command", test_stop_command),
            ("turning", test_turning),
        ]:
            try:
                mod.main()
                results[name] = "PASS"
            except NotImplementedError:
                results[name] = "SKIP (not implemented)"
            except AssertionError as e:
                results[name] = f"FAIL: {e}"
            except Exception as e:
                results[name] = f"ERROR: {type(e).__name__}: {e}"
    finally:
        for k, v in results.items():
            print(f"  {k:15s} {v}")
        if all(v == "PASS" for v in results.values()) and results:
            print("\nALL PASS")


if __name__ == "__main__":
    main()
