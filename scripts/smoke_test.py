"""
Smoke test: Validates that all core package imports work correctly.
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Checks: (name, import_func, requires_gpu)
CHECKS = [
    ("src.config.ModelConfig", "src.config", "ModelConfig", False),
    ("src.config.TrainConfig", "src.config", "TrainConfig", False),
    ("src.data.DataProcessor", "src.data", "DataProcessor", False),
    ("src.train.train_model", "src.train", "train_model", False),
    ("src.core.factory.ModelFactory", "src.core.factory", "ModelFactory", True),
    ("src.core.model_runner.ModelRunner", "src.core.model_runner", "ModelRunner", True),
    ("src.utils.env.HardwareManager", "src.utils.env", "HardwareManager", False),
    ("src.cli.main", "src.cli", "main", True),
]


def run_smoke_test():
    print("=" * 50)
    print("  Smoke Test: Checking Package Imports")
    print("=" * 50)

    passed = 0
    failed = 0
    skipped = 0

    for name, module, attr, requires_gpu in CHECKS:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"  [PASS] {name}")
            passed += 1
        except ModuleNotFoundError as e:
            missing = str(e)
            if requires_gpu and ("unsloth" in missing or "triton" in missing):
                print(f"  [SKIP] {name} (requires GPU package: {missing})")
                skipped += 1
            else:
                print(f"  [FAIL] {name} -- {e}")
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {name} -- {e}")
            failed += 1

    print("-" * 50)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  All critical imports successful. Structure OK.\n")


if __name__ == "__main__":
    run_smoke_test()
