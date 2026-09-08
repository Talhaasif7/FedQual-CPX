import subprocess
import sys

test_files = [
    "tests/test_drift.py",
    "tests/test_detectors.py",
    "tests/test_selectors.py",
    "tests/test_models.py",
    "tests/test_evaluation.py",
    "tests/test_cross_dataset.py",
]

all_passed = True
for tf in test_files:
    print(f"Running {tf} ...")
    res = subprocess.run([sys.executable, tf], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  [PASSED] {tf}")
    else:
        print(f"  [FAILED] {tf}")
        print(res.stdout)
        print(res.stderr)
        all_passed = False

if all_passed:
    print("\nALL TEST SUITES PASSED SUCCESSFULLY!")
else:
    sys.exit(1)
