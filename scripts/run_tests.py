"""Run every offline test directory in an isolated process (standard library only)."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    directories = sorted({path.parent for path in (ROOT / "docs").rglob("test_*.py")})
    if not directories:
        print("No test files found.", file=sys.stderr)
        return 1
    failed = []
    for directory in directories:
        relative = directory.relative_to(ROOT)
        print(f"Running {relative}", flush=True)
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(relative), "-p", "test_*.py", "-v"], cwd=ROOT)
        if result.returncode:
            failed.append(str(relative))
    print(f"Test directories: {len(directories)}; failed: {len(failed)}", flush=True)
    for directory in failed:
        print(f"FAILED: {directory}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
