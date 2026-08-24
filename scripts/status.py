"""Print Nishana's status numbers from committed artifacts.

    uv run python scripts/status.py
    uv run python scripts/status.py --json

Every number this prints comes from a file in results/, and every file there
names the script that produced it. Nothing here estimates, extrapolates, or
fills in: a section with no artifact yet says so and names its command. The
headline number is the same one the portfolio page carries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nishana.report import RESULTS_DIR, render_status, status_number  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the headline number as JSON")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    if args.json:
        headline = status_number(Path(args.results_dir))
        print(json.dumps(headline or {"metric": None, "value": None}, indent=2))
        return 0
    print(render_status(Path(args.results_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
