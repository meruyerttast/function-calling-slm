"""Emit LaTeX rows for the report's V1-V6 result tables from scores.json.

Reads results/v{1..6}/scores.json and prints the rows for:
  - Non-Live sub-buckets (data_non_live.csv columns)
  - V4-weighted top-level summary (top_level + overall)

Usage:
    python tools/fill_report.py            # print rows to stdout
    python tools/fill_report.py --check    # warn on missing variants
"""

import argparse
import csv
import json
import sys
from pathlib import Path

VARIANTS = [
    ("v1", "V1 -- Base"),
    ("v2", "V2 -- SFT"),
    ("v3", "V3 -- SFT+Masking"),
    ("v4", "V4 -- GRPO cold"),
    ("v5", "V5 -- GRPO+Masked"),
    ("v6", "V6 -- Full Recipe"),
]


def fmt_pct(v):
    if v is None:
        return r"\tbd"
    return f"{float(v) * 100:.2f}\\%"


def load_scores(variant: str) -> dict | None:
    path = Path(f"results/{variant}/scores.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_non_live_subs(variant: str) -> dict[str, float | None]:
    """Pull per-subcategory Non-Live values from data_non_live.csv."""
    csv_path = Path(f"results/{variant}/data_non_live.csv")
    cols = ("Non-Live Simple AST", "Non-Live Multiple AST",
            "Non-Live Parallel AST", "Non-Live Parallel Multiple AST",
            "Irrelevance Detection")
    out = {c: None for c in cols}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return out
    row = rows[0]
    for c in cols:
        s = (row.get(c) or "").rstrip("%").strip()
        try:
            v = float(s)
            out[c] = v / 100 if v > 1 else v
        except (ValueError, TypeError):
            pass
    return out


def emit_non_live_table():
    print("% --- Non-Live result table (auto-generated) ---")
    for vid, label in VARIANTS:
        scores = load_scores(vid)
        subs = load_non_live_subs(vid)
        overall = scores["top_level"]["non_live"] if scores else None
        cells = [
            label,
            fmt_pct(overall),
            fmt_pct(subs["Non-Live Simple AST"]),
            fmt_pct(subs["Non-Live Multiple AST"]),
            fmt_pct(subs["Non-Live Parallel AST"]),
            fmt_pct(subs["Non-Live Parallel Multiple AST"]),
            fmt_pct(subs["Irrelevance Detection"]),
        ]
        print(" & ".join(cells) + r" \\")


def emit_top_level_table():
    print("% --- Top-level V4-weighted table (auto-generated) ---")
    for vid, label in VARIANTS:
        scores = load_scores(vid)
        if scores is None:
            cells = [label] + [r"\tbd"] * 6
        else:
            tl = scores["top_level"]
            overall = scores.get("overall")
            cells = [
                label,
                fmt_pct(tl.get("non_live")),
                fmt_pct(tl.get("live")),
                fmt_pct(tl.get("hallucination")),
                fmt_pct(tl.get("multi_turn")),
                fmt_pct(tl.get("agentic")),
                fmt_pct(overall),
            ]
        print(" & ".join(cells) + r" \\")


def check_missing():
    missing = [v for v, _ in VARIANTS if load_scores(v) is None]
    if missing:
        print(f"[fill_report] missing scores.json for: {missing}", file=sys.stderr)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if any variant scores.json is missing.")
    args = parser.parse_args()
    if args.check:
        sys.exit(check_missing())
    emit_non_live_table()
    print()
    emit_top_level_table()


if __name__ == "__main__":
    main()
