"""
run_stats.py — Run all Q1 statistical tests on merged results.
Uses eval_metrics.run_full_analysis() on the full merged dataset.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"


def load_merged_results():
    """Load all unique model×scenario records from all run directories."""
    records = []
    seen = set()
    run_dirs = sorted(
        [d for d in RESULTS_DIR.iterdir()
         if d.is_dir() and (d / "full_results.json").exists()],
        reverse=True,
    )
    for rd in run_dirs:
        recs = json.load(open(rd / "full_results.json", encoding="utf-8"))
        for rec in recs:
            key = f"{rec['model']}:{rec['scenario_id']}"
            if key not in seen:
                seen.add(key)
                records.append(rec)
    print(f"Loaded {len(records)} records from {len(run_dirs)} run directories")

    # Write to a temp file so run_full_analysis can read it
    tmp = RESULTS_DIR / "_merged_for_stats.json"
    tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(tmp)


if __name__ == "__main__":
    from evaluation.eval_metrics import run_full_analysis
    path = load_merged_results()
    run_full_analysis(results_path=path)
