"""
merge_runs.py — Merge separate per-model run directories into one unified result.
Run: python evaluation/merge_runs.py
"""
import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"

# Find all run directories that have full_results.json
run_dirs = sorted([d for d in RESULTS_DIR.iterdir()
                   if d.is_dir() and (d / "full_results.json").exists()], reverse=True)

print(f"Found {len(run_dirs)} completed run directories:")
for d in run_dirs:
    records = json.load(open(d / "full_results.json", encoding="utf-8"))
    models = list({r["model"] for r in records})
    print(f"  {d.name}: {len(records)} records, models={models}")

# Take the 4 most recent (one per model)
all_records = []
seen_models = set()
for run_dir in run_dirs:
    records = json.load(open(run_dir / "full_results.json", encoding="utf-8"))
    new_records = [r for r in records if r["model"] not in seen_models]
    if new_records:
        models_added = list({r["model"] for r in new_records})
        seen_models.update(models_added)
        all_records.extend(new_records)
        print(f"  -> Added {len(new_records)} records from {run_dir.name} ({models_added})")
    if len(seen_models) >= 4:
        break

print(f"\nMerged: {len(all_records)} total records across {len(seen_models)} models")

# Recompute aggregate metrics from merged results
def safe_mean(lst):
    return round(sum(lst) / len(lst), 3) if lst else None

model_stats = defaultdict(lambda: {
    "doctrinal_accuracy": [], "hallucination_rate": [],
    "citation_recall": [], "conflict_correct": [],
    "pluralism_score": [], "concept_coverage": [],
    "elapsed_sec": [], "traps_hit_count": 0, "total": 0, "errors": 0,
})

for r in all_records:
    m = r["model"]
    met = r["metrics"]
    model_stats[m]["total"] += 1
    if "error" in r.get("output", {}):
        model_stats[m]["errors"] += 1
    if met["doctrinal_accuracy"] is not None:
        model_stats[m]["doctrinal_accuracy"].append(met["doctrinal_accuracy"])
    model_stats[m]["hallucination_rate"].append(met["hallucination_rate"])
    model_stats[m]["citation_recall"].append(met["citation_recall"])
    model_stats[m]["conflict_correct"].append(int(met["conflict_correct"]))
    model_stats[m]["pluralism_score"].append(met["pluralism_score"])
    if met["concept_coverage"] is not None:
        model_stats[m]["concept_coverage"].append(met["concept_coverage"])
    model_stats[m]["elapsed_sec"].append(r["output"].get("elapsed_sec", 0))
    model_stats[m]["traps_hit_count"] += len(met["traps_hit"])

unified_summary = {}
for model, stats in model_stats.items():
    unified_summary[model] = {
        "total_scenarios": stats["total"],
        "errors": stats["errors"],
        "doctrinal_accuracy_mean": safe_mean(stats["doctrinal_accuracy"]),
        "hallucination_rate_mean": safe_mean(stats["hallucination_rate"]),
        "citation_recall_mean": safe_mean(stats["citation_recall"]),
        "conflict_detection_accuracy": safe_mean(stats["conflict_correct"]),
        "pluralism_score_mean": safe_mean(stats["pluralism_score"]),
        "concept_coverage_mean": safe_mean(stats["concept_coverage"]),
        "avg_response_time_sec": safe_mean(stats["elapsed_sec"]),
        "total_traps_hit": stats["traps_hit_count"],
    }

# Save unified results
(RESULTS_DIR / "latest_results.json").write_text(
    json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
(RESULTS_DIR / "latest_summary.json").write_text(
    json.dumps(unified_summary, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n=== UNIFIED METRICS SUMMARY ===")
print(f"{'Model':<18} {'Acc':>7} {'Halluc':>8} {'CitRecall':>10} {'ConflAcc':>10} {'Plural':>8} {'Traps':>6}")
print("─" * 70)
for model in ["vanilla_llm", "rag", "symbolic", "hybrid"]:
    s = unified_summary.get(model, {})
    if not s:
        continue
    acc = f"{s['doctrinal_accuracy_mean']:.3f}" if s['doctrinal_accuracy_mean'] else "  N/A"
    hal = f"{s['hallucination_rate_mean']:.3f}" if s['hallucination_rate_mean'] is not None else "  N/A"
    cit = f"{s['citation_recall_mean']:.3f}" if s['citation_recall_mean'] is not None else "  N/A"
    con = f"{s['conflict_detection_accuracy']:.3f}" if s['conflict_detection_accuracy'] is not None else "  N/A"
    plu = f"{s['pluralism_score_mean']:.3f}" if s['pluralism_score_mean'] is not None else "  N/A"
    trp = str(s['total_traps_hit'])
    print(f"{model:<18} {acc:>7} {hal:>8} {cit:>10} {con:>10} {plu:>8} {trp:>6}")

print(f"\nSaved to: {RESULTS_DIR}/latest_results.json + latest_summary.json")
