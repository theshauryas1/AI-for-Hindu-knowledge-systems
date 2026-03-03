"""
print_results.py — Quick results printer for latest eval run.
Run: .venv\Scripts\python.exe evaluation/print_results.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path("evaluation/results")

# Merged summary
summary_path = RESULTS_DIR / "latest_summary.json"
results_path = RESULTS_DIR / "latest_results.json"

if not summary_path.exists():
    print("No merged results yet. Run: python evaluation/merge_runs.py")
    exit(1)

s = json.load(open(summary_path, encoding="utf-8"))
r = json.load(open(results_path, encoding="utf-8"))

print("\n" + "="*65)
print("HINDUMIND EVALUATION — FINAL RESULTS SUMMARY")
print("="*65)
header = f"{'Model':<16} {'n':>3} {'Acc':>7} {'Halluc':>8} {'ConflAcc':>10} {'Plural':>8}"
print(header)
print("-"*55)
for model in ["vanilla_llm", "rag", "symbolic", "hybrid"]:
    d = s.get(model)
    if not d:
        continue
    n   = d["total_scenarios"]
    acc = f"{d['doctrinal_accuracy_mean']:.3f}" if d["doctrinal_accuracy_mean"] is not None else "  N/A"
    hal = f"{d['hallucination_rate_mean']:.3f}" if d["hallucination_rate_mean"] is not None else "  N/A"
    con = f"{d['conflict_detection_accuracy']:.3f}" if d["conflict_detection_accuracy"] is not None else "  N/A"
    plu = f"{d['pluralism_score_mean']:.3f}" if d["pluralism_score_mean"] is not None else "  N/A"
    print(f"{model:<16} {n:>3} {acc:>7} {hal:>8} {con:>10} {plu:>8}")

# Per-model run directories
print("\n" + "="*65)
print("ABLATION TABLE")
print("="*65)
rows = [
    ("A. Vanilla LLM", "❌", "❌", "❌", "✅", "vanilla_llm"),
    ("B. RAG",         "❌", "❌", "✅", "✅", "rag"),
    ("C. Symbolic",    "✅", "❌", "❌", "❌", "symbolic"),
    ("D. Hybrid",      "✅", "✅", "✅", "✅", "hybrid"),
]
print(f"{'Model':<20} {'KG':>3} {'Agt':>4} {'Ret':>4} {'LLM':>4} {'Acc':>7} {'Halluc':>8} {'Conflict':>9}")
print("-"*62)
for label, kg, agt, ret, llm, mkey in rows:
    d = s.get(mkey, {})
    acc = f"{d['doctrinal_accuracy_mean']:.3f}" if d.get("doctrinal_accuracy_mean") is not None else "  N/A"
    hal = f"{d['hallucination_rate_mean']:.3f}" if d.get("hallucination_rate_mean") is not None else "  N/A"
    con = f"{d['conflict_detection_accuracy']:.3f}" if d.get("conflict_detection_accuracy") is not None else "  N/A"
    print(f"{label:<20} {kg:>3} {agt:>4} {ret:>4} {llm:>4} {acc:>7} {hal:>8} {con:>9}")

# Latest hybrid run details
print("\n" + "="*65)
print("LATEST HYBRID RUN — PER-SCENARIO DETAIL")
print("="*65)
hyb = [x for x in r if x["model"] == "hybrid"]
hyb.sort(key=lambda x: x["scenario_id"])
for rec in hyb:
    sid   = rec["scenario_id"]
    q     = rec["question"]
    verd  = rec["output"].get("school_verdicts", {})
    cites = rec["output"].get("citations_claimed", [])
    err   = rec["output"].get("error", "")
    acc   = rec["metrics"]["doctrinal_accuracy"]
    cdet  = rec["output"].get("conflict_detected", False)
    cok   = rec["metrics"]["conflict_correct"]
    hal   = rec["metrics"]["hallucination_rate"]
    agents = rec["output"].get("agent_responses", {})
    synth  = rec["output"].get("response", "")

    print(f"\n[{sid}] {q[:70]}")
    if err:
        print(f"  ERROR: {err[:120]}")
    else:
        print(f"  School Verdicts : {verd}")
        print(f"  Ground Truth    : {rec['ground_truth']['expected_verdicts']}")
        print(f"  Accuracy={acc}  Halluc={hal}  ConflictCorrect={cok}")
        print(f"  Conflict detected: {cdet} (expected: {rec['ground_truth']['conflict_expected']})")
        print(f"  Citations: {cites}")
        print(f"  Synthesis (200c): {synth[:200]}")
        for school, text in agents.items():
            print(f"  [{school}] {str(text)[:120]}")

# School-level accuracy for hybrid
print("\n" + "="*65)
print("PER-SCHOOL ACCURACY (Hybrid)")
print("="*65)
from collections import defaultdict
school_stats = defaultdict(lambda: {"correct": 0, "total": 0})
for rec in hyb:
    gt = rec["ground_truth"]["expected_verdicts"]
    matches = rec["metrics"].get("school_matches", {})
    for sch, ok in matches.items():
        if gt.get(sch, "N") not in ("N", "X"):
            school_stats[sch]["total"] += 1
            if ok:
                school_stats[sch]["correct"] += 1

for sch in ["advaita", "dvaita", "vishishtadvaita", "mimamsa", "nyaya", "bhakti"]:
    d = school_stats[sch]
    t = d["total"]
    acc = (d["correct"] / t) if t > 0 else None
    acc_s = f"{acc:.3f}" if acc is not None else "N/A"
    print(f"  {sch:<20} correct={d['correct']}/{t}  acc={acc_s}")

print("\nDone.")
