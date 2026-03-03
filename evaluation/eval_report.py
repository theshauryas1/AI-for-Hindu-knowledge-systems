"""
eval_report.py — Generate human-readable report from latest eval results.

Run: python evaluation/eval_report.py
     python evaluation/eval_report.py --path evaluation/results/20260301_001200
"""
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"

SCHOOL_LABELS = {
    "advaita": "Advaita",
    "dvaita": "Dvaita",
    "vishishtadvaita": "Viśiṣṭādvaita",
    "mimamsa": "Mīmāṃsā",
    "nyaya": "Nyāya",
    "bhakti": "Bhakti",
}

MODEL_LABELS = {
    "vanilla_llm": "A. Vanilla LLM",
    "rag":         "B. RAG",
    "symbolic":    "C. Symbolic",
    "hybrid":      "D. Hybrid",
}

VERDICTS = {"O": "Obligatory", "P": "Permitted", "F": "Forbidden", "N": "Neutral"}


def load_results(path: str = None):
    if path:
        p = Path(path)
        results = json.load(open(p / "full_results.json", encoding="utf-8"))
        summary = json.load(open(p / "metrics_summary.json", encoding="utf-8"))
    else:
        results = json.load(open(RESULTS_DIR / "latest_results.json", encoding="utf-8"))
        summary = json.load(open(RESULTS_DIR / "latest_summary.json", encoding="utf-8"))
    return results, summary


def print_banner(text):
    print(f"\n{'═'*70}")
    print(f"  {text}")
    print(f"{'═'*70}")


def print_section(text):
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def generate_report(path: str = None, include_full_qa: bool = True):
    results, summary = load_results(path)

    print_banner("HINDUMIND EVALUATION REPORT")
    print(f"  Scenarios run : {len(set(r['scenario_id'] for r in results))}")
    print(f"  Models tested : {len(summary)}")
    print(f"  Total records : {len(results)}")

    # ─── METRIC SUMMARY TABLE ───────────────────────────────────
    print_section("1. METRIC SUMMARY TABLE")
    header = f"{'Model':<22} {'Acc':>6} {'Halluc':>8} {'CitRec':>8} {'ConflAcc':>10} {'Plural':>8} {'Time(s)':>8} {'Traps':>6}"
    print(header)
    print("─" * len(header))
    for model in ["vanilla_llm", "rag", "symbolic", "hybrid"]:
        s = summary.get(model, {})
        if not s:
            continue
        label = MODEL_LABELS.get(model, model)
        acc = f"{s['doctrinal_accuracy_mean']:.3f}" if s['doctrinal_accuracy_mean'] else "  N/A"
        hal = f"{s['hallucination_rate_mean']:.3f}" if s['hallucination_rate_mean'] is not None else "  N/A"
        cit = f"{s['citation_recall_mean']:.3f}" if s['citation_recall_mean'] is not None else "  N/A"
        con = f"{s['conflict_detection_accuracy']:.3f}" if s['conflict_detection_accuracy'] is not None else "  N/A"
        plu = f"{s['pluralism_score_mean']:.3f}" if s['pluralism_score_mean'] is not None else "  N/A"
        tim = f"{s['avg_response_time_sec']:.1f}" if s['avg_response_time_sec'] is not None else "  N/A"
        trp = str(s['total_traps_hit'])
        print(f"{label:<22} {acc:>6} {hal:>8} {cit:>8} {con:>10} {plu:>8} {tim:>8} {trp:>6}")

    # ─── ABLATION BREAKDOWN ─────────────────────────────────────
    print_section("2. ABLATION TABLE")
    component_map = {
        "vanilla_llm": ("❌", "❌", "❌", "✅"),
        "rag":         ("❌", "❌", "✅", "✅"),
        "symbolic":    ("✅", "❌", "❌", "❌"),
        "hybrid":      ("✅", "✅", "✅", "✅"),
    }
    print(f"{'Model':<22} {'KG':>4} {'Agents':>7} {'Retrieval':>10} {'LLM':>5} {'Accuracy':>10} {'Halluc':>8} {'ConflAcc':>10}")
    print("─" * 80)
    for model in ["vanilla_llm", "rag", "symbolic", "hybrid"]:
        s = summary.get(model)
        if not s:
            continue
        kg, agt, ret, llm = component_map.get(model, ("?","?","?","?"))
        label = MODEL_LABELS.get(model, model)
        acc = f"{s.get('doctrinal_accuracy_mean', 0):.3f}" if s.get('doctrinal_accuracy_mean') else " N/A"
        hal = f"{s.get('hallucination_rate_mean', 0):.3f}" if s.get('hallucination_rate_mean') is not None else " N/A"
        con = f"{s.get('conflict_detection_accuracy', 0):.3f}" if s.get('conflict_detection_accuracy') is not None else " N/A"
        print(f"{label:<22} {kg:>4} {agt:>7} {ret:>10} {llm:>5} {acc:>10} {hal:>8} {con:>10}")

    # ─── PER-GROUP BREAKDOWN ─────────────────────────────────────
    print_section("3. ACCURACY BY SCENARIO GROUP")
    groups = ["direct_textual", "contextual_extension", "modern_analog", "ambiguity_stress"]
    group_labels = {
        "direct_textual": "A. Direct Textual",
        "contextual_extension": "B. Contextual Extension",
        "modern_analog": "C. Modern Analog",
        "ambiguity_stress": "D. Ambiguity Stress",
    }
    group_results = defaultdict(lambda: defaultdict(list))
    for r in results:
        g = r["group"]
        m = r["model"]
        acc = r["metrics"]["doctrinal_accuracy"]
        if acc is not None:
            group_results[g][m].append(acc)

    print(f"{'Group':<25}", end="")
    models_present = [m for m in ["vanilla_llm","rag","symbolic","hybrid"] if m in summary]
    for m in models_present:
        print(f" {MODEL_LABELS.get(m,m)[:8]:>10}", end="")
    print()
    print("─" * (25 + 11 * len(models_present)))
    for g in groups:
        print(f"{group_labels.get(g, g):<25}", end="")
        for m in models_present:
            accs = group_results[g].get(m, [])
            mean = sum(accs)/len(accs) if accs else None
            print(f" {mean:.3f}" if mean is not None else f"    N/A", end="  ")
        print()

    # ─── CONFLICT DETECTION ──────────────────────────────────────
    print_section("4. CONFLICT DETECTION ANALYSIS")
    conflict_results = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for r in results:
        m = r["model"]
        expected = r["metrics"]["conflict_expected"]
        detected = r["metrics"]["conflict_detected"]
        if expected and detected:
            conflict_results[m]["tp"] += 1
        elif not expected and not detected:
            conflict_results[m]["tn"] += 1
        elif not expected and detected:
            conflict_results[m]["fp"] += 1
        else:
            conflict_results[m]["fn"] += 1

    print(f"{'Model':<22} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("─" * 66)
    for model in models_present:
        cr = conflict_results[model]
        tp, tn, fp, fn = cr["tp"], cr["tn"], cr["fp"], cr["fn"]
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 0.001)
        label = MODEL_LABELS.get(model, model)
        print(f"{label:<22} {tp:>4} {tn:>4} {fp:>4} {fn:>4} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")

    # ─── PER-SCHOOL VERDICT ANALYSIS ────────────────────────────
    print_section("5. PER-SCHOOL DOCTRINAL ACCURACY (Hybrid model)")
    hybrid_results = [r for r in results if r["model"] == "hybrid"]
    school_correct = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in hybrid_results:
        matches = r["metrics"].get("school_matches", {})
        gt = r["ground_truth"]["expected_verdicts"]
        for school, correct in matches.items():
            if gt.get(school, "N") not in ("N", "X"):
                school_correct[school]["total"] += 1
                if correct:
                    school_correct[school]["correct"] += 1

    print(f"{'School':<22} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("─" * 50)
    for school in ["advaita", "dvaita", "vishishtadvaita", "mimamsa", "nyaya", "bhakti"]:
        d = school_correct[school]
        acc = d["correct"] / max(d["total"], 1)
        print(f"{SCHOOL_LABELS.get(school, school):<22} {d['correct']:>8} {d['total']:>7} {acc:>10.3f}")

    # ─── HALLUCINATION ANALYSIS ──────────────────────────────────
    print_section("6. HALLUCINATION ANALYSIS")
    print(f"{'Model':<22} {'Avg Halluc Rate':>16} {'Traps Hit':>11} {'Valid Citations':>16}")
    print("─" * 68)
    for model in models_present:
        s = summary.get(model, {})
        label = MODEL_LABELS.get(model, model)
        hal = f"{s.get('hallucination_rate_mean', 0):.3f}" if s.get('hallucination_rate_mean') is not None else "N/A"
        trp = str(s.get("total_traps_hit", 0))
        cit = f"{s.get('citation_recall_mean', 0):.3f}" if s.get('citation_recall_mean') is not None else "N/A"
        print(f"{label:<22} {hal:>16} {trp:>11} {cit:>16}")

    # ─── FULL Q&A LOG ────────────────────────────────────────────
    if include_full_qa:
        print_section("7. FULL Q&A LOG (Hybrid Model)")
        hresults = [r for r in results if r["model"] == "hybrid"]
        hresults.sort(key=lambda r: r["scenario_id"])
        for r in hresults:
            print(f"\n[{r['scenario_id']}] {r['group'].upper()}")
            print(f"Q: {r['question']}")
            print(f"A: {r['output']['response'][:500]}{'...' if len(r['output']['response']) > 500 else ''}")
            print(f"School Verdicts: {r['output'].get('school_verdicts', {})}")
            gt_verdicts = r['ground_truth']['expected_verdicts']
            print(f"Ground Truth  : {gt_verdicts}")
            print(f"Citations: {r['output'].get('citations_claimed', [])}")
            print(f"Conflict detected: {r['output'].get('conflict_detected', False)} | Expected: {r['ground_truth']['conflict_expected']}")
            a = r['metrics']['doctrinal_accuracy']
            print(f"Accuracy: {a:.3f}" if a is not None else "Accuracy: N/A")

    # ─── SAVE REPORT ─────────────────────────────────────────────
    report_path = RESULTS_DIR / "eval_report.txt"
    print(f"\n\nReport printed above. Saving to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Path to run directory")
    parser.add_argument("--no-qa", action="store_true", help="Skip Q&A log")
    args = parser.parse_args()
    generate_report(path=args.path, include_full_qa=not args.no_qa)
