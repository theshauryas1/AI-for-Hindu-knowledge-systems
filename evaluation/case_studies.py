"""
case_studies.py — Extract 3-5 detailed case studies from run logs for Q1 paper.

Selects scenarios where Hybrid and Symbolic diverge most strongly,
and produces structured case study writeups showing:
- The question
- Hybrid reasoning (per school)
- Symbolic output
- Ground truth
- Why hybrid got it right / wrong
- Error type

Run: .venv\Scripts\python.exe evaluation/case_studies.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.eval_scenarios import SCENARIOS
from evaluation.eval_metrics import compute_weighted_accuracy

RESULTS_DIR = Path(__file__).parent / "results"
SCHOOLS = ["advaita", "dvaita", "vishishtadvaita", "mimamsa", "nyaya", "bhakti"]

GROUPS = {
    "direct_textual":       "A. Direct Textual",
    "contextual_extension": "B. Contextual Extension",
    "modern_analog":        "C. Modern Analog",
    "ambiguity_stress":     "D. Ambiguity Stress",
}


def load_all_results():
    records = {}
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
                m = rec["model"]
                sid = rec["scenario_id"]
                records.setdefault(sid, {})[m] = rec
    return records


def select_case_studies(records):
    """
    Select cases where Hybrid and Symbolic diverge most — both hits and misses.
    Strategy: pick 1 from each group where delta is largest.
    """
    sc_map = {s["id"]: s for s in SCENARIOS}
    cases = []

    for sid, model_recs in records.items():
        sc = sc_map.get(sid)
        if not sc:
            continue
        if "hybrid" not in model_recs or "symbolic" not in model_recs:
            continue

        h_acc = compute_weighted_accuracy(sc, model_recs["hybrid"]["output"])
        s_acc = compute_weighted_accuracy(sc, model_recs["symbolic"]["output"])

        h_partial = h_acc.get("partial_acc") or 0
        s_partial = s_acc.get("partial_acc") or 0
        delta = h_partial - s_partial

        cases.append({
            "scenario_id": sid,
            "group": sc["group"],
            "question": sc["question"],
            "delta": delta,
            "hybrid_partial": h_partial,
            "symbolic_partial": s_partial,
            "hybrid_rec": model_recs["hybrid"],
            "symbolic_rec": model_recs["symbolic"],
            "scenario": sc,
        })

    # Pick best hybrid win per group
    selected = []
    for grp in GROUPS:
        grp_cases = [c for c in cases if c["group"] == grp]
        if grp_cases:
            best = max(grp_cases, key=lambda x: x["delta"])
            selected.append(best)
    # Also pick one where symbolic won over hybrid
    worst_hybrid = min(cases, key=lambda x: x["delta"])
    if worst_hybrid not in selected:
        selected.append(worst_hybrid)

    return selected


def format_case_study(case: dict, index: int) -> str:
    sc = case["scenario"]
    h_rec = case["hybrid_rec"]
    s_rec = case["symbolic_rec"]

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"CASE STUDY {index}: {sc['id']} — {GROUPS.get(sc['group'], sc['group'])}")
    lines.append(f"{'='*70}")
    lines.append(f"\nQuestion:\n  {sc['question']}\n")

    lines.append(f"Performance:")
    lines.append(f"  Hybrid  partial_acc = {case['hybrid_partial']:.3f}")
    lines.append(f"  Symbolic partial_acc = {case['symbolic_partial']:.3f}")
    lines.append(f"  Delta (Hybrid - Symbolic) = {case['delta']:+.3f}\n")

    lines.append("Ground Truth (per school):")
    gt = sc.get("expected_verdict_per_school", {})
    for school in SCHOOLS:
        v = gt.get(school, "N")
        lines.append(f"  {school:<20} {v}")
    lines.append(f"  Conflict expected: {sc.get('conflict_expected', False)}")
    lines.append(f"  Supporting verses: {', '.join(sc.get('supporting_verses', []))}\n")

    lines.append("Hybrid Output (school agent verdicts):")
    h_verdicts = h_rec["output"].get("school_verdicts", {})
    for school in SCHOOLS:
        v = h_verdicts.get(school, "N")
        expected = gt.get(school, "N")
        match = "OK" if v == expected else ("?" if expected == "X" else "WRONG")
        lines.append(f"  {school:<20} predicted={v}  expected={expected}  [{match}]")
    h_conflict = h_rec["output"].get("conflict_detected", False)
    lines.append(f"  Conflict detected: {h_conflict}")
    lines.append(f"  Citations: {h_rec['output'].get('citations_claimed', [])[:5]}")

    h_response = h_rec["output"].get("response", "")
    lines.append(f"\n  Response excerpt:\n    {h_response[:500].strip()}")

    lines.append("\nSymbolic Output:")
    s_verdicts = s_rec["output"].get("school_verdicts", {})
    for school in SCHOOLS:
        v = s_verdicts.get(school, "N")
        expected = gt.get(school, "N")
        match = "OK" if v == expected else ("?" if expected == "X" else "WRONG")
        lines.append(f"  {school:<20} predicted={v}  expected={expected}  [{match}]")
    s_conflict = s_rec["output"].get("conflict_detected", False)
    lines.append(f"  Conflict detected: {s_conflict}")
    lines.append(f"  Citations: {s_rec['output'].get('citations_claimed', [])[:5]}")

    lines.append("\nAnalysis:")
    delta = case["delta"]
    if delta > 0.3:
        lines.append(f"  >> HYBRID WINS (+{delta:.2f}): demonstrates superior reasoning on this scenario type.")
        if sc["group"] == "modern_analog":
            lines.append("     Likely because: no direct verse maps to this modern case — hybrid infers from principles.")
        elif sc["group"] == "ambiguity_stress":
            lines.append("     Likely because: inter-school conflict requires dynamic synthesis, not rule lookup.")
        elif sc["group"] == "contextual_extension":
            lines.append("     Likely because: contextual extension requires analogical reasoning beyond rule base.")
    elif delta < -0.2:
        lines.append(f"  >> SYMBOLIC WINS ({delta:.2f}): deterministic rule lookup outperformed LLM synthesis here.")
        lines.append("     Paper framing: symbolic wins on direct-textual, hybrid wins where rules are incomplete.")
    else:
        lines.append(f"  >> NEAR-TIE ({delta:+.2f}): both models handled this scenario similarly.")

    return "\n".join(lines)


def run_case_studies():
    records = load_all_results()
    selected = select_case_studies(records)

    output_lines = [
        "HinduMind Evaluation — Case Studies for Q1 Paper",
        "="*70,
        f"Selected {len(selected)} case studies showing Hybrid vs Symbolic divergence.",
        "Each case shows: question, ground truth, per-school verdicts, citations, analysis.",
        "",
    ]

    for i, case in enumerate(selected, 1):
        output_lines.append(format_case_study(case, i))

    report = "\n".join(output_lines)

    out_path = RESULTS_DIR / "case_studies.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n\nSaved: {out_path}")


if __name__ == "__main__":
    run_case_studies()
