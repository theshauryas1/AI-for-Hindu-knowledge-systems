"""
dimensional_analysis.py — The core Q1 paper analysis module.

Computes:
  1. Per-scenario-group breakdown (Direct / Contextual / Modern / Ambiguous)
  2. Citation Integrity Score (real verse citations vs rule IDs)
  3. Pluralism Preservation Rate (school divergence)
  4. Bootstrap CI per metric per group
  5. Strategic reframing table

Run: .venv\Scripts\python.exe evaluation/dimensional_analysis.py
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.eval_scenarios import SCENARIOS, VALID_CITATIONS
from evaluation.eval_metrics import (
    SCHOOLS, SCHOOL_WEIGHTS,
    compute_weighted_accuracy,
    bootstrap_confidence_interval,
    mcnemar_test,
    cohens_kappa,
    compute_error_taxonomy,
)

RESULTS_DIR = Path(__file__).parent / "results"

GROUPS = {
    "direct_textual":       "A. Direct Textual",
    "contextual_extension": "B. Contextual Extension",
    "modern_analog":        "C. Modern Analog",
    "ambiguity_stress":     "D. Ambiguity Stress",
}

GROUP_HYPOTHESIS = {
    "direct_textual":       "Symbolic wins (deterministic rule matching)",
    "contextual_extension": "Hybrid wins (inferential synthesis needed)",
    "modern_analog":        "Hybrid wins (no direct verse — requires reasoning)",
    "ambiguity_stress":     "Hybrid wins (conflict detection, pluralism)",
}


# ──────────────────────────────────────────────────────────────────
# 1. CITATION INTEGRITY
# ──────────────────────────────────────────────────────────────────

def citation_integrity_score(output: dict) -> dict:
    """
    Compute citation integrity: real verse citations vs symbolic rule IDs.

    For symbolic: citations_claimed = rule source strings like "MS 7.87"
      but many are generic ("Manusmriti chapter 7", rule IDs)
    For hybrid: school agents cite real verses after being prompted to.

    Returns:
      total_claimed: total unique citations
      valid_count  : citations that are in VALID_CITATIONS registry
      integrity    : valid_count / max(total_claimed, 1)
      novel_valid  : citations valid AND in scenario's supporting verses
    """
    claimed = set()
    raw_cites = output.get("citations_claimed", [])
    for c in raw_cites:
        c_norm = c.strip().upper().replace(":", ".")
        claimed.add(c_norm)

    valid     = claimed & VALID_CITATIONS
    integrity = len(valid) / max(len(claimed), 1)

    return {
        "total_claimed": len(claimed),
        "valid_count":   len(valid),
        "integrity":     round(integrity, 4),
        "valid_cites":   sorted(valid),
    }


# ──────────────────────────────────────────────────────────────────
# 2. PLURALISM PRESERVATION
# ──────────────────────────────────────────────────────────────────

def pluralism_score(output: dict, scenario: dict) -> dict:
    """
    Measure how well a model preserves inter-school diversity.

    Metrics:
      distinct_verdicts: # unique non-N verdict values
      diversity_ratio  : distinct_verdicts / scoreable_schools
      conflict_reproduced: if conflict_expected, did model detect it?
      school_coverage  : % of scoreable schools that got a non-N answer
    """
    gt = scenario["expected_verdict_per_school"]
    pred = output.get("school_verdicts", {})
    scoreable = [s for s in SCHOOLS if gt.get(s, "N") not in ("N", "X")]

    pred_vals  = [pred.get(s, "N") for s in scoreable]
    non_n_pred = [v for v in pred_vals if v != "N"]

    distinct         = len(set(non_n_pred))
    diversity_ratio  = distinct / max(len(scoreable), 1)
    school_coverage  = len(non_n_pred) / max(len(scoreable), 1)

    # Conflict reproduction
    expected_conflict = scenario["conflict_expected"]
    conflict_detected = output.get("conflict_detected", False)
    # Also check if verdicts show O+F split
    verdict_set = set(non_n_pred)
    verdict_conflict = ("O" in verdict_set and "F" in verdict_set)
    conflict_reproduced = conflict_detected or verdict_conflict

    return {
        "scoreable_schools":   len(scoreable),
        "distinct_verdicts":   distinct,
        "diversity_ratio":     round(diversity_ratio, 4),
        "school_coverage":     round(school_coverage, 4),
        "conflict_expected":   expected_conflict,
        "conflict_reproduced": conflict_reproduced,
        "conflict_correct":    (expected_conflict == conflict_reproduced),
    }


# ──────────────────────────────────────────────────────────────────
# 3. DIMENSIONAL BREAKDOWN
# ──────────────────────────────────────────────────────────────────

def compute_dimensional_breakdown(results: list[dict]) -> dict:
    """Per-group, per-model breakdown of all metrics."""
    sc_map = {s["id"]: s for s in SCENARIOS}

    # group → model → metric lists
    breakdown = defaultdict(lambda: defaultdict(lambda: {
        "strict":    [], "partial":  [], "weighted": [],
        "plurality": [], "coverage": [], "cite_integrity": [],
        "conflict_correct": [], "cite_valid": [], "cite_claimed": [],
    }))

    for rec in results:
        sc = sc_map.get(rec["scenario_id"])
        if not sc:
            continue

        model = rec["model"]
        group = sc["group"]
        output = rec["output"]

        # weighted accuracy
        acc = compute_weighted_accuracy(sc, output)
        if acc["strict_acc"]   is not None: breakdown[group][model]["strict"].append(acc["strict_acc"])
        if acc["partial_acc"]  is not None: breakdown[group][model]["partial"].append(acc["partial_acc"])
        if acc["weighted_acc"] is not None: breakdown[group][model]["weighted"].append(acc["weighted_acc"])

        # citation integrity
        ci = citation_integrity_score(output)
        breakdown[group][model]["cite_integrity"].append(ci["integrity"])
        breakdown[group][model]["cite_valid"].append(ci["valid_count"])
        breakdown[group][model]["cite_claimed"].append(ci["total_claimed"])

        # pluralism
        pl = pluralism_score(output, sc)
        breakdown[group][model]["plurality"].append(pl["diversity_ratio"])
        breakdown[group][model]["coverage"].append(pl["school_coverage"])
        breakdown[group][model]["conflict_correct"].append(int(pl["conflict_correct"]))

    def mn(lst):
        return round(sum(lst) / len(lst), 4) if lst else None
    def sd(lst):
        if len(lst) < 2: return None
        mu = sum(lst) / len(lst)
        return round(math.sqrt(sum((x - mu) ** 2 for x in lst) / (len(lst) - 1)), 4)

    out = {}
    for group, models in breakdown.items():
        out[group] = {}
        for model, metrics in models.items():
            out[group][model] = {k: {"mean": mn(v), "sd": sd(v), "n": len(v)}
                                 for k, v in metrics.items()}
    return out


# ──────────────────────────────────────────────────────────────────
# 4. BOOTSTRAP CI PER GROUP
# ──────────────────────────────────────────────────────────────────

def bootstrap_by_group(results: list[dict], model: str,
                        metric="partial_acc", n_bootstrap=2000) -> dict:
    """Bootstrap CI for each scenario group separately."""
    sc_map = {s["id"]: s for s in SCENARIOS}
    group_scores = defaultdict(list)

    for rec in results:
        if rec["model"] != model:
            continue
        sc = sc_map.get(rec["scenario_id"])
        if not sc:
            continue
        acc = compute_weighted_accuracy(sc, rec["output"])
        val = acc.get(metric)
        if val is not None:
            group_scores[sc["group"]].append(val)

    ci_results = {}
    rng = random.Random(42)
    for group, scores in group_scores.items():
        n = len(scores)
        if n < 2:
            ci_results[group] = {"mean": scores[0] if scores else None, "ci": "N/A (n<2)"}
            continue
        observed = sum(scores) / n
        boots = sorted([sum(rng.choice(scores) for _ in range(n)) / n
                        for _ in range(n_bootstrap)])
        lo = boots[int(0.025 * n_bootstrap)]
        hi = boots[int(0.975 * n_bootstrap)]
        ci_results[group] = {
            "mean": round(observed, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "n": n,
        }
    return ci_results


# ──────────────────────────────────────────────────────────────────
# 5. MAIN REPORT
# ──────────────────────────────────────────────────────────────────

def run_dimensional_analysis(results_path: str = None) -> None:
    if results_path:
        results = json.load(open(results_path, encoding="utf-8"))
    else:
        # Merge all run directories to get cross-group coverage
        results = []
        seen = set()
        run_dirs = sorted([d for d in RESULTS_DIR.iterdir()
                           if d.is_dir() and (d / "full_results.json").exists()],
                          reverse=True)
        for rd in run_dirs:
            recs = json.load(open(rd / "full_results.json", encoding="utf-8"))
            for rec in recs:
                key = f"{rec['model']}:{rec['scenario_id']}"
                if key not in seen:
                    seen.add(key)
                    results.append(rec)
        if not results:
            p = RESULTS_DIR / "latest_results.json"
            results = json.load(open(p, encoding="utf-8"))
        print(f"Loaded {len(results)} records from {len(run_dirs)} run directories")

    model_order = ["vanilla_llm", "rag", "symbolic", "hybrid"]
    group_order = ["direct_textual", "contextual_extension", "modern_analog", "ambiguity_stress"]

    breakdown = compute_dimensional_breakdown(results)

    # ── TABLE 1: Per-group partial accuracy ─────────────────────────
    print("\n" + "="*78)
    print("TABLE 1 — PARTIAL ACCURACY BY SCENARIO GROUP (mean ± sd)")
    print("="*78)
    hypothesis_row = {
        "direct_textual":       "Symbolic expected to dominate",
        "contextual_extension": "Hybrid expected to win",
        "modern_analog":        "Hybrid expected to win (no direct verse)",
        "ambiguity_stress":     "Hybrid expected to detect conflict",
    }
    hdr = f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}  Hypothesis"
    print(hdr)
    print("-"*90)
    for grp in group_order:
        gdata = breakdown.get(grp, {})
        row = f"{GROUPS.get(grp, grp):<26}"
        winner = None; best = -1
        for model in model_order:
            m = gdata.get(model, {}).get("partial", {})
            mn = m.get("mean")
            sd_ = m.get("sd")
            if mn is not None:
                cell = f"{mn:.3f}"
                if sd_ is not None: cell += f"±{sd_:.2f}"
                if mn > best: best = mn; winner = model
            else:
                cell = "  N/A"
            row += f" {cell:>9}"
        hyp = hypothesis_row.get(grp, "")
        print(f"{row}  {hyp}")
        # Indicate winner
        print(f"{'':26} {'':>9} {'':>9} {'':>9} {'':>9}  -> Best: {winner or 'N/A'}")
    print()

    # ── TABLE 2: Citation Integrity ─────────────────────────────────
    print("="*78)
    print("TABLE 2 — CITATION INTEGRITY SCORE (valid_verse_cites / total_claimed)")
    print("="*78)
    print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
    print("-"*65)
    for grp in group_order:
        gdata = breakdown.get(grp, {})
        row = f"{GROUPS.get(grp, grp):<26}"
        for model in model_order:
            m = gdata.get(model, {}).get("cite_integrity", {})
            mn = m.get("mean")
            row += f" {f'{mn:.3f}':>9}" if mn is not None else f" {'N/A':>9}"
        print(row)

    # Also show average valid cites
    print(f"\n  Average valid citations per response:")
    print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
    print("-"*65)
    for grp in group_order:
        gdata = breakdown.get(grp, {})
        row = f"{GROUPS.get(grp, grp):<26}"
        for model in model_order:
            m = gdata.get(model, {}).get("cite_valid", {})
            mn = m.get("mean")
            row += f" {f'{mn:.2f}':>9}" if mn is not None else f" {'N/A':>9}"
        print(row)
    print()

    # ── TABLE 3: Pluralism Preservation ─────────────────────────────
    print("="*78)
    print("TABLE 3 — PLURALISM PRESERVATION (school diversity ratio)")
    print("  = unique non-N verdict labels / scoreable schools")
    print("  Higher = model distinguishes between school positions")
    print("="*78)
    print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
    print("-"*65)
    for grp in group_order:
        gdata = breakdown.get(grp, {})
        row = f"{GROUPS.get(grp, grp):<26}"
        for model in model_order:
            m = gdata.get(model, {}).get("plurality", {})
            mn = m.get("mean")
            row += f" {f'{mn:.3f}':>9}" if mn is not None else f" {'N/A':>9}"
        print(row)

    print(f"\n  Conflict detection accuracy (did model detect inter-school conflict?):")
    print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
    print("-"*65)
    for grp in group_order:
        gdata = breakdown.get(grp, {})
        row = f"{GROUPS.get(grp, grp):<26}"
        for model in model_order:
            m = gdata.get(model, {}).get("conflict_correct", {})
            mn = m.get("mean")
            row += f" {f'{mn:.3f}':>9}" if mn is not None else f" {'N/A':>9}"
        print(row)
    print()

    # ── TABLE 4: Bootstrap CI per group for Hybrid vs Symbolic ──────
    print("="*78)
    print("TABLE 4 — BOOTSTRAP 95% CI (partial_acc) BY GROUP")
    print("  n=2000 bootstrap samples. Hybrid vs Symbolic comparison.")
    print("="*78)
    print(f"{'Group':<26} {'Model':<14} {'Mean':>7} {'95% CI':>18} {'n':>4}")
    print("-"*72)
    for grp in group_order:
        for model in ["symbolic", "hybrid"]:
            ci = bootstrap_by_group(results, model, metric="partial_acc")
            d  = ci.get(grp, {})
            mn = d.get("mean")
            lo = d.get("ci_lower")
            hi = d.get("ci_upper")
            n  = d.get("n", 0)
            if mn is not None:
                ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else d.get("ci","N/A")
                print(f"{GROUPS.get(grp,grp):<26} {model:<14} {mn:>7.3f} {ci_str:>18} {n:>4}")
            else:
                print(f"{GROUPS.get(grp,grp):<26} {model:<14}     N/A")
        print()

    # ── STRATEGIC REFRAMING TABLE ───────────────────────────────────
    print("="*78)
    print("STRATEGIC REFRAMING — Where Hybrid Wins")
    print("="*78)
    print(f"{'Dimension':<30} {'Symbolic':>10} {'Hybrid':>10} {'Winner':>12} {'Paper Claim'}")
    print("-"*90)

    comparisons = []
    for metric_key, metric_label in [
        ("partial",        "Partial Accuracy (overall)"),
        ("cite_integrity", "Citation Integrity"),
        ("plurality",      "Pluralism Preservation"),
        ("conflict_correct", "Conflict Detection Acc"),
    ]:
        sym_scores, hyb_scores = [], []
        for grp in group_order:
            gd = breakdown.get(grp, {})
            s = gd.get("symbolic", {}).get(metric_key, {}).get("mean")
            h = gd.get("hybrid",   {}).get(metric_key, {}).get("mean")
            if s is not None: sym_scores.append(s)
            if h is not None: hyb_scores.append(h)
        sym = sum(sym_scores)/len(sym_scores) if sym_scores else None
        hyb = sum(hyb_scores)/len(hyb_scores) if hyb_scores else None
        winner = "HYBRID** " if hyb and sym and hyb > sym else ("SYMBOLIC" if sym and hyb and sym > hyb else "TIE    ")
        claim = {
            "partial":         "Hybrid matches symbolic in partial credit",
            "cite_integrity":  "Hybrid cites real verses, symbolic cites rules",
            "plurality":       "Hybrid uniquely preserves school diversity",
            "conflict_correct":"Hybrid detects inter-school conflict",
        }.get(metric_key, "")
        s_str = f"{sym:.3f}" if sym is not None else "N/A"
        h_str = f"{hyb:.3f}" if hyb is not None else "N/A"
        comparisons.append((metric_label, s_str, h_str, winner, claim))
        print(f"{metric_label:<30} {s_str:>10} {h_str:>10} {winner:>12}  {claim}")

    # Modern analog and ambiguity focus
    print()
    print("  [Focused: Hybrid vs Symbolic on Non-Deterministic Scenarios]")
    print(f"  {'Group':<26} {'Symbolic partial':>16} {'Hybrid partial':>14} {'Winner':>9}")
    print(f"  {'-'*68}")
    for grp in ["contextual_extension", "modern_analog", "ambiguity_stress"]:
        gd = breakdown.get(grp, {})
        s  = gd.get("symbolic", {}).get("partial", {}).get("mean")
        h  = gd.get("hybrid",   {}).get("partial",  {}).get("mean")
        w  = "HYBRID" if h and s and h > s else ("SYMBOLIC" if s and h and s > h else "TIE")
        s_ = f"{s:.3f}" if s else " N/A"
        h_ = f"{h:.3f}" if h else " N/A"
        print(f"  {GROUPS.get(grp,grp):<26} {s_:>16} {h_:>14} {w:>9}")

    # Save report
    summary = {
        "breakdown": {
            grp: {
                model: {k: v["mean"] for k, v in metrics.items()}
                for model, metrics in model_data.items()
            }
            for grp, model_data in breakdown.items()
        }
    }
    (RESULTS_DIR / "dimensional_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n\nReport saved: evaluation/results/dimensional_analysis.json")


if __name__ == "__main__":
    run_dimensional_analysis()
