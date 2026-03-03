"""
eval_metrics.py — Weighted, multi-label and partial-credit doctrinal metrics.

Replaces rigid O/P/F binary with:
  1. Partial correctness score  — verdicts on a spectrum
  2. Multi-label scoring        — essay can address multiple valid verdicts
  3. Weighted doctrinal align   — school importance weighting
  4. Statistical tests          — McNemar, Bootstrap CI, Cohen's kappa
  5. Error taxonomy             — classify WHY each error occurred

Run: .venv\Scripts\python.exe evaluation/eval_metrics.py
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ───────────────────────────────────────────────────────────────
# 1. VERDICT COMPATIBILITY MATRIX (partial credit)
#
# Treating verdicts as an ordered spectrum:
#   F  ←  N  ←  P  ←  O
# Score = 1.0 if exact match, 0.5 if adjacent, 0.0 if opposite
# ───────────────────────────────────────────────────────────────

VERDICT_ORDER = {"F": 0, "N": 1, "P": 2, "O": 3}
VERDICT_PARTIAL = {
    ("O", "O"): 1.0, ("P", "P"): 1.0, ("F", "F"): 1.0, ("N", "N"): 1.0,
    ("O", "P"): 0.5, ("P", "O"): 0.5,  # adjacent
    ("O", "N"): 0.25,("N", "O"): 0.25,
    ("P", "N"): 0.5, ("N", "P"): 0.5,
    ("P", "F"): 0.25,("F", "P"): 0.25,
    ("F", "N"): 0.5, ("N", "F"): 0.5,
    ("O", "F"): 0.0, ("F", "O"): 0.0,  # opposite — no credit
}

# School theological importance weights (for weighted accuracy)
# Reflects canonical authority and breadth of tradition
SCHOOL_WEIGHTS = {
    "advaita":         0.22,   # Shankara — dominant monist tradition
    "dvaita":          0.16,   # Madhva — strong individuality
    "vishishtadvaita": 0.18,   # Ramanuja — broad bhakti-vedanta
    "mimamsa":         0.18,   # Ritual authority, Vedic hermeneutics
    "nyaya":           0.12,   # Logic/epistemology
    "bhakti":          0.14,   # Popular devotional stream
}

SCHOOLS = list(SCHOOL_WEIGHTS.keys())


def partial_verdict_score(pred: str, gt: str) -> float:
    """Score a single school verdict with partial credit."""
    if gt in ("N", "X"):   # not scoreable
        return None
    return VERDICT_PARTIAL.get((pred, gt), VERDICT_PARTIAL.get((gt, pred), 0.0))


def compute_weighted_accuracy(scenario: dict, output: dict) -> dict:
    """
    Compute three accuracy variants:
      strict_acc   : exact O/P/F/N match (original)
      partial_acc  : partial credit via VERDICT_PARTIAL matrix
      weighted_acc : partial credit × school importance weight
    """
    gt = scenario["expected_verdict_per_school"]
    pred = output.get("school_verdicts", {})

    strict_scores, partial_scores, weighted_scores = [], [], []

    for school in SCHOOLS:
        gt_v  = gt.get(school, "N")
        pr_v  = pred.get(school, "N")
        w     = SCHOOL_WEIGHTS[school]

        if gt_v in ("N", "X"):
            continue

        s_strict  = 1.0 if gt_v == pr_v else 0.0
        s_partial = VERDICT_PARTIAL.get((pr_v, gt_v), 0.0)

        strict_scores.append(s_strict)
        partial_scores.append(s_partial)
        weighted_scores.append(s_partial * w)

    def safe(lst, denom=None):
        if not lst: return None
        return round(sum(lst) / (denom or len(lst)), 4)

    total_weight = sum(
        SCHOOL_WEIGHTS[s] for s in SCHOOLS
        if gt.get(s, "N") not in ("N", "X")
    )

    return {
        "strict_acc":   safe(strict_scores),
        "partial_acc":  safe(partial_scores),
        "weighted_acc": safe(weighted_scores, total_weight) if total_weight > 0 else None,
        "school_detail": {
            s: {
                "gt":      gt.get(s, "N"),
                "pred":    pred.get(s, "N"),
                "partial": VERDICT_PARTIAL.get((pred.get(s,"N"), gt.get(s,"N")), 0.0),
                "weight":  SCHOOL_WEIGHTS[s],
            }
            for s in SCHOOLS if gt.get(s,"N") not in ("N","X")
        }
    }


# ───────────────────────────────────────────────────────────────
# 2. ERROR TAXONOMY
# Classify WHY a prediction failed
# ───────────────────────────────────────────────────────────────

ERROR_CATEGORIES = {
    "retrieval_failure":     "No relevant passages retrieved; response is generic",
    "role_misclassify":      "School agent classified question under wrong domain",
    "school_ambiguity":      "Scenario genuinely contested — multiple valid verdicts",
    "synthesis_distortion":  "LLM synthesis changed or blurred school position",
    "advaita_nondual_gap":   "Advaita's non-dual framing resists deontic O/P/F",
    "partial_match":         "Verdict direction correct but wrong modal operator",
    "correct":               "Verdict matches ground truth",
}


def classify_error(scenario: dict, output: dict, school: str) -> str:
    """Classify error type for a single school prediction."""
    gt_v   = scenario["expected_verdict_per_school"].get(school, "N")
    pred_v = output.get("school_verdicts", {}).get(school, "N")

    if gt_v in ("N", "X"):
        return "—"
    if gt_v == pred_v:
        return "correct"

    resp = output.get("agent_responses", {}).get(school, "")
    synth = output.get("response", "")
    citations = output.get("citations_claimed", [])

    # Advaita non-dual gap
    if school == "advaita" and pred_v == "N":
        return "advaita_nondual_gap"

    # No citations at all → retrieval failure
    if not citations and output.get("model") in ("hybrid", "rag"):
        return "retrieval_failure"

    # Response very short or generic
    if len(str(resp)) < 80:
        return "retrieval_failure"

    # O↔P or P↔F (adjacent) = partial match
    if abs(VERDICT_ORDER.get(pred_v, 1) - VERDICT_ORDER.get(gt_v, 1)) == 1:
        return "partial_match"

    # Conflict scenario → school ambiguity
    if scenario.get("conflict_expected") and school in scenario.get("conflict_schools", []):
        return "school_ambiguity"

    # Synthesis vs agent response differ
    if resp and synth:
        resp_v  = _quick_verdict(str(resp))
        synth_v = _quick_verdict(synth)
        if resp_v != synth_v and synth_v == pred_v:
            return "synthesis_distortion"

    return "role_misclassify"


def _quick_verdict(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ["must not","forbidden","prohibited","adharma","sinful"]):
        return "F"
    if any(w in lower for w in ["obligat","must perform","duty requires","shall"]):
        return "O"
    if any(w in lower for w in ["permitted","may be","allowed","context"]):
        return "P"
    return "N"


def compute_error_taxonomy(results: list[dict]) -> dict:
    """Build error taxonomy across all results."""
    taxonomy = defaultdict(lambda: defaultdict(int))  # model → error_type → count
    school_errors = defaultdict(lambda: defaultdict(int))  # school → error_type → count

    for rec in results:
        model   = rec["model"]
        scenario = {"expected_verdict_per_school": rec["ground_truth"]["expected_verdicts"],
                    "conflict_expected": rec["ground_truth"]["conflict_expected"],
                    "conflict_schools": rec["ground_truth"].get("conflict_schools", [])}
        output   = rec["output"]

        for school in SCHOOLS:
            err = classify_error(scenario, output, school)
            if err != "—":
                taxonomy[model][err] += 1
                school_errors[school][err] += 1

    return {
        "by_model": {m: dict(v) for m, v in taxonomy.items()},
        "by_school": {s: dict(v) for s, v in school_errors.items()},
    }


# ───────────────────────────────────────────────────────────────
# 3. STATISTICAL SIGNIFICANCE TESTS
# ───────────────────────────────────────────────────────────────

def mcnemar_test(results: list[dict], model_a: str, model_b: str,
                 school: str = None) -> dict:
    """
    McNemar's test of accuracy difference between two models.
    Uses strict accuracy (correct/incorrect) per scenario.

    Returns: chi2, p_value, interpretation
    """
    # Match scenarios that appear in both models
    recs_a = {r["scenario_id"]: r for r in results if r["model"] == model_a}
    recs_b = {r["scenario_id"]: r for r in results if r["model"] == model_b}
    common = set(recs_a) & set(recs_b)

    if len(common) < 5:
        return {"error": f"Only {len(common)} common scenarios — need ≥5"}

    b, c = 0, 0   # b: A correct, B wrong; c: A wrong, B correct
    for sid in common:
        ra = recs_a[sid];  rb = recs_b[sid]
        gt = ra["ground_truth"]["expected_verdicts"]

        if school:
            schools_eval = [school]
        else:
            schools_eval = [s for s in SCHOOLS if gt.get(s,"N") not in ("N","X")]

        correct_a = all(
            ra["output"].get("school_verdicts", {}).get(s,"N") == gt.get(s,"N")
            for s in schools_eval
        )
        correct_b = all(
            rb["output"].get("school_verdicts", {}).get(s,"N") == gt.get(s,"N")
            for s in schools_eval
        )

        if correct_a and not correct_b: b += 1
        if not correct_a and correct_b: c += 1

    n = b + c
    if n == 0:
        return {
            "chi2": 0.0, "p_value": 1.0, "b": 0, "c": 0,
            "significant_p05": False, "note": "No discordant pairs",
            "interpretation": "No discordant pairs — cannot determine significance"
        }

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Approximate p-value (chi2 df=1)
    # Using Wilson-Hilferty approximation for chi2 CDF
    p = _chi2_pvalue(chi2, df=1)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_scenarios": len(common),
        "b_only_a_correct": b,
        "c_only_b_correct": c,
        "chi2": round(chi2, 4),
        "p_value": round(p, 4),
        "significant_p05": p < 0.05,
        "interpretation": (
            f"{model_a} significantly better than {model_b}" if b > c and p < 0.05
            else f"{model_b} significantly better than {model_a}" if c > b and p < 0.05
            else "No significant difference"
        )
    }


def bootstrap_confidence_interval(results: list[dict], model: str,
                                   n_bootstrap: int = 2000,
                                   metric: str = "partial_acc",
                                   ci: float = 0.95) -> dict:
    """Bootstrap CI for a model's accuracy metric."""
    from evaluation.eval_scenarios import SCENARIOS

    recs = [r for r in results if r["model"] == model]
    if not recs:
        return {"error": f"No results for model {model}"}

    # Recompute weighted accuracy for each record
    scenario_map = {s["id"]: s for s in SCENARIOS}

    def get_score(r):
        sc = scenario_map.get(r["scenario_id"])
        if not sc:
            return None
        acc = compute_weighted_accuracy(sc, r["output"])
        return acc.get(metric)

    scores = [s for r in recs if (s := get_score(r)) is not None]
    if len(scores) < 3:
        return {"error": f"Too few valid scores: {len(scores)}"}

    n = len(scores)
    observed_mean = sum(scores) / n

    boot_means = []
    rng = random.Random(42)
    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo = boot_means[int((1 - ci) / 2 * n_bootstrap)]
    hi = boot_means[int((1 + ci) / 2 * n_bootstrap)]

    return {
        "model": model,
        "metric": metric,
        "n_scenarios": n,
        "mean": round(observed_mean, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def cohens_kappa(results: list[dict], model_a: str, model_b: str) -> dict:
    """
    Cohen's kappa measuring agreement between two models' verdicts
    (treats each school-scenario as one annotation pair).
    """
    recs_a = {r["scenario_id"]: r for r in results if r["model"] == model_a}
    recs_b = {r["scenario_id"]: r for r in results if r["model"] == model_b}
    common = set(recs_a) & set(recs_b)

    labels = ["O", "P", "F", "N"]
    # co-occurrence matrix
    comat = defaultdict(lambda: defaultdict(int))

    pairs = 0
    for sid in common:
        vtop_a = recs_a[sid]["output"].get("school_verdicts", {})
        vtop_b = recs_b[sid]["output"].get("school_verdicts", {})
        for school in SCHOOLS:
            a = vtop_a.get(school, "N")
            b = vtop_b.get(school, "N")
            comat[a][b] += 1
            pairs += 1

    if pairs == 0:
        return {"error": "No pairs to compare"}

    # observed agreement
    po = sum(comat[l][l] for l in labels) / pairs

    # expected agreement
    totals_a = {l: sum(comat[l].values()) / pairs for l in labels}
    totals_b = {l: sum(comat[al][l] for al in labels) / pairs for l in labels}
    pe = sum(totals_a[l] * totals_b[l] for l in labels)

    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0

    interp = (
        "Almost perfect" if kappa >= 0.81 else
        "Substantial" if kappa >= 0.61 else
        "Moderate" if kappa >= 0.41 else
        "Fair" if kappa >= 0.21 else
        "Slight" if kappa >= 0.01 else
        "Poor"
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_pairs": pairs,
        "p_observed": round(po, 4),
        "p_expected": round(pe, 4),
        "kappa": round(kappa, 4),
        "interpretation": interp,
    }


def _chi2_pvalue(chi2: float, df: int = 1) -> float:
    """Approximate p-value for chi-squared distribution using Wilson-Hilferty."""
    if chi2 <= 0:
        return 1.0
    # Wilson-Hilferty: standardize to normal
    z = (chi2 / df) ** (1/3)
    mu = 1 - 2 / (9 * df)
    sigma = math.sqrt(2 / (9 * df))
    z_norm = (z - mu) / sigma
    # Approximate normal CDF tail P(Z > z_norm)
    return _normal_sf(z_norm)


def _normal_sf(z: float) -> float:
    """Survival function P(Z > z) for standard normal (Abramowitz approximation)."""
    if z < 0:
        return 1.0 - _normal_sf(-z)
    t = 1 / (1 + 0.2316419 * z)
    poly = t * (0.319381530
                + t * (-0.356563782
                + t * (1.781477937
                + t * (-1.821255978
                + t * 1.330274429))))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return pdf * poly


# ───────────────────────────────────────────────────────────────
# MAIN — run on latest results
# ───────────────────────────────────────────────────────────────

def run_full_analysis(results_path: str = None) -> dict:
    from evaluation.eval_scenarios import SCENARIOS

    RESULTS_DIR = Path(__file__).parent / "results"
    if results_path:
        results = json.load(open(results_path, encoding="utf-8"))
    else:
        results = json.load(open(RESULTS_DIR / "latest_results.json", encoding="utf-8"))

    scenario_map = {s["id"]: s for s in SCENARIOS}

    # ── Weighted Accuracy per model ───────────────────────────
    print("\n" + "="*65)
    print("WEIGHTED ACCURACY METRICS")
    print("="*65)
    print(f"{'Model':<18} {'Strict':>8} {'Partial':>8} {'Weighted':>9}")
    print("-"*46)

    model_metrics: dict[str, dict] = defaultdict(lambda: {
        "strict": [], "partial": [], "weighted": []
    })

    enhanced_results = []
    for rec in results:
        sc = scenario_map.get(rec["scenario_id"])
        if not sc:
            enhanced_results.append(rec)
            continue
        acc = compute_weighted_accuracy(sc, rec["output"])
        rec["metrics"]["strict_acc"]   = acc["strict_acc"]
        rec["metrics"]["partial_acc"]  = acc["partial_acc"]
        rec["metrics"]["weighted_acc"] = acc["weighted_acc"]
        rec["metrics"]["school_detail"] = acc["school_detail"]
        enhanced_results.append(rec)

        m = rec["model"]
        if acc["strict_acc"]   is not None: model_metrics[m]["strict"].append(acc["strict_acc"])
        if acc["partial_acc"]  is not None: model_metrics[m]["partial"].append(acc["partial_acc"])
        if acc["weighted_acc"] is not None: model_metrics[m]["weighted"].append(acc["weighted_acc"])

    def mn(lst): return round(sum(lst)/len(lst), 3) if lst else None

    model_order = ["vanilla_llm", "rag", "symbolic", "hybrid"]
    for m in model_order:
        d = model_metrics[m]
        s = f"{mn(d['strict']):.3f}" if mn(d['strict']) is not None else "  N/A"
        p = f"{mn(d['partial']):.3f}" if mn(d['partial']) is not None else "  N/A"
        w = f"{mn(d['weighted']):.3f}" if mn(d['weighted']) is not None else "  N/A"
        print(f"{m:<18} {s:>8} {p:>8} {w:>9}")

    # ── Error Taxonomy ─────────────────────────────────────────
    print("\n" + "="*65)
    print("ERROR TAXONOMY BY MODEL")
    print("="*65)
    taxonomy = compute_error_taxonomy(enhanced_results)
    for model in model_order:
        d = taxonomy["by_model"].get(model, {})
        if not d: continue
        total = sum(d.values())
        print(f"\n  {model.upper()} (total errors+correct: {total})")
        for err, count in sorted(d.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"    {err:<28} {count:>3}  {pct:5.1f}%  {bar}")

    print("\n  Error Taxonomy by School (hybrid):")
    for school in SCHOOLS:
        d = taxonomy["by_school"].get(school, {})
        if not d: continue
        total = sum(d.values())
        errs  = [f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: -x[1]) if k != "correct"]
        print(f"    {school:<20} total={total} | {', '.join(errs[:3])}")

    # ── McNemar Tests ─────────────────────────────────────────
    print("\n" + "="*65)
    print("McNEMAR SIGNIFICANCE TESTS")
    print("="*65)
    pairs = [("hybrid","symbolic"), ("hybrid","vanilla_llm"),
             ("hybrid","rag"), ("symbolic","vanilla_llm")]
    for a, b in pairs:
        mc = mcnemar_test(enhanced_results, a, b)
        if "error" in mc:
            print(f"  {a} vs {b}: {mc['error']}")
        else:
            sig = "✓ SIGNIFICANT" if mc.get("significant_p05") else "✗ not significant"
            print(f"  {a} vs {b}: chi2={mc['chi2']}, p={mc['p_value']} → {sig}")
            print(f"    {mc['interpretation']}")

    # ── Bootstrap CI ───────────────────────────────────────────
    print("\n" + "="*65)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%, 2000 samples)")
    print("="*65)
    for m in model_order:
        ci = bootstrap_confidence_interval(enhanced_results, m, metric="partial_acc")
        if "error" in ci:
            print(f"  {m}: {ci['error']}")
        else:
            print(f"  {m:<18} partial_acc = {ci['mean']:.3f}  95% CI [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    # ── Cohen's Kappa ───────────────────────────────────────────
    print("\n" + "="*65)
    print("COHEN'S KAPPA (inter-model agreement)")
    print("="*65)
    kappa_pairs = [("hybrid","symbolic"), ("hybrid","vanilla_llm"), ("symbolic","vanilla_llm")]
    for a, b in kappa_pairs:
        kp = cohens_kappa(enhanced_results, a, b)
        if "error" in kp:
            print(f"  {a} vs {b}: {kp['error']}")
        else:
            print(f"  {a} vs {b}: κ={kp['kappa']:.3f} ({kp['interpretation']}, n={kp['n_pairs']} pairs)")

    # ── Save enhanced results ──────────────────────────────────
    RESULTS_DIR = Path(__file__).parent / "results"
    enhanced_path = RESULTS_DIR / "enhanced_results.json"
    enhanced_path.write_text(json.dumps(enhanced_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEnhanced results saved: {enhanced_path}")

    return {
        "model_metrics": {m: {k: mn(v) for k, v in d.items()} for m, d in model_metrics.items()},
        "taxonomy": taxonomy,
    }


if __name__ == "__main__":
    run_full_analysis()
