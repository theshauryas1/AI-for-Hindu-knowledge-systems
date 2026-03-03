"""
eval_harness.py — Main evaluation runner for HinduMind system.

Runs 4 model variants against 34 gold scenarios, records every parameter.

Model Variants:
  A. Vanilla LLM    — no KG, no retrieval, direct LLM call
  B. RAG            — embedding similarity retrieval, no KG agents
  C. Symbolic       — KG + rules only, no LLM synthesis
  D. Hybrid         — full system (KG + agents + retrieval + LLM)

Metrics computed per run:
  1. Doctrinal Accuracy     — per-school verdict match vs ground truth
  2. Citation Fidelity      — hallucination rate, valid citation rate
  3. Conflict Detection     — did system flag inter-school conflict?
  4. Pluralism Score        — distinct verdicts per scenario
  5. Response Completeness  — does answer address all valid concepts?

Output:
  evaluation/results/full_results.json        (raw per-scenario per-model)
  evaluation/results/metrics_summary.json     (aggregated metrics)
  evaluation/results/run_log.txt              (readable log of all Q&A)

Run: python evaluation/eval_harness.py
"""

import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────
# LOAD DEPENDENCIES
# ─────────────────────────────────────────────────────────────────

from evaluation.eval_scenarios import SCENARIOS, VALID_CITATIONS

# We'll load env vars for LLM
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCHOOLS = ["advaita", "dvaita", "vishishtadvaita", "mimamsa", "nyaya", "bhakti"]

# ─────────────────────────────────────────────────────────────────
# MODEL VARIANT A — VANILLA LLM
# Pure LLM call with no retrieval or KG
# ─────────────────────────────────────────────────────────────────

def run_vanilla_llm(question: str) -> dict:
    """Vanilla LLM: single prompt, no structured context."""
    from agents.llm_adapters import get_adapter
    adapter = get_adapter()

    system = (
        "You are a Hindu philosophy scholar. Answer the question directly. "
        "Cite specific verses where relevant. Be concise."
    )
    prompt = f"Question: {question}\n\nAnswer:"

    t0 = time.time()
    text = adapter.generate(prompt, system)
    elapsed = time.time() - t0

    return {
        "model": "vanilla_llm",
        "response": text,
        "citations_claimed": _extract_citations(text),
        "concepts_mentioned": [],
        "school_verdicts": {},            # Vanilla gives no per-school breakdown
        "conflict_detected": False,
        "conflict_reason": "",
        "retrieval_used": False,
        "kg_used": False,
        "elapsed_sec": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────
# MODEL VARIANT B — RAG (embedding similarity retrieval)
# ─────────────────────────────────────────────────────────────────

def run_rag(question: str) -> dict:
    """RAG: retrieve relevant passages from verse_db, then prompt LLM."""
    from agents.llm_adapters import get_adapter
    from kg.verse_retriever import VerseRetriever

    adapter = get_adapter()
    retriever = VerseRetriever()

    # Get top passages by keyword overlap (simulates RAG)
    keywords = [w.lower() for w in question.split() if len(w) > 4]
    passages = retriever.get_pdf_passages_for_concepts(
        concepts=keywords[:6],
        query=question,
        top_k=5
    )

    context_text = ""
    for p in passages:
        src = p.get("verse_id") or p.get("text_id", "")
        txt = p.get("text", "")[:300]
        context_text += f"[{src}] {txt}\n\n"

    system = (
        "You are a Hindu philosophy scholar. Use ONLY the retrieved passages below "
        "to answer. Do not invent citations not in the passages."
    )
    prompt = (
        f"Retrieved Passages:\n{context_text}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    t0 = time.time()
    text = adapter.generate(prompt, system)
    elapsed = time.time() - t0

    return {
        "model": "rag",
        "response": text,
        "citations_claimed": _extract_citations(text),
        "concepts_mentioned": [],
        "school_verdicts": {},
        "conflict_detected": False,
        "conflict_reason": "",
        "retrieval_used": True,
        "kg_used": False,
        "retrieved_passages": [p.get("verse_id", p.get("text_id", "")) for p in passages],
        "elapsed_sec": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────
# MODEL VARIANT C — SYMBOLIC (KG + rules only, no LLM)
# ─────────────────────────────────────────────────────────────────

def run_symbolic(question: str) -> dict:
    """Symbolic: KG lookup + rule base only. No LLM generation."""
    t0 = time.time()

    # Load extended rule base
    rules_path = Path(__file__).parent.parent / "dharma_engine" / "rules_extended.json"
    concept_idx_path = Path(__file__).parent.parent / "dharma_engine" / "rule_concept_index.json"

    verdict_per_school = {}
    matched_rules = []
    conflict_detected = False
    conflict_reason = ""

    if rules_path.exists():
        rules = json.load(open(rules_path, encoding="utf-8"))
        concept_idx = json.load(open(concept_idx_path, encoding="utf-8"))

        # Score rules by keyword relevance
        q_words = set(question.lower().split())
        scored = []
        for rule in rules:
            score = 0
            for w in q_words:
                if w in rule.get("action", "").lower():
                    score += 2
                if w in rule.get("text", "").lower():
                    score += 1
                for c in rule.get("concepts", []):
                    if w in c:
                        score += 3
            if score > 0:
                scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_rules = [r for _, r in scored[:5]]

        for rule in top_rules:
            matched_rules.append({
                "rule_id": rule["id"],
                "source": rule["source"],
                "operator": rule["operator"],
                "action": rule["action"],
                "score": scored[[r for _, r in scored].index(rule)][0] if rule in [r for _, r in scored] else 0
            })

        # Derive school verdicts from rule weights
        for rule in top_rules:
            op = rule["operator"]  # O, P, or F
            weights = rule.get("school_weights", {})
            for school, weight in weights.items():
                if school == "all":
                    for s in SCHOOLS:
                        if s not in verdict_per_school or weight > 0.8:
                            verdict_per_school[s] = op
                elif school in SCHOOLS and weight >= 0.7:
                    verdict_per_school[school] = op

        # Detect conflict
        ops = set(verdict_per_school.values())
        if "O" in ops and "F" in ops:
            conflict_detected = True
            o_schools = [s for s, v in verdict_per_school.items() if v == "O"]
            f_schools = [s for s, v in verdict_per_school.items() if v == "F"]
            conflict_reason = f"Schools {o_schools} hold O while {f_schools} hold F"

    # Build symbolic response text
    if matched_rules:
        response_parts = [f"Symbolic reasoning from rule base ({len(matched_rules)} rules matched):"]
        for r in matched_rules[:3]:
            op_text = {"O": "OBLIGATORY", "P": "PERMITTED", "F": "FORBIDDEN"}.get(r["operator"], r["operator"])
            response_parts.append(f"  [{r['source']}] Action is {op_text}: {r['action']}")
        response_parts.append(f"\nSchool verdicts: {verdict_per_school}")
        if conflict_detected:
            response_parts.append(f"CONFLICT DETECTED: {conflict_reason}")
        response_text = "\n".join(response_parts)
    else:
        response_text = "[SYMBOLIC] No matching rules found in rule base for this query."

    elapsed = time.time() - t0
    return {
        "model": "symbolic",
        "response": response_text,
        "citations_claimed": [r["source"] for r in matched_rules],
        "concepts_mentioned": [],
        "school_verdicts": verdict_per_school,
        "conflict_detected": conflict_detected,
        "conflict_reason": conflict_reason,
        "matched_rules": matched_rules,
        "retrieval_used": False,
        "kg_used": True,
        "elapsed_sec": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────
# MODEL VARIANT D — HYBRID (full HinduMind system)
# ─────────────────────────────────────────────────────────────────

def run_hybrid(question: str) -> dict:
    """Full HinduMind: ConceptMapper → 6 SchoolAgents → DeonticReasoner → MetaAgent."""
    t0 = time.time()

    try:
        from agents.meta_agent import MetaAgent
        from agents.school_agents import SchoolAgent
        from kg.hpo_graph import HPOGraph
        from kg.populate_kg import populate
        from kg.concept_mapper import ConceptMapper
        from kg.verse_retriever import VerseRetriever
        from agents.llm_adapters import get_adapter

        # 1. Build KG
        g = HPOGraph()
        populate(g)

        # 2. Map question to concepts
        mapper = ConceptMapper()
        mapped = mapper.map(question)
        concepts = (mapped.primary_concepts or ["dharma"])[:6]
        primary_concept = concepts[0]

        # 3. Retrieve passages
        retriever = VerseRetriever(g)
        try:
            retrieval_result = retriever.retrieve(question, concepts, top_k=5)
        except Exception:
            retrieval_result = None

        # 4. Run 6 school agents
        llm = get_adapter()
        school_responses = {}
        for school_id in SCHOOLS:
            try:
                agent = SchoolAgent(school_id=school_id, kg=g, llm=llm)
                school_responses[school_id] = agent.reason(query=question, concepts=concepts)
            except Exception as ae:
                school_responses[school_id] = {"school": school_id, "response": f"[err]{ae}"}
            time.sleep(0.5)

        # 5. Deontic reasoning (optional)
        deontic_verdict = None
        try:
            from dharma_engine.dharma_pipeline import DharmaPipeline
            deontic_verdict = DharmaPipeline(g).evaluate(question, concepts)
        except Exception:
            pass

        # 6. Meta synthesis
        meta = MetaAgent(g)
        result = meta.synthesize(
            query=question,
            concept=primary_concept,
            school_responses=school_responses,
            retrieval_result=retrieval_result,
            mapped_question=mapped,
            deontic_verdict=deontic_verdict,
        )

        elapsed = time.time() - t0
        synthesis_text = getattr(result, "synthesis", str(result))

        # Extract per-school verdicts — read structured verdict_label first
        verdicts = {}
        for school in SCHOOLS:
            resp = school_responses.get(school, {})
            if isinstance(resp, dict):
                # Use explicit verdict_label if available (from improved SchoolAgent)
                label = resp.get("verdict_label")
                if label and label in ("O", "P", "F", "N"):
                    verdicts[school] = label
                else:
                    verdicts[school] = _infer_verdict_from_text(
                        resp.get("response", "")
                    )
            else:
                verdicts[school] = _infer_verdict_from_text(str(resp))

        # Extract citations
        all_cites = []
        for school in SCHOOLS:
            resp = school_responses.get(school, {})
            raw = resp.get("response", "") if isinstance(resp, dict) else str(resp)
            all_cites.extend(_extract_citations(raw))
        all_cites.extend(_extract_citations(synthesis_text))

        # Conflict detection
        contradictions = getattr(result, "contradictions", []) or []
        conflict_detected = len(contradictions) > 0
        conflict_reason = "; ".join(
            str(c.get("description", c) if isinstance(c, dict) else c)
            for c in contradictions[:3]
        )
        if not conflict_detected:
            vset = set(v for v in verdicts.values() if v in ("O", "F"))
            if "O" in vset and "F" in vset:
                conflict_detected = True
                conflict_reason = "O/F split across schools"

        return {
            "model": "hybrid",
            "response": synthesis_text,
            "agent_responses": {
                s: (school_responses[s].get("response", "")
                    if isinstance(school_responses.get(s), dict)
                    else str(school_responses.get(s, "")))
                for s in SCHOOLS
            },
            "citations_claimed": list(set(all_cites)),
            "concepts_mentioned": _extract_concepts(synthesis_text),
            "school_verdicts": verdicts,
            "conflict_detected": conflict_detected,
            "conflict_reason": conflict_reason,
            "agreements": list(getattr(result, "agreements", [])),
            "primary_concept": primary_concept,
            "mapped_concepts": concepts,
            "retrieval_used": True,
            "kg_used": True,
            "confidence": getattr(result, "confidence", None),
            "elapsed_sec": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "model": "hybrid",
            "response": f"[ERROR] {e}\n{traceback.format_exc()}",
            "citations_claimed": [], "concepts_mentioned": [],
            "school_verdicts": {}, "conflict_detected": False,
            "conflict_reason": "", "retrieval_used": True, "kg_used": True,
            "elapsed_sec": round(elapsed, 2), "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────

def _extract_citations(text: str) -> list[str]:
    """Extract verse citation patterns from response text."""
    patterns = [
        r'BG\s+\d+[\.:]\d+',         # BG 2.47
        r'MS\s+\d+[\.:]\d+',         # MS 7.87
        r'YS\s+\d+[\.:]\d+',         # YS 2.30
        r'BAU\s+[\d\.]+',
        r'CU\s+[\d\.]+',
        r'MU\s+[\d\.]+',
        r'BS\s+[\d\.]+',
        r'AS\s+[\d\.]+',
        r'BP\s+[\d\.]+',
        r'NS\s+[\d\.]+',
        r'NBS\s+\d+',
        r'VC\s+\d+',
        r'Manusmriti\s+\d+[\.:]\d+',
        r'Bhagavad\s+Gita\s+\d+[\.:]\d+',
        r'Yoga\s+Sutras?\s+\d+[\.:]\d+',
        r'Arthashastra\s+\d+[\.:]\d+',
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, text, re.IGNORECASE))
    # Normalize
    return [c.strip().upper().replace(":", ".") for c in found]


def _extract_concepts(text: str) -> list[str]:
    """Find which KG concept terms appear in the response."""
    known = [
        "karma", "dharma", "moksha", "atman", "brahman", "maya", "ahimsa",
        "satya", "ahankara", "viveka", "vairagya", "jnana", "bhakti",
        "karma yoga", "nishkama", "sannyasa", "grihastha", "brahmachari",
        "kshatriya", "brahmin", "vaishya", "shudra", "yajna", "tapas",
        "daya", "kshama", "pratyaksha", "anumana", "vyapti", "sabda",
        "sattva", "rajas", "tamas", "pralaya", "avatar", "lila",
        "jiva", "ishvara", "prakriti", "purusha", "samadhi", "samsara",
    ]
    found = []
    lower_text = text.lower()
    for c in known:
        if c in lower_text:
            found.append(c)
    return found


def _infer_verdict_from_text(text: str) -> str:
    """Infer O/P/F/N from response text (priority: F > O > P > N)."""
    lower = text.lower()
    # Forbidden signals take priority over obligation
    if any(w in lower for w in ["must not", "forbidden", "prohibited", "not permitted",
                                  "violates dharma", "adharma", "sinful", "will incur sin",
                                  "should not", "ought not"]):
        return "F"
    if any(w in lower for w in ["obligat", "is the duty", "duty requires", "must perform",
                                  "required by", "shall perform", "shall fight",
                                  "is required", "one must"]):
        return "O"
    if any(w in lower for w in ["is permitted", "may perform", "is allowed", "is optional",
                                  "can be done", "context-dependent", "conditionally",
                                  "may be", "can be"]):
        return "P"
    # Weak fallbacks
    if any(w in lower for w in ["duty", "must", "required", "shall"]):
        return "O"
    if any(w in lower for w in ["permitted", "may", "can", "allowed"]):
        return "P"
    return "N"


def compute_metrics(scenario: dict, model_output: dict) -> dict:
    """Compute all metrics for a single (scenario, model_output) pair."""

    # ── Metric 1: Doctrinal Accuracy per school ──────────────────
    ground_truth = scenario["expected_verdict_per_school"]
    predicted = model_output.get("school_verdicts", {})

    school_matches = {}
    for school in SCHOOLS:
        gt = ground_truth.get(school, "N")
        pred = predicted.get(school, "N")
        school_matches[school] = (gt == pred)

    # Only count schools where GT is not N/X
    scoreable = {s: v for s, v in ground_truth.items() if v not in ("N", "X")}
    if scoreable:
        accuracy = sum(school_matches.get(s, False) for s in scoreable) / len(scoreable)
    else:
        accuracy = None

    # ── Metric 2: Citation Fidelity ──────────────────────────────
    claimed = set(model_output.get("citations_claimed", []))
    valid = set(scenario["supporting_verses"])
    trap_phrases = set(scenario.get("hallucination_traps", []))

    valid_claimed = claimed & VALID_CITATIONS   # subset of claimed that are real verses
    invalid_claimed = claimed - VALID_CITATIONS # verses claimed not in valid registry
    relevant_cited = claimed & valid             # cited and relevant to this scenario

    hallucination_rate = len(invalid_claimed) / max(len(claimed), 1)
    citation_recall = len(relevant_cited) / max(len(valid), 1)

    # Check for hallucination traps in response
    response_text = model_output.get("response", "").lower()
    traps_hit = [t for t in trap_phrases if t.lower() in response_text]

    # ── Metric 3: Conflict Detection ──────────────────────────────
    expected_conflict = scenario["conflict_expected"]
    detected_conflict = model_output.get("conflict_detected", False)
    conflict_correct = (expected_conflict == detected_conflict)

    # ── Metric 4: Pluralism Score ─────────────────────────────────
    verdicts = list(predicted.values())
    distinct_verdicts = len(set(verdicts)) if verdicts else 0
    # Higher is better if conflict expected; lower is better if no conflict
    pluralism_score = distinct_verdicts / max(len(SCHOOLS), 1)

    # ── Metric 5: Concept Coverage ───────────────────────────────
    mentioned = set(model_output.get("concepts_mentioned", []))
    required = set(scenario.get("valid_concepts", []))
    concept_coverage = len(mentioned & required) / max(len(required), 1) if required else None

    return {
        "doctrinal_accuracy": round(accuracy, 3) if accuracy is not None else None,
        "school_matches": school_matches,
        "citations_claimed_count": len(claimed),
        "citations_valid_count": len(valid_claimed),
        "citations_relevant_count": len(relevant_cited),
        "hallucination_rate": round(hallucination_rate, 3),
        "citation_recall": round(citation_recall, 3),
        "traps_hit": traps_hit,
        "conflict_expected": expected_conflict,
        "conflict_detected": detected_conflict,
        "conflict_correct": conflict_correct,
        "distinct_verdicts": distinct_verdicts,
        "pluralism_score": round(pluralism_score, 3),
        "concept_coverage": round(concept_coverage, 3) if concept_coverage is not None else None,
    }


def aggregate_metrics(all_results: list[dict]) -> dict:
    """Compute aggregate metrics across all scenarios for all models."""
    model_stats = defaultdict(lambda: {
        "doctrinal_accuracy": [], "hallucination_rate": [],
        "citation_recall": [], "conflict_correct": [],
        "pluralism_score": [], "concept_coverage": [],
        "elapsed_sec": [], "traps_hit_count": 0,
        "total": 0, "errors": 0,
    })

    for r in all_results:
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

    def safe_mean(lst): return round(sum(lst) / len(lst), 3) if lst else None

    summary = {}
    for model, stats in model_stats.items():
        summary[model] = {
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

    return summary


# ─────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────

MODEL_RUNNERS = {
    "vanilla_llm": run_vanilla_llm,
    "rag":         run_rag,
    "symbolic":    run_symbolic,
    "hybrid":      run_hybrid,
}


def run_evaluation(
    models: list[str] = None,
    scenario_ids: list[str] = None,
    max_scenarios: int = None,
    delay_sec: float = 2.0,
) -> str:
    """
    Run full evaluation.

    Args:
        models: which model variants to run (default: all 4)
        scenario_ids: which scenario IDs to run (default: all 34)
        max_scenarios: limit number of scenarios (for quick testing)
        delay_sec: pause between LLM calls to avoid rate limits

    Returns path to full_results.json
    """
    if models is None:
        models = list(MODEL_RUNNERS.keys())

    scenarios = SCENARIOS
    if scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in scenario_ids]
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_timestamp
    run_dir.mkdir(exist_ok=True)

    all_results = []
    log_lines = []

    total = len(models) * len(scenarios)
    done = 0

    print(f"\n{'='*60}")
    print(f"HinduMind Evaluation — {run_timestamp}")
    print(f"Models: {models}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Total runs: {total}")
    print(f"{'='*60}\n")

    for model_name in models:
        runner = MODEL_RUNNERS[model_name]
        print(f"\n[{model_name.upper()}] Starting...")

        for scenario in scenarios:
            sid = scenario["id"]
            q = scenario["question"]
            done += 1
            print(f"  [{done}/{total}] {sid} — {q[:60]}...")

            try:
                output = runner(q)
            except Exception as e:
                output = {
                    "model": model_name,
                    "response": f"[RUNNER ERROR] {e}",
                    "citations_claimed": [],
                    "concepts_mentioned": [],
                    "school_verdicts": {},
                    "conflict_detected": False,
                    "conflict_reason": "",
                    "retrieval_used": False,
                    "kg_used": False,
                    "elapsed_sec": 0,
                    "error": str(e),
                }

            metrics = compute_metrics(scenario, output)

            record = {
                "run_id": f"{run_timestamp}_{model_name}_{sid}",
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "scenario_id": sid,
                "group": scenario["group"],
                "question": q,
                "output": output,
                "metrics": metrics,
                "ground_truth": {
                    "expected_verdicts": scenario["expected_verdict_per_school"],
                    "supporting_verses": scenario["supporting_verses"],
                    "conflict_expected": scenario["conflict_expected"],
                    "conflict_schools": scenario.get("conflict_schools", []),
                    "valid_concepts": scenario.get("valid_concepts", []),
                }
            }
            all_results.append(record)

            # Readable log
            log_lines.append(f"\n{'─'*60}")
            log_lines.append(f"Model: {model_name} | Scenario: {sid} | Group: {scenario['group']}")
            log_lines.append(f"Q: {q}")
            log_lines.append(f"A: {output['response'][:400]}{'...' if len(output['response']) > 400 else ''}")
            log_lines.append(f"Citations: {output.get('citations_claimed', [])}")
            log_lines.append(f"Verdicts: {output.get('school_verdicts', {})}")
            log_lines.append(f"Conflict detected: {output.get('conflict_detected', False)}")
            log_lines.append(f"Metrics: accuracy={metrics['doctrinal_accuracy']} | hallucination={metrics['hallucination_rate']} | conflict_correct={metrics['conflict_correct']}")

            print(f"      ✓ acc={metrics['doctrinal_accuracy']}, halluc={metrics['hallucination_rate']}, conflict={metrics['conflict_correct']}, time={output.get('elapsed_sec', 0)}s")

            # Rate limit delay
            if model_name != "symbolic" and delay_sec > 0:
                time.sleep(delay_sec)

        print(f"  [{model_name}] Done.")

    # Aggregate metrics
    summary = aggregate_metrics(all_results)

    # Save everything
    results_path = run_dir / "full_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = run_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log_path = run_dir / "run_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # Also write latest symlink
    (RESULTS_DIR / "latest_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS_DIR / "latest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<18} {'Accuracy':>10} {'Halluc%':>9} {'CitRecall':>10} {'ConflictAcc':>12} {'Pluralism':>10} {'Time(s)':>8}")
    print(f"{'─'*18} {'─'*10} {'─'*9} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
    for model, s in summary.items():
        acc = f"{s['doctrinal_accuracy_mean']:.3f}" if s['doctrinal_accuracy_mean'] else "  N/A"
        hal = f"{s['hallucination_rate_mean']:.3f}" if s['hallucination_rate_mean'] is not None else "  N/A"
        cit = f"{s['citation_recall_mean']:.3f}" if s['citation_recall_mean'] is not None else "  N/A"
        con = f"{s['conflict_detection_accuracy']:.3f}" if s['conflict_detection_accuracy'] is not None else "  N/A"
        plu = f"{s['pluralism_score_mean']:.3f}" if s['pluralism_score_mean'] is not None else "  N/A"
        tim = f"{s['avg_response_time_sec']:.1f}" if s['avg_response_time_sec'] is not None else "  N/A"
        print(f"{model:<18} {acc:>10} {hal:>9} {cit:>10} {con:>12} {plu:>10} {tim:>8}")

    print(f"\nResults saved to: {run_dir}")
    print(f"  {results_path.name}")
    print(f"  {summary_path.name}")
    print(f"  {log_path.name}")

    return str(results_path)


# ─────────────────────────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────────────────────────

def run_ablation_study(quick: bool = True) -> dict:
    """
    Run ablation to measure contribution of each component.

    quick=True runs on first 5 scenarios for speed.
    """
    n = 5 if quick else len(SCENARIOS)
    print("\nRunning ablation study...")

    ablation_models = ["vanilla_llm", "rag", "symbolic", "hybrid"]
    results_path = run_evaluation(
        models=ablation_models,
        max_scenarios=n,
        delay_sec=1.5,
    )

    summary = json.load(open(Path(results_path).parent / "metrics_summary.json", encoding="utf-8"))

    print("\nAblation Table:")
    print(f"{'Variant':<18} {'KG':>4} {'Agents':>7} {'Retrieval':>10} {'LLM':>5} {'Accuracy':>10} {'Halluc':>8} {'ConflictAcc':>12}")
    component_map = {
        "vanilla_llm": ("❌", "❌", "❌", "✅"),
        "rag":         ("❌", "❌", "✅", "✅"),
        "symbolic":    ("✅", "❌", "❌", "❌"),
        "hybrid":      ("✅", "✅", "✅", "✅"),
    }
    for model in ablation_models:
        s = summary.get(model, {})
        kg, agt, ret, llm = component_map.get(model, ("?","?","?","?"))
        acc = f"{s.get('doctrinal_accuracy_mean', 0):.3f}" if s.get('doctrinal_accuracy_mean') else " N/A"
        hal = f"{s.get('hallucination_rate_mean', 0):.3f}" if s.get('hallucination_rate_mean') is not None else " N/A"
        con = f"{s.get('conflict_detection_accuracy', 0):.3f}" if s.get('conflict_detection_accuracy') is not None else " N/A"
        print(f"{model:<18} {kg:>4} {agt:>7} {ret:>10} {llm:>5} {acc:>10} {hal:>8} {con:>12}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HinduMind Evaluation Harness")
    parser.add_argument("--models", nargs="+",
                        choices=["vanilla_llm", "rag", "symbolic", "hybrid", "all"],
                        default=["all"])
    parser.add_argument("--scenarios", nargs="+", help="Specific scenario IDs (e.g. A01 B03)")
    parser.add_argument("--max", type=int, default=None, help="Max scenarios (for quick test)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between LLM calls (sec)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 scenarios, symbolic+hybrid only")
    args = parser.parse_args()

    if args.quick:
        run_evaluation(
            models=["symbolic", "hybrid"],
            max_scenarios=3,
            delay_sec=1.0,
        )
    elif args.ablation:
        run_ablation_study(quick=(args.max is not None and args.max <= 5))
    else:
        models = list(MODEL_RUNNERS.keys()) if "all" in args.models else args.models
        run_evaluation(
            models=models,
            scenario_ids=args.scenarios,
            max_scenarios=args.max,
            delay_sec=args.delay,
        )
