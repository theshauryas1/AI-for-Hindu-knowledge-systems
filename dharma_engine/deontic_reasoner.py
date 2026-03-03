"""
deontic_reasoner.py — Deontic Logic Reasoner for HinduMind

Implements formal deontic logic operators:
  O(p) — it is Obligatory to perform action p
  P(p) — it is Permitted to perform action p
  F(p) — it is Forbidden to perform action p

Given a parsed ScenarioContext and the rule base, returns:
  - Applicable rules (matched by varna, ashrama, action, situation)
  - Deontic judgments per operator
  - School-weighted verdicts
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from dharma_engine.context_parser import ScenarioContext


# ─────────────────────────────────────────────────────────────────
# Deontic Operators
# ─────────────────────────────────────────────────────────────────

class DeonticOp:
    OBLIGATORY = "O"    # Must perform
    PERMITTED  = "P"    # May perform
    FORBIDDEN  = "F"    # Must not perform

OPERATOR_LABELS = {
    "O": "Obligatory",
    "P": "Permitted",
    "F": "Forbidden"
}

# Logical consistency: O(p) implies P(p); F(p) implies ¬P(p)
def is_consistent(judgments: list[str]) -> bool:
    has_O = DeonticOp.OBLIGATORY in judgments
    has_F = DeonticOp.FORBIDDEN  in judgments
    return not (has_O and has_F)   # O and F on same action = contradiction


# ─────────────────────────────────────────────────────────────────
# DeonticJudgment dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class DeonticJudgment:
    rule_id: str
    action: str
    operator: str          # O / P / F
    operator_label: str    # Obligatory / Permitted / Forbidden
    source: str
    condition: str
    obligation_text: str
    school_weights: dict   # {school: HIGH/MEDIUM/LOW}
    match_strength: float  # 0.0–1.0 (how precisely the rule matched)
    contextual_override: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DharmaVerdict:
    scenario: str
    context: dict
    applicable_rules: list[dict]
    judgments: list[dict]             # List of DeonticJudgment dicts
    school_verdicts: dict[str, str]   # {school: "Obligatory" / "Permitted" / "Forbidden" / "Nuanced"}
    overall_verdict: str
    is_consistent: bool
    dominant_operator: str            # Most common O/P/F across matching rules

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty_print(self):
        print("=" * 65)
        print(f"  DHARMA ENGINE VERDICT")
        print("=" * 65)
        print(f"  Scenario : {self.scenario[:60]}...")
        print(f"  Agent    : {self.context.get('agent_role','?')} / {self.context.get('ashrama','?')}")
        print(f"  Action   : {self.context.get('action','?')}")
        print(f"  Situation: {self.context.get('situation','?')}")
        print()
        print(f"  Applicable Rules ({len(self.applicable_rules)} matched):")
        for r in self.applicable_rules:
            print(f"    [{r['id']}] {r['obligation'][:60]}...")
            print(f"           Source: {r['source']} | Operator: {r['deontic_operator']}")
        print()
        print(f"  School Verdicts:")
        for school, verdict in self.school_verdicts.items():
            print(f"    {school:<20} → {verdict}")
        print()
        print(f"  Overall Verdict (dominant): {self.overall_verdict}")
        print(f"  Consistent (no O+F conflict): {self.is_consistent}")
        print("=" * 65)


# ─────────────────────────────────────────────────────────────────
# Rule Matching Logic
# ─────────────────────────────────────────────────────────────────

WEIGHT_SCORES = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

def compute_match_strength(rule: dict, ctx: ScenarioContext) -> float:
    """
    Score how well a rule matches the context (0.0–1.0).
    More specific matches score higher.
    """
    score = 0
    max_score = 4

    # Varna match
    rule_varna = rule.get("varna", "all")
    if rule_varna == "all" or rule_varna == ctx.agent_role:
        score += 1

    # Ashrama match
    rule_ashrama = rule.get("ashrama", "all")
    if rule_ashrama == "all" or rule_ashrama == ctx.ashrama:
        score += 1

    # Action match (check if action appears in condition text)
    condition = rule.get("condition", "").lower()
    action = ctx.action.lower().replace("_", " ")
    if action in condition or ctx.action in condition:
        score += 1.5

    # Situation match
    situation = ctx.situation.lower()
    if situation in condition or ctx.situation in condition:
        score += 0.5

    return min(score / max_score, 1.0)


def rule_applies(rule: dict, ctx: ScenarioContext) -> bool:
    """Hard gate: rule must match varna and ashrama at minimum."""
    rule_varna  = rule.get("varna", "all")
    rule_ashrama = rule.get("ashrama", "all")

    varna_ok   = (rule_varna == "all" or rule_varna == ctx.agent_role)
    ashrama_ok = (rule_ashrama == "all" or rule_ashrama == ctx.ashrama)

    return varna_ok and ashrama_ok


# ─────────────────────────────────────────────────────────────────
# DeonticReasoner
# ─────────────────────────────────────────────────────────────────

class DeonticReasoner:
    """
    Main deontic reasoning engine.

    Usage:
        reasoner = DeonticReasoner()
        verdict = reasoner.evaluate_scenario(context, schools=["advaita","dvaita","nyaya"])
    """

    def __init__(self, rule_base_path: str = None):
        if rule_base_path is None:
            rule_base_path = str(
                Path(__file__).parent / "rule_base.json"
            )
        with open(rule_base_path, "r", encoding="utf-8") as f:
            self._rules: list[dict] = json.load(f)

    def _get_applicable_rules(self, ctx: ScenarioContext, min_strength: float = 0.25) -> list[tuple]:
        """Return (rule, match_strength) pairs above threshold, sorted by strength desc."""
        matched = []
        for rule in self._rules:
            if rule_applies(rule, ctx):
                strength = compute_match_strength(rule, ctx)
                if strength >= min_strength:
                    matched.append((rule, strength))
        matched.sort(key=lambda x: x[1], reverse=True)
        return matched

    def _school_verdict(self, judgments: list[DeonticJudgment], school: str) -> str:
        """
        Compute a school-specific verdict weighting rules by that school's authority.
        Returns: "Obligatory" / "Permitted" / "Forbidden" / "Nuanced" / "No ruling"
        """
        if not judgments:
            return "No ruling"

        op_weighted: dict[str, float] = {"O": 0.0, "P": 0.0, "F": 0.0}
        for jdg in judgments:
            weight_str = jdg.school_weights.get(school, "MEDIUM")
            weight_val = WEIGHT_SCORES.get(weight_str, 2)
            op_weighted[jdg.operator] += weight_val * jdg.match_strength

        dominant_op = max(op_weighted, key=op_weighted.get)
        dominant_val = op_weighted[dominant_op]

        # If O and F are close, it's Nuanced
        if (op_weighted["O"] > 0 and op_weighted["F"] > 0 and
                abs(op_weighted["O"] - op_weighted["F"]) < 1.0):
            return "Nuanced (conflict)"

        if dominant_val == 0.0:
            return "Neutral"
        return OPERATOR_LABELS.get(dominant_op, "Unknown")

    def _dominant_operator(self, judgments: list[DeonticJudgment]) -> str:
        """Overall dominant operator across all rules equally weighted."""
        counts = {"O": 0, "P": 0, "F": 0}
        for jdg in judgments:
            counts[jdg.operator] += 1
        if not any(counts.values()):
            return "None"
        dom = max(counts, key=counts.get)
        return OPERATOR_LABELS.get(dom, "Unknown")

    def evaluate_scenario(self,
                          ctx: ScenarioContext,
                          schools: list[str] = None) -> DharmaVerdict:
        """
        Evaluate an ethical scenario and produce a DharmaVerdict.

        Parameters
        ----------
        ctx     : ScenarioContext from ContextParser
        schools : List of school IDs to compute verdicts for
        """
        schools = schools or ["advaita", "dvaita", "nyaya", "mimamsa", "bhakti"]

        matched_rules = self._get_applicable_rules(ctx)

        judgments = []
        for rule, strength in matched_rules:
            jdg = DeonticJudgment(
                rule_id=rule["id"],
                action=ctx.action,
                operator=rule.get("deontic_operator", "P"),
                operator_label=OPERATOR_LABELS.get(rule.get("deontic_operator", "P"), "Permitted"),
                source=rule.get("source", ""),
                condition=rule.get("condition", ""),
                obligation_text=rule.get("obligation", ""),
                school_weights=rule.get("school_weights", {}),
                match_strength=round(strength, 3),
                contextual_override=ctx.contextual_override
            )
            judgments.append(jdg)

        # Per-school verdicts
        school_verdicts = {s: self._school_verdict(judgments, s) for s in schools}

        # Consistency check (no O + F simultaneously)
        all_ops = [j.operator for j in judgments]
        consistent = is_consistent(all_ops)

        # Overall verdict
        dominant = self._dominant_operator(judgments)

        # Overall text verdict
        if not judgments:
            overall = "No applicable Dharmaśāstra rule found."
        elif not consistent:
            overall = (f"Contextual conflict: rules produce both Obligatory and Forbidden "
                       f"judgments. Contextual override '{ctx.contextual_override}' may resolve.")
        else:
            overall = f"{dominant} — {len(judgments)} rule(s) applied."

        return DharmaVerdict(
            scenario=ctx.raw_scenario,
            context=ctx.to_dict(),
            applicable_rules=[r for r, _ in matched_rules],
            judgments=[j.to_dict() for j in judgments],
            school_verdicts=school_verdicts,
            overall_verdict=overall,
            is_consistent=consistent,
            dominant_operator=dominant
        )
