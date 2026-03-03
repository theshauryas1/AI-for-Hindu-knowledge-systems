"""
confidence.py — Meta-Confidence & Transparency Layer (Layer 3)

Computes:
  1. Confidence score (0.0 – 1.0)
  2. Reasoning method (hard_grounding / contextual_synthesis / uncertain)
  3. Ambiguity flags
  4. Conflict detection
  5. "No direct textual equivalent" flag (for modern questions)

This is what gives HinduMind academic credibility and reviewer trust.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────

class ReasoningMethod(str, Enum):
    HARD_GROUNDING        = "hard_grounding"         # Layer 1: exact verse
    CONTEXTUAL_SYNTHESIS  = "contextual_synthesis"   # Layer 2: top-K + constrained LLM
    UNCERTAIN             = "uncertain"              # No KG support, low confidence


class ConfidenceBand(str, Enum):
    HIGH     = "High"     # ≥ 0.75, direct match or near-exact KG support
    MODERATE = "Moderate" # 0.45 – 0.74, contextual synthesis
    LOW      = "Low"      # 0.20 – 0.44, partial / ambiguous
    VERY_LOW = "Very Low" # < 0.20, modern question, no strong mapping


# ─────────────────────────────────────────────────────────────────
# TransparencyReport
# ─────────────────────────────────────────────────────────────────

@dataclass
class TransparencyReport:
    """
    The full transparency metadata block attached to every HinduMindResponse.
    Designed for ACM JOCCH reviewers — provides complete reasoning provenance.
    """
    # Method
    reasoning_method: str                   # ReasoningMethod value
    confidence_score: float                 # 0.0 – 1.0
    confidence_band: str                    # ConfidenceBand value

    # Textual grounding
    has_direct_textual_match: bool
    direct_match_citation: str              # "" if no match
    has_no_direct_equivalent: bool          # True for modern questions

    # Source traceability
    concepts_used: list[str]                # KG concept IDs actually used
    primary_sources: list[str]              # Verse citations used
    schools_with_positions: list[str]       # Schools that have explicit KG stance

    # Conflict & ambiguity
    has_school_conflict: bool               # ≥2 schools take opposite positions
    conflicting_schools: list[dict]         # [{school_a, school_b, on}]
    is_ambiguous: bool                      # Question has multi-valid answers
    ambiguity_note: str                     # Why it's ambiguous

    # Flags
    is_modern_question: bool
    modern_terms_mapped: list[str]
    constrained_to_kg: bool                 # Was LLM constrained to KG nodes?

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """Render the transparency block as readable text (like UI shown in spec)."""
        lines = []
        lines.append("─" * 52)
        lines.append("  📊 TRANSPARENCY REPORT")
        lines.append("─" * 52)

        # Method
        method_label = {
            "hard_grounding": "Hard Grounding (Layer 1 — Direct Verse Match)",
            "contextual_synthesis": "Contextual Synthesis (Layer 2 — Top-K Retrieval + Constrained LLM)",
            "uncertain": "Uncertain (low KG support)"
        }.get(self.reasoning_method, self.reasoning_method)
        lines.append(f"  Method          : {method_label}")
        lines.append(f"  Confidence      : {self.confidence_band} ({self.confidence_score:.2f})")

        # Direct match
        if self.has_direct_textual_match:
            lines.append(f"  Direct Match    : ✓ {self.direct_match_citation}")
        else:
            lines.append(f"  Direct Match    : ✗ None")
            if self.has_no_direct_equivalent:
                lines.append(f"  ⚠  No direct textual equivalent in classical sources")
                lines.append(f"     (modern question mapped via conceptual bridge)")

        # Concepts
        if self.concepts_used:
            lines.append(f"  Concepts Used   : {', '.join(self.concepts_used)}")

        # Sources
        if self.primary_sources:
            lines.append("  Primary Sources :")
            for src in self.primary_sources:
                lines.append(f"    - {src}")

        # Schools
        if self.schools_with_positions:
            lines.append(f"  Schools Engaged : {', '.join(self.schools_with_positions)}")

        # Conflicts
        if self.has_school_conflict:
            lines.append("  ⚡ School Conflicts:")
            for c in self.conflicting_schools:
                lines.append(f"    {c.get('school_a','')} ↔ {c.get('school_b','')} "
                             f"on '{c.get('on','')}'")

        # Ambiguity
        if self.is_ambiguous:
            lines.append(f"  ⚠  Ambiguity     : {self.ambiguity_note}")

        lines.append("─" * 52)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# ConfidenceScorer
# ─────────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Computes a TransparencyReport from the pipeline's intermediate outputs.

    Input:
      - retrieval_result (from VerseRetriever)
      - deontic_verdict  (from DeonticReasoner)
      - school_responses (from school agents)
      - mapped_question  (from ConceptMapper)

    Output:
      - TransparencyReport
    """

    def _compute_confidence(self,
                            has_direct_match: bool,
                            n_retrieved: int,
                            n_sources: int,
                            has_conflict: bool,
                            is_modern: bool,
                            n_rules_matched: int) -> float:
        score = 0.0

        # Base from retrieval
        if has_direct_match:
            score += 0.50      # Strong foundation

        # Nodes retrieved
        score += min(n_retrieved * 0.06, 0.25)

        # Source verses
        score += min(n_sources * 0.04, 0.15)

        # Deontic rule matches
        score += min(n_rules_matched * 0.04, 0.12)

        # Penalties
        if is_modern:
            score -= 0.10      # Modern questions carry uncertainty
        if has_conflict:
            score -= 0.08      # Intra-school conflicts reduce certainty

        return round(min(max(score, 0.05), 1.0), 3)

    def _confidence_band(self, score: float) -> str:
        if score >= 0.75:   return ConfidenceBand.HIGH
        if score >= 0.45:   return ConfidenceBand.MODERATE
        if score >= 0.20:   return ConfidenceBand.LOW
        return ConfidenceBand.VERY_LOW

    def _reasoning_method(self, has_direct_match: bool, score: float) -> str:
        if has_direct_match:
            return ReasoningMethod.HARD_GROUNDING
        if score >= 0.20:
            return ReasoningMethod.CONTEXTUAL_SYNTHESIS
        return ReasoningMethod.UNCERTAIN

    def _detect_ambiguity(self, school_verdicts: dict) -> tuple[bool, str]:
        """Is the question genuinely ambiguous across schools?"""
        if not school_verdicts:
            return False, ""
        unique_verdicts = set(school_verdicts.values())
        if len(unique_verdicts) >= 3:
            return True, f"Schools give {len(unique_verdicts)} different verdicts — no consensus exists"
        if "Nuanced (conflict)" in unique_verdicts:
            return True, "Question involves irresolvable doctrinal tension in at least one school"
        return False, ""

    def score(self,
              retrieval_result=None,
              deontic_verdict=None,
              school_responses: dict = None,
              mapped_question=None,
              contradictions: list = None) -> TransparencyReport:
        """Produce a full TransparencyReport."""

        # Handle None inputs gracefully
        retrieval_result = retrieval_result or type('R', (), {
            'has_direct_match': False, 'direct_match_source': '',
            'retrieved_nodes': [], 'query_concepts': [], 'query': ''
        })()
        deontic_verdict = deontic_verdict or type('V', (), {
            'applicable_rules': [], 'school_verdicts': {}, 'contradictions': []
        })()
        school_responses = school_responses or {}
        mapped_question = mapped_question or type('M', (), {
            'is_modern': False, 'modern_terms_found': [],
            'primary_concepts': [], 'primary_texts': []
        })()
        contradictions = contradictions or []

        # Gather metrics
        has_direct = getattr(retrieval_result, 'has_direct_match', False)
        direct_src = getattr(retrieval_result, 'direct_match_source', '')
        n_nodes = len(getattr(retrieval_result, 'retrieved_nodes', []))
        retrieved_nodes = getattr(retrieval_result, 'retrieved_nodes', [])

        n_sources = sum(len(n.source_verses) for n in retrieved_nodes if hasattr(n, 'source_verses'))
        n_rules   = len(getattr(deontic_verdict, 'applicable_rules', []))
        is_modern = getattr(mapped_question, 'is_modern', False)

        # Conflicts
        has_conflict = len(contradictions) > 0
        school_verdicts_dict = getattr(deontic_verdict, 'school_verdicts', {})

        # Confidence
        conf_score = self._compute_confidence(has_direct, n_nodes, n_sources,
                                              has_conflict, is_modern, n_rules)
        conf_band  = self._confidence_band(conf_score)
        method     = self._reasoning_method(has_direct, conf_score)

        # Ambiguity
        is_ambi, ambi_note = self._detect_ambiguity(school_verdicts_dict)

        # Concepts + sources
        concepts_used = list(set(
            getattr(mapped_question, 'primary_concepts', []) +
            getattr(retrieval_result, 'query_concepts', [])
        ))[:8]
        primary_sources_raw = getattr(mapped_question, 'primary_texts', [])
        verse_sources = [s for n in retrieved_nodes
                         for s in getattr(n, 'source_verses', []) if s]
        all_sources = list(dict.fromkeys(primary_sources_raw + verse_sources))[:6]

        # Schools with positions
        schools_with_pos = [s for s, r in school_responses.items() if r]

        return TransparencyReport(
            reasoning_method=method,
            confidence_score=conf_score,
            confidence_band=conf_band,
            has_direct_textual_match=has_direct,
            direct_match_citation=direct_src,
            has_no_direct_equivalent=(is_modern and not has_direct),
            concepts_used=concepts_used,
            primary_sources=all_sources,
            schools_with_positions=schools_with_pos,
            has_school_conflict=has_conflict,
            conflicting_schools=contradictions[:3],
            is_ambiguous=is_ambi,
            ambiguity_note=ambi_note,
            is_modern_question=is_modern,
            modern_terms_mapped=getattr(mapped_question, 'modern_terms_found', []),
            constrained_to_kg=True   # Always true by design
        )
