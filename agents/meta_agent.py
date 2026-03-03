"""
meta_agent.py — Meta-Agent (Synthesizer) for HinduMind — Three-Layer Architecture

Layer 1: Hard Grounding  (direct verse → use that only)
Layer 2: Contextual Synthesis  (top-K retrieval + constrained LLM)
Layer 3: Meta-Confidence & Transparency  (score, flags, conflict)

Accepts:
  - Retrieved doctrinal context (VerseRetriever)
  - Deontic verdict (DeonticReasoner)
  - All school-agent responses
  - Confidence metadata (ConfidenceScorer)

Produces: HinduMindResponse — the single unified structured output.
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_adapters import get_adapter, BaseLLMAdapter
from agents.confidence import ConfidenceScorer, TransparencyReport
from kg.hpo_graph import HPOGraph


# ─────────────────────────────────────────────────────────────────
# HinduMindResponse — Upgraded Output Schema (Three-Layer)
# ─────────────────────────────────────────────────────────────────

@dataclass
class HinduMindResponse:
    """
    The canonical structured output of a HinduMind query.

    Three-layer architecture:
      layer_used      = "hard_grounding" | "contextual_synthesis" | "uncertain"
      transparency    = TransparencyReport (full metadata)
      responses       = per-school reasoning text
      verdict         = deontic verdict string
      synthesis       = constrained LLM paragraph (Layer 2 only)
    """
    concept: str
    query: str
    layer_used: str                     # Which layer produced this response

    # Per-school outputs
    responses: dict[str, str]           # {school: response text}

    # Cross-school analysis
    agreements: list[str]
    contradictions: list[dict]

    # Layer 2 synthesis (constrained to KG context)
    synthesis: str

    # Transparency (Layer 3)
    transparency: dict                  # TransparencyReport.to_dict()

    # Deontic verdict (when applicable)
    verdict: Optional[str] = None       # DharmaVerdict.overall_verdict
    school_verdicts: dict = field(default_factory=dict)

    # Sources
    kg_sources: list[str] = field(default_factory=list)
    schools_queried: list[str] = field(default_factory=list)
    method: str = "three-layer-hybrid"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def pretty_print(self):
        tr = TransparencyReport(**self.transparency)

        print("=" * 68)
        print("  🕉️  HinduMind Response")
        print("=" * 68)
        print(f"  Query   : {self.query}")
        print(f"  Concept : {self.concept}")
        print(f"  Layer   : {self.layer_used.upper()}")
        print()

        for school, resp in self.responses.items():
            print(f"  ── {school.upper()} " + "─" * (52 - len(school)))
            print(f"  {resp[:260]}{'...' if len(resp)>260 else ''}")
            print()

        if self.agreements:
            print("  ── AGREEMENTS ──────────────────────────────────────────")
            for a in self.agreements:
                print(f"  • {a}")
            print()

        if self.contradictions:
            print("  ── CONTRADICTIONS ──────────────────────────────────────")
            for c in self.contradictions:
                sca = c.get('school_a', c.get('schools', ['?'])[0] if isinstance(c.get('schools'), list) else '?')
                scb = c.get('school_b', '?')
                on  = c.get('on', c.get('note', ''))
                print(f"  • {sca} ↔ {scb}: {on}")
            print()

        if self.verdict:
            print("  ── DEONTIC VERDICT ─────────────────────────────────────")
            print(f"  {self.verdict}")
            for school, v in self.school_verdicts.items():
                print(f"    {school:<20} → {v}")
            print()

        print("  ── SYNTHESIS (Constrained to KG) ───────────────────────")
        print(f"  {self.synthesis}")
        print()

        print(tr.render())
        print("=" * 68)


# ─────────────────────────────────────────────────────────────────
# Agreement / Contradiction Detection
# ─────────────────────────────────────────────────────────────────

def detect_agreements_heuristic(responses: dict[str, str]) -> list[str]:
    agreements = []
    if all(any(t in r.lower() for t in ["moksha", "liberation", "mukti"]) for r in responses.values()):
        agreements.append("All schools affirm that mokṣa (liberation) is the ultimate goal.")
    if all("karma" in r.lower() for r in responses.values()):
        agreements.append("All schools acknowledge karma as conditioning the jīva's embodied state.")
    if all(any(t in r.lower() for t in ["dharma", "duty"]) for r in responses.values()):
        agreements.append("All schools recognize dharma as a binding ethical framework.")
    if all(any(t in r.lower() for t in ["veda", "upanishad", "gita", "scripture"]) for r in responses.values()):
        agreements.append("All schools accept some form of Vedic scriptural authority.")
    return agreements


def detect_contradictions_from_kg(kg: HPOGraph, schools: list[str]) -> list[dict]:
    kg_contras = kg.get_school_contradictions()
    return [c for c in kg_contras
            if c.get("school_a") in schools and c.get("school_b") in schools]


KNOWN_CONTRADICTIONS = [
    {"school_a": "advaita", "school_b": "dvaita",
     "on": "ātman-Brahman identity", "key": "atman"},
    {"school_a": "advaita", "school_b": "dvaita",
     "on": "reality of the world (māyā vs real)", "key": "maya"},
    {"school_a": "advaita", "school_b": "vishishtadvaita",
     "on": "causation (vivarta vs pariṇāma)", "key": "maya"},
    {"school_a": "nyaya", "school_b": "mimamsa",
     "on": "existence of Īśvara as world-creator", "key": "ishvara"},
]


# ─────────────────────────────────────────────────────────────────
# MetaAgent — Three-Layer Synthesizer
# ─────────────────────────────────────────────────────────────────

class MetaAgent:
    """
    Three-layer synthesizer:
      Layer 1  Hard Grounding    (if direct verse exists → cite only that)
      Layer 2  Contextual Synth  (top-K context + constrained LLM prompt)
      Layer 3  Confidence        (TransparencyReport attached to every response)
    """

    SYSTEM_PROMPT = """\
You are a comparative Indian philosophy scholar. You MUST:
1. Draw ONLY from the retrieved doctrinal context provided.
2. NEVER cite verses, texts, or concepts not listed in the context block.
3. Note where schools agree and where they diverge.
4. Write in academic English, past-tense, third-person.
5. Be concise: 4–6 sentences maximum.
6. If no direct ancient parallel exists, state this clearly.\
"""

    def __init__(self, kg: HPOGraph, llm: BaseLLMAdapter = None):
        self.kg = kg
        self.scorer = ConfidenceScorer()
        if llm is None:
            import os
            backend = os.getenv("LLM_BACKEND", "auto")
            if backend == "mock":
                from agents.llm_adapters import MockAdapter
                llm = MockAdapter(school="advaita")
            else:
                llm = get_adapter()
        self.llm = llm

    # ── Layer 1 — Hard Grounding ────────────────────────────────

    def _layer1_response(self, retrieval_result) -> str:
        """When an exact verse is present, return a tightly cited response."""
        citation = retrieval_result.direct_match_source
        return (
            f"[Layer 1 — Hard Grounding]\n\n"
            f"A direct textual parallel exists: {citation}\n"
            f"This verse directly addresses the query. All reasoning is grounded solely "
            f"in this canonical source. No inferential extension is required."
        )

    # ── Layer 2 — Contextual Synthesis (Constrained LLM) ───────

    def _layer2_prompt(self, query: str, concept: str,
                       retrieval_result, responses_text: dict,
                       contradictions: list,
                       pdf_passages: list = None) -> str:
        """Build the constrained prompt. LLM cannot go outside this context."""
        context_block = retrieval_result.build_context_block(pdf_passages=pdf_passages) \
            if hasattr(retrieval_result, 'build_context_block') else ""

        contra_str = "\n".join(
            f"• {c.get('school_a','?')} ↔ {c.get('school_b','?')}: {c.get('on','')}"
            for c in contradictions[:3]
        ) or "None detected."

        school_block = "\n".join(
            f"[{s.upper()}]: {r[:300]}" for s, r in responses_text.items()
        )

        return f"""QUERY: {query}
PRIMARY CONCEPT: {concept}

{context_block}

SCHOOL RESPONSES (use these as starting points, do not invent new ones):
{school_block}

KNOWN DOCTRINAL TENSIONS:
{contra_str}

TASK: Write a 4–6 sentence comparative synthesis.
You MUST reference only the concepts and verses listed in the context block above.
If PDF passages are included above, you MAY quote from them directly (cite their source label).
If this is a modern question with no direct ancient parallel, state that explicitly first.
Do NOT fabricate citations."""

    # ── Main synthesize() ───────────────────────────────────────

    def synthesize(self,
                   query: str,
                   concept: str,
                   school_responses: dict,
                   retrieval_result=None,
                   mapped_question=None,
                   deontic_verdict=None,
                   kg_sources: list[str] = None) -> HinduMindResponse:
        """
        Full three-layer synthesis.

        Parameters
        ----------
        query            : User's original query
        concept          : Primary KG concept
        school_responses : {school_id: agent.reason() dict}
        retrieval_result : From VerseRetriever (optional)
        mapped_question  : From ConceptMapper (optional)
        deontic_verdict  : From DeonticReasoner (optional)
        kg_sources       : Additional source citations
        """
        schools = list(school_responses.keys())
        responses_text = {s: d.get("response", "") if isinstance(d, dict) else str(d)
                          for s, d in school_responses.items()}

        # ── Agreements & contradictions ────────────────────────
        agreements = detect_agreements_heuristic(responses_text)
        kg_contras = detect_contradictions_from_kg(self.kg, schools)
        known_contras = [c for c in KNOWN_CONTRADICTIONS
                         if c["school_a"] in schools and c["school_b"] in schools]
        all_contras = list({f"{c['school_a']}_{c['school_b']}": c
                            for c in kg_contras + known_contras}.values())

        # ── Layer routing ─────────────────────────────────────
        has_direct = retrieval_result.has_direct_match \
            if retrieval_result and hasattr(retrieval_result, 'has_direct_match') else False
        is_modern  = mapped_question.is_modern \
            if mapped_question and hasattr(mapped_question, 'is_modern') else False

        # ── PDF Passages from verse_db ────────────────────────
        retrieval_concepts = []
        if retrieval_result and hasattr(retrieval_result, 'retrieved_nodes'):
            retrieval_concepts = [n.node_id for n in retrieval_result.retrieved_nodes[:4]]
        all_concepts = list(set(
            [concept] + retrieval_concepts +
            (getattr(mapped_question, 'primary_concepts', []) or [])
        ))

        pdf_passages = []
        if retrieval_result and hasattr(retrieval_result, 'get_pdf_passages_for_concepts'):
            pdf_passages = retrieval_result.get_pdf_passages_for_concepts(
                concepts=all_concepts,
                query_lower=query.lower(),
                top_n=4
            )

        if has_direct:
            layer_used = "hard_grounding"
            synthesis_text = self._layer1_response(retrieval_result)
        else:
            layer_used = "contextual_synthesis"
            prompt = self._layer2_prompt(query, concept, retrieval_result,
                                          responses_text, all_contras,
                                          pdf_passages=pdf_passages)
            synthesis_text = self.llm.generate(prompt, self.SYSTEM_PROMPT)

        # ── Confidence / Transparency (Layer 3) ──────────────
        transparency = self.scorer.score(
            retrieval_result=retrieval_result,
            deontic_verdict=deontic_verdict,
            school_responses=school_responses,
            mapped_question=mapped_question,
            contradictions=all_contras
        )

        # ── KG Sources ────────────────────────────────────────
        if kg_sources is None:
            node = self.kg.get_node(concept)
            if node:
                text_nodes = self.kg.get_texts_for_concept(concept)
                kg_sources = [f"{t.get('label','')} — {t.get('source','')}"
                              for t in text_nodes[:5] if t.get("source")]
            else:
                kg_sources = []

        # ── Deontic info ─────────────────────────────────────
        verdict_str   = getattr(deontic_verdict, 'overall_verdict', None)
        school_v_dict = getattr(deontic_verdict, 'school_verdicts', {})

        return HinduMindResponse(
            concept=concept,
            query=query,
            layer_used=layer_used,
            responses=responses_text,
            agreements=agreements,
            contradictions=all_contras,
            synthesis=synthesis_text,
            transparency=transparency.to_dict(),
            verdict=verdict_str,
            school_verdicts=school_v_dict,
            kg_sources=kg_sources,
            schools_queried=schools
        )
