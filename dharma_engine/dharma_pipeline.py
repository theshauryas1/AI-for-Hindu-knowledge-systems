"""
dharma_pipeline.py — Three-Layer HinduMind Pipeline

Orchestrates ALL three layers in sequence:
  1. ConceptMapper    → maps question to ancient framework
  2. VerseRetriever   → top-K KG retrieval + direct-verse detection
  3. DeonticReasoner  → O/P/F ethical judgment (for dharmic questions)
  4. School Agents    → per-school constrained reasoning
  5. MetaAgent        → Layer 1/2/3 routing + synthesis + TransparencyReport

Output: HinduMindResponse (full structured JSON with transparency)
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.concept_mapper import ConceptMapper
from kg.verse_retriever import VerseRetriever
from dharma_engine.context_parser import ContextParser
from dharma_engine.deontic_reasoner import DeonticReasoner
from agents.school_agents import build_agents
from agents.meta_agent import MetaAgent, HinduMindResponse
from agents.school_classifier import SchoolClassifier
from kg.populate_kg import populate
from kg.hpo_graph import HPOGraph


class DharmaPipeline:
    """
    Full three-layer HinduMind reasoning pipeline.

    Usage:
        pipeline = DharmaPipeline(schools=["advaita","dvaita","nyaya"])
        result = pipeline.run("Is AI surveillance ethical?")
        result.pretty_print()
    """

    def __init__(self,
                 kg: HPOGraph = None,
                 schools: list[str] = None,
                 llm_backend: str = None):
        self.kg       = kg or self._build_kg()
        self.schools  = schools or ["advaita", "dvaita", "nyaya"]

        # Layer components
        self.mapper   = ConceptMapper()
        self.retriever = VerseRetriever(self.kg)
        self.parser   = ContextParser()
        self.reasoner = DeonticReasoner()
        self.clf      = SchoolClassifier(use_transformer=False)
        self.agents   = build_agents(self.kg, self.schools, llm_backend)
        self.meta     = MetaAgent(self.kg, llm=None)  # auto picks backend

    @staticmethod
    def _build_kg() -> HPOGraph:
        g = HPOGraph()
        populate(g)
        return g

    def run(self, query: str, verbose: bool = False) -> HinduMindResponse:
        """
        Execute the three-layer pipeline on a query.

        Returns HinduMindResponse (Layer 1/2/3 unified output).
        """
        if verbose:
            print(f"\n[Pipeline] Query: {query}")
            print("─" * 60)

        # ── Step 1: Classify + Map ──────────────────────────
        clf_result = self.clf.classify(query)
        mapped = self.mapper.map(query)

        if verbose:
            print(f"[Step 1] Domain: {mapped.detected_domain} | Frame: {mapped.dharma_frame}")
            print(f"         Concepts: {', '.join(mapped.primary_concepts[:5])}")
            print(f"         Modern: {mapped.is_modern} | Terms: {mapped.modern_terms_found}")

        # ── Step 2: Top-K Verse Retrieval ───────────────────
        retrieval = self.retriever.retrieve(
            query=query,
            concepts=mapped.primary_concepts + clf_result.get("concepts", []),
            top_k=6
        )

        if verbose:
            print(f"[Step 2] Retrieved {len(retrieval.retrieved_nodes)} nodes "
                  f"| Direct match: {retrieval.has_direct_match}")
            if retrieval.has_direct_match:
                print(f"         → {retrieval.direct_match_source}")

        # ── Step 3: Deontic Reasoning (for ethical queries) ─
        deontic_verdict = None
        ctx = self.parser.parse(query)
        if ctx.action and ctx.action != "unknown":
            deontic_verdict = self.reasoner.evaluate_scenario(ctx, self.schools)
            if verbose:
                print(f"[Step 3] Deontic: {deontic_verdict.overall_verdict}")

        # ── Step 4: School Agents (KG-grounded) ─────────────
        if verbose:
            print(f"[Step 4] Running {len(self.schools)} school agents...")

        school_responses = {}
        retrieval_concepts = [n.node_id for n in retrieval.retrieved_nodes[:4]]
        for school, agent in self.agents.items():
            resp = agent.reason(
                query=query,
                concepts=retrieval_concepts
            )
            school_responses[school] = resp
            if verbose:
                print(f"         {school}: ✓")

        # ── Step 5: Meta-Agent (Three-Layer Routing) ─────────
        if verbose:
            layer = "hard_grounding" if retrieval.has_direct_match else "contextual_synthesis"
            print(f"[Step 5] Meta-Agent → Layer: {layer}")

        # Determine primary concept
        primary = retrieval_concepts[0] if retrieval_concepts else \
                  (mapped.primary_concepts[0] if mapped.primary_concepts else "dharma")

        response = self.meta.synthesize(
            query=query,
            concept=primary,
            school_responses=school_responses,
            retrieval_result=retrieval,
            mapped_question=mapped,
            deontic_verdict=deontic_verdict,
            kg_sources=mapped.primary_texts
        )

        if verbose:
            from agents.confidence import TransparencyReport
            tr = TransparencyReport(**response.transparency)
            print(f"\n{tr.render()}")

        return response


# ─────────────────────────────────────────────────────────────────
# Demo Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("LLM_BACKEND", "mock")

    pipeline = DharmaPipeline(schools=["advaita", "dvaita", "nyaya"])

    DEMO_QUERIES = [
        "Is AI surveillance ethical from a dharmic perspective?",
        "Is it dharmic for a kshatriya to refuse battle when his kingdom is threatened?",
        "What is the relationship between atman and brahman?",
    ]

    for query in DEMO_QUERIES:
        resp = pipeline.run(query, verbose=True)
        resp.pretty_print()
        print()
