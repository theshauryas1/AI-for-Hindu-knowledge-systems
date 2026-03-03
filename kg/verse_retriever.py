"""
verse_retriever.py — Top-K Doctrinal Node & Verse Retrieval (Layer 2)

Retrieves the most relevant concepts, texts, and verse citations
from the HPO Knowledge Graph for a given query or concept list.

Used by the Contextual Synthesis Layer when no single verse
directly answers the question.
"""

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from kg.hpo_graph import HPOGraph

_PROCESSED_DIR = Path(__file__).parent / "processed"
_VERSE_DB_PATH  = _PROCESSED_DIR / "verse_db.json"
_CONCEPT_IDX_PATH = _PROCESSED_DIR / "concept_verse_index.json"


# ─────────────────────────────────────────────────────────────────
# RetrievedNode — one ranked doctrinal result
# ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedNode:
    node_id: str
    label: str
    node_type: str
    relevance_score: float          # 0.0–1.0
    source_verses: list[str]        # e.g. ["BG 3.20", "MS 8.15"]
    school_stances: dict            # {school_id: "endorsed"/"rejected"/"neutral"}
    definition: str = ""
    match_reason: str = ""          # Why this node was retrieved

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    query_concepts: list[str]
    retrieved_nodes: list[RetrievedNode]
    has_direct_match: bool          # Layer 1 flag: exact verse found
    direct_match_source: str        # Citation if Layer 1 applies
    top_k: int

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_concepts": self.query_concepts,
            "retrieved_nodes": [n.to_dict() for n in self.retrieved_nodes],
            "has_direct_match": self.has_direct_match,
            "direct_match_source": self.direct_match_source,
            "top_k": self.top_k
        }

    def build_context_block(self, pdf_passages: list[dict] = None) -> str:
        """
        Build a structured plain-text context block for LLM prompt injection.
        Includes both KG concept nodes AND real PDF passages.
        This is what the constrained LLM receives — no fabrication possible.
        """
        lines = ["RETRIEVED DOCTRINAL CONTEXT (HPO Knowledge Graph + PDF Corpus)"]
        lines.append("=" * 60)

        if self.has_direct_match:
            lines.append(f"⚡ DIRECT TEXTUAL MATCH: {self.direct_match_source}")
            lines.append("   → Use Layer 1 (Hard Grounding) only")
            lines.append("")

        # KG Concept Nodes
        lines.append("── KG CONCEPTS ──────────────────────────────────────")
        for i, node in enumerate(self.retrieved_nodes, 1):
            lines.append(f"[{i}] {node.label} ({node.node_type})")
            if node.definition:
                lines.append(f"    Definition: {node.definition[:200]}")
            if node.source_verses:
                lines.append(f"    Primary Sources: {' | '.join(node.source_verses)}")
            if node.school_stances:
                stances = [f"{s}: {v}" for s, v in node.school_stances.items() if v != "neutral"]
                if stances:
                    lines.append(f"    School Stances: {' | '.join(stances)}")
            lines.append(f"    Relevance: {node.relevance_score:.2f} — {node.match_reason}")
            lines.append("")

        # PDF Passages (real text from Gita + Puranas)
        if pdf_passages:
            lines.append("── PDF PASSAGES (Bhagavad Gītā / 18 Purāṇas) ─────────")
            for j, p in enumerate(pdf_passages[:4], 1):
                src = p.get('source', p.get('verse_id', ''))
                text = p.get('translation_en', '')[:500]
                commentary = p.get('commentary', '')
                lines.append(f"[P{j}] {src}")
                if commentary:
                    lines.append(f"    Commentary: {commentary}")
                lines.append(f"    Text: {text}")
                lines.append("")

        lines.append("INSTRUCTION: Your response MUST draw ONLY from the above nodes and passages.")
        lines.append("Do NOT cite verses or concepts not listed above.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# VerseRetriever
# ─────────────────────────────────────────────────────────────────

# Keyword → KG concept mapping (for retrieval scoring)
CONCEPT_KEYWORD_MAP = {
    "dharma":      ["duty", "dharma", "righteousness", "law", "obligation"],
    "ahimsa":      ["violence", "harm", "hurt", "non-violence", "killing", "surveillance", "monitoring"],
    "karma":       ["action", "deed", "consequence", "effect", "karma"],
    "moksha":      ["liberation", "freedom", "moksha", "salvation"],
    "satya":       ["truth", "lie", "deceit", "honesty", "privacy", "data", "transparency"],
    "brahman":     ["ultimate reality", "consciousness", "god", "absolute"],
    "atman":       ["self", "soul", "identity", "individual"],
    "karma_yoga":  ["work", "duty without attachment", "selfless action", "nishkama"],
    "raja_yoga":   ["meditation", "mental discipline", "mind control"],
    "varnashrama": ["role", "social", "caste", "duty by station", "governance", "king", "ruler"],
    "jnana":       ["knowledge", "wisdom", "understanding", "AI", "information", "data"],
    "bhakti":      ["devotion", "love", "compassion", "care"],
    "pratyaksha":  ["perception", "observation", "surveillance", "evidence"],
    "anumana":     ["inference", "reasoning", "logic", "deduction"],
    "ahankara":    ["ego", "power", "control", "authority"],
    "nishkama_karma": ["without attachment", "selfless", "public service"],
    "tapas":       ["discipline", "constraint", "regulation", "rule"],
    "sannyasa":    ["renunciation", "detachment", "privacy", "withdrawal"],
    "gunas":       ["qualities", "nature", "character", "disposition"],
    "ishvara":     ["god", "creator", "divine", "supreme being"]
}

# Modern scenario → ancient concept bridge
MODERN_CONCEPT_BRIDGE = {
    "ai": ["jnana", "anumana", "ahankara", "pratyaksha"],
    "surveillance": ["ahimsa", "satya", "varnashrama", "ahankara"],
    "privacy": ["satya", "ahimsa", "sannyasa"],
    "euthanasia": ["ahimsa", "dharma", "karma", "karma_yoga"],
    "abortion": ["ahimsa", "dharma", "karma"],
    "war": ["dharma", "karma_yoga", "ahimsa", "varnashrama"],
    "environment": ["ahimsa", "dharma", "karma"],
    "data": ["satya", "jnana", "ahimsa"],
    "genetics": ["karma", "dharma", "atman"],
    "cloning": ["atman", "karma", "dharma"],
    "cryptocurrency": ["satya", "dharma", "karma"],
    "democracy": ["varnashrama", "dharma", "karma_yoga"],
    "capitalism": ["asteya", "aparigraha", "dharma", "karma"],
    "crime": ["dharma", "varnashrama", "ahimsa", "karma"],
    "punishment": ["karma", "dharma", "varnashrama"],
    "addiction": ["karma", "vasana", "chitta", "raja_yoga"],
    "depression": ["chitta", "raja_yoga", "bhakti"],
    "loneliness": ["bhakti", "atman", "brahman"],
}


class VerseRetriever:
    """
    Retrieves top-K relevant doctrinal nodes from the HPO KG for a query.

    Scoring formula:
      score = (query_keyword_match * 0.4)
            + (concept_centrality * 0.2)
            + (has_source_verse * 0.3)
            + (school_consensus * 0.1)
    """

    def __init__(self, kg: HPOGraph):
        self.kg = kg
        self._verse_db: list[dict] = self._load_verse_db()
        self._concept_idx: dict = self._load_concept_index()

    @staticmethod
    def _load_verse_db() -> list[dict]:
        """Load verse_db.json from processed/ — includes real PDF passages."""
        if _VERSE_DB_PATH.exists():
            try:
                with open(_VERSE_DB_PATH, encoding="utf-8") as f:
                    db = json.load(f)
                print(f"[VerseRetriever] Loaded verse_db: {len(db)} passages")
                return db
            except Exception as e:
                print(f"[VerseRetriever] verse_db load error: {e}")
        return []

    @staticmethod
    def _load_concept_index() -> dict:
        """Load concept_verse_index.json for fast concept → verse lookup."""
        if _CONCEPT_IDX_PATH.exists():
            try:
                with open(_CONCEPT_IDX_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_pdf_passages_for_concepts(self, concepts: list[str],
                                       query_lower: str = "",
                                       top_n: int = 4) -> list[dict]:
        """
        Retrieve PDF passages from verse_db that match the given concepts.
        Also scores by query keyword overlap for better relevance.
        """
        if not self._verse_db:
            return []

        # Candidate passages for these concepts
        relevant_ids: set[str] = set()
        for concept in concepts:
            for vid in self._concept_idx.get(concept, []):
                relevant_ids.add(vid)

        # Score passages
        scored = []
        for passage in self._verse_db:
            vid = passage.get("verse_id", "")
            text_lower = passage.get("translation_en", "").lower()
            p_concepts = passage.get("concepts", [])

            score = 0.0
            # Concept overlap
            overlap = len(set(concepts) & set(p_concepts))
            score += overlap * 0.3

            # Query keyword overlap
            if query_lower:
                q_words = set(query_lower.split())
                t_words = set(text_lower.split())
                word_overlap = len(q_words & t_words) / max(len(q_words), 1)
                score += word_overlap * 0.4

            # Prefer PDF passages over embedded stubs
            loader = passage.get("loader", "")
            if loader in ("pdf_gita", "pdf_puranas"):
                score += 0.2
            
            # Prefer passages with sanskrit
            if passage.get("sanskrit"):
                score += 0.1

            if score > 0.1:
                scored.append((score, passage))

        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_n]]

    def _score_node(self, node_id: str, node_data: dict, query_lower: str,
                    query_concepts: list[str]) -> tuple[float, str]:
        """Score a concept node for relevance to the query. Returns (score, reason)."""
        score = 0.0
        reasons = []

        label_lower = node_data.get("label", "").lower()
        defn_lower  = node_data.get("hasDefinition", "").lower()
        sanskrit    = node_data.get("hasSanskritName", "").lower()

        # Direct concept mention in query
        if node_id in query_concepts or label_lower in query_lower or sanskrit in query_lower:
            score += 0.5
            reasons.append("direct concept match")

        # Keyword mapping
        kw_list = CONCEPT_KEYWORD_MAP.get(node_id, [])
        kw_hits = sum(1 for kw in kw_list if kw in query_lower)
        if kw_hits:
            score += min(kw_hits * 0.15, 0.35)
            reasons.append(f"keyword match ({kw_hits} hits)")

        # Modern concept bridge
        for modern_term, ancient_concepts in MODERN_CONCEPT_BRIDGE.items():
            if modern_term in query_lower and node_id in ancient_concepts:
                score += 0.3
                reasons.append(f"modern bridge: '{modern_term}'→{node_id}")
                break

        # Has source verse
        texts = self.kg.get_texts_for_concept(node_id)
        if texts:
            score += 0.2
            reasons.append(f"{len(texts)} source verse(s)")

        # Definition matches query terms
        query_words = set(query_lower.split())
        defn_words  = set(defn_lower.split())
        overlap = len(query_words & defn_words)
        if overlap > 2:
            score += min(overlap * 0.02, 0.15)
            reasons.append(f"definition overlap ({overlap} words)")

        return min(score, 1.0), " | ".join(reasons) if reasons else "general relevance"

    def _check_direct_match(self, query_lower: str, concepts: list[str]) -> tuple[bool, str]:
        """
        Layer 1 check: is there an exact verse that answers this?
        Returns (has_match, citation_string)
        """
        DIRECT_VERSE_MAP = {
            # query keyword → (concept_id, verse citation, threshold_keywords)
            ("atman", "brahman"): ("atman", "Chāndogya Upaniṣad 6.8.7 — 'tat tvam asi'"),
            ("karma", "battlefield"): ("karma_yoga", "Bhagavad Gītā 2.47 — 'karmaṇy evādhikāras te'"),
            ("karma", "action"): ("karma_yoga", "Bhagavad Gītā 3.19 — 'nityam kuru karma tvam'"),
            ("moksha", "liberation"): ("moksha", "Brahma Sūtras 1.1.1 — 'athāto brahmajijñāsā'"),
            ("ahimsa", "non-violence"): ("ahimsa", "Yoga Sūtras 2.30 — ahiṃsā listed as first yama"),
            ("satya", "truth"): ("satya", "Yoga Sūtras 2.30 — satya listed as second yama"),
            ("dharma", "duty"): ("dharma", "Bhagavad Gītā 4.7-8 — 'yadā yadā hi dharmasya glāniḥ'"),
            ("kshatriya", "battle"): ("varnashrama", "Bhagavad Gītā 2.31 — 'svadharmam api cāvekṣya'"),
            ("lie", "save"): ("ahimsa", "Mahābhārata Śānti Parva 109.5 — untruth to save innocent"),
            ("karma", "persist", "moksha"): ("karma", "Brahma Sūtras 4.1.13 — karma dissolution at mokṣa"),
        }

        for key_tuple, (concept_id, citation) in DIRECT_VERSE_MAP.items():
            if isinstance(key_tuple, tuple):
                # All keywords in tuple must appear in query for match
                if all(kw in query_lower for kw in key_tuple):
                    if concept_id in concepts or True:  # allow even if concept not detected
                        return True, citation

        return False, ""

    def retrieve(self, query: str, concepts: list[str], top_k: int = 6) -> RetrievalResult:
        """
        Main retrieval method.
        Returns top-K relevant RetrievedNode objects + PDF-passage-enriched context.
        """
        query_lower = query.lower()

        # Expand concepts via modern bridge
        expanded = set(concepts)
        for modern_term, ancient_list in MODERN_CONCEPT_BRIDGE.items():
            if modern_term in query_lower:
                expanded.update(ancient_list)

        # Score all concept nodes
        candidate_concepts = self.kg.get_nodes_by_type("MetaphysicalConcept") + \
                             self.kg.get_nodes_by_type("EthicalConcept") + \
                             self.kg.get_nodes_by_type("EpistemicConcept")

        scored = []
        for node_data in candidate_concepts:
            nid = node_data["id"]
            score, reason = self._score_node(nid, node_data, query_lower, list(expanded))
            if score > 0.05:
                # Gather source verses
                texts = self.kg.get_texts_for_concept(nid)
                sources = [t.get("source", "") for t in texts if t.get("source")][:3]

                # School stances
                cross = self.kg.get_concept_cross_school_view(nid)
                stances = {s: d.get("stance", "neutral") for s, d in cross.items()}

                scored.append(RetrievedNode(
                    node_id=nid,
                    label=node_data.get("label", nid),
                    node_type=node_data.get("type", "Concept"),
                    relevance_score=round(score, 3),
                    source_verses=sources,
                    school_stances=stances,
                    definition=node_data.get("hasDefinition", "")[:200],
                    match_reason=reason
                ))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        top_nodes = scored[:top_k]

        # Layer 1 direct match check
        has_direct, direct_citation = self._check_direct_match(query_lower, list(expanded))

        return RetrievalResult(
            query=query,
            query_concepts=list(expanded),
            retrieved_nodes=top_nodes,
            has_direct_match=has_direct,
            direct_match_source=direct_citation,
            top_k=top_k
        )
