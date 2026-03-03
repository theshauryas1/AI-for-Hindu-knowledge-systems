"""
concept_mapper.py — Modern Question → Ancient Concept Mapper

Decomposes a modern ethical/philosophical question into:
  1. Raw keywords
  2. Ancient Indian concepts (via bridge map + KG matching)
  3. Applicable governance framework (varna / governance / universal)

This is the entry point for Layer 2 (Contextual Synthesis).
It handles questions that no single ancient verse can answer directly.

Examples:
  "Is AI surveillance ethical?"
      → concepts: [ahimsa, satya, varnashrama, ahankara, pratyaksha]
      → framework: raja-dharma
      → governance: [Manusmriti, Arthashastra, BG 3.20]

  "Should euthanasia be permitted?"
      → concepts: [ahimsa, dharma, karma, karma_yoga]
      → framework: individual + social dharma
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Modern Domain → Conceptual Frame
# ─────────────────────────────────────────────────────────────────

DOMAIN_FRAMES = {
    "governance": {
        "keywords": ["government", "state", "king", "ruler", "policy", "law", "surveillance",
                     "military", "police", "taxation", "democracy", "election"],
        "dharma_frame": "Raja-dharma (ruler's duty)",
        "primary_texts": ["Arthaśāstra", "Manusmṛti 7", "Bhagavad Gītā 3.20 (lokasaṅgraha)"],
        "primary_concepts": ["varnashrama", "dharma", "karma_yoga", "satya"]
    },
    "technology": {
        "keywords": ["ai", "artificial intelligence", "algorithm", "data", "surveillance",
                     "privacy", "robot", "automation", "internet", "social media"],
        "dharma_frame": "Knowledge (Jñāna) and power (Śakti) ethics",
        "primary_texts": ["Bhagavad Gītā 4.33 (jñāna-yajña)", "Nyāya Sūtras (pramāṇa)"],
        "primary_concepts": ["jnana", "ahimsa", "satya", "ahankara", "pratyaksha"]
    },
    "life_death": {
        "keywords": ["euthanasia", "suicide", "death", "killing", "abortion", "life", "dying",
                     "terminal", "end of life", "murder"],
        "dharma_frame": "Ahiṃsā and Karma in relation to jīva",
        "primary_texts": ["Bhagavad Gītā 2.19-20 (ātman is unborn)", "Manusmṛti 5.45 (ahiṃsā)"],
        "primary_concepts": ["ahimsa", "karma", "dharma", "atman", "karma_yoga"]
    },
    "environment": {
        "keywords": ["environment", "ecology", "nature", "climate", "pollution", "animals",
                     "forest", "resource", "sustainability", "species"],
        "dharma_frame": "Dharma toward all beings (sarva-bhūta-hita)",
        "primary_texts": ["Manusmṛti 5.45 (ahiṃsā)", "Bhagavad Gītā 3.10-11"],
        "primary_concepts": ["ahimsa", "dharma", "karma", "satya"]
    },
    "economy": {
        "keywords": ["money", "wealth", "debt", "economy", "tax", "trade", "profit", "poverty",
                     "inequality", "capitalism", "crypto", "finance"],
        "dharma_frame": "Artha within Dharma (dharmic wealth-seeking)",
        "primary_texts": ["Manusmṛti 7.80", "Arthaśāstra 2.1"],
        "primary_concepts": ["dharma", "asteya", "aparigraha", "karma_yoga", "karma"]
    },
    "mental_health": {
        "keywords": ["depression", "anxiety", "suicide", "mental", "loneliness", "trauma",
                     "therapy", "well-being", "happiness"],
        "dharma_frame": "Citta-śuddhi (mind purification) and Yoga",
        "primary_texts": ["Yoga Sūtras 1.2 (citta-vṛtti-nirodhaḥ)", "Bhagavad Gītā 2.14"],
        "primary_concepts": ["chitta", "raja_yoga", "vairagya", "bhakti", "atman"]
    },
    "social_justice": {
        "keywords": ["equality", "discrimination", "caste", "gender", "race", "justice",
                     "rights", "oppression", "privilege"],
        "dharma_frame": "Varṇāśrama-dharma and Sāmānya-dharma (universal ethics)",
        "primary_texts": ["Bhagavad Gītā 5.18 (equal vision)", "Manusmṛti 1.87"],
        "primary_concepts": ["dharma", "varnashrama", "ahimsa", "satya", "karma"]
    },
}

# Universal dharma applicable always
UNIVERSAL_DHARMA_CONCEPTS = ["ahimsa", "satya", "dharma", "karma"]

# Arjuna-like conflict patterns (existential dilemmas)
DILEMMA_PATTERNS = [
    (["duty", "harm"], "Svadharma vs Ahiṃsā dilemma (Arjuna paradigm)"),
    (["truth", "harm"], "Satya vs Ahiṃsā dilemma"),
    (["obey", "ethical"], "Ājñā (command) vs Dharma dilemma"),
    (["individual", "society"], "Vyakti (individual) vs Samāja (society) dharma"),
    (["family", "duty"], "Kula dharma vs Rāja dharma tension"),
]


# ─────────────────────────────────────────────────────────────────
# Mapped Question dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class MappedQuestion:
    raw_query: str
    detected_domain: str             # e.g. "technology", "governance"
    dharma_frame: str                # Human-readable frame
    primary_concepts: list[str]      # KG concept IDs
    primary_texts: list[str]         # Text citations for retrieval
    dilemma_type: Optional[str]      # If a known dilemma pattern is detected
    is_modern: bool                  # True if no direct ancient parallel
    modern_terms_found: list[str]    # e.g. ["AI", "surveillance"]

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        lines = [
            f"Query: {self.raw_query}",
            f"Domain: {self.detected_domain}",
            f"Frame: {self.dharma_frame}",
            f"Concepts: {', '.join(self.primary_concepts)}",
            f"Texts: {', '.join(self.primary_texts)}",
        ]
        if self.dilemma_type:
            lines.append(f"Dilemma: {self.dilemma_type}")
        if self.modern_terms_found:
            lines.append(f"Modern terms mapped: {', '.join(self.modern_terms_found)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# ConceptMapper
# ─────────────────────────────────────────────────────────────────

class ConceptMapper:
    """
    Decomposes a natural-language question into its ancient concept frame.
    Works fully offline — no LLM needed at this stage.
    """

    def _detect_domain(self, text: str) -> tuple[str, dict]:
        text_l = text.lower()
        best_domain = "universal"
        best_score = 0
        best_frame = {}

        for domain, frame in DOMAIN_FRAMES.items():
            hits = sum(1 for kw in frame["keywords"] if kw in text_l)
            if hits > best_score:
                best_score = hits
                best_domain = domain
                best_frame = frame

        if best_score == 0:
            return "universal", {
                "dharma_frame": "Sāmānya-dharma (universal ethics)",
                "primary_texts": ["Bhagavad Gītā 18.66", "Manusmṛti 1.10"],
                "primary_concepts": UNIVERSAL_DHARMA_CONCEPTS
            }
        return best_domain, best_frame

    def _detect_dilemma(self, text: str) -> Optional[str]:
        text_l = text.lower()
        for keywords, label in DILEMMA_PATTERNS:
            if all(kw in text_l for kw in keywords):
                return label
        return None

    def _detect_modern_terms(self, text: str) -> list[str]:
        from kg.verse_retriever import MODERN_CONCEPT_BRIDGE
        text_l = text.lower()
        return [term for term in MODERN_CONCEPT_BRIDGE if term in text_l]

    def _is_modern(self, modern_terms: list[str]) -> bool:
        """A question is 'modern' if it contains tech/contemporary terms."""
        MODERN_MARKERS = ["ai", "surveillance", "internet", "data", "crypto",
                          "robot", "automation", "genetics", "cloning", "social media"]
        return any(t in MODERN_MARKERS for t in modern_terms)

    def map(self, query: str) -> MappedQuestion:
        """Decompose a query into its ancient conceptual frame."""
        domain, frame = self._detect_domain(query)
        dilemma = self._detect_dilemma(query)
        modern_terms = self._detect_modern_terms(query)
        is_modern = self._is_modern(modern_terms)

        # Merge concepts: frame concepts + universal dharma
        concepts = list(set(frame["primary_concepts"] + UNIVERSAL_DHARMA_CONCEPTS))

        return MappedQuestion(
            raw_query=query,
            detected_domain=domain,
            dharma_frame=frame["dharma_frame"],
            primary_concepts=concepts,
            primary_texts=frame["primary_texts"],
            dilemma_type=dilemma,
            is_modern=is_modern,
            modern_terms_found=modern_terms
        )


# ─────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mapper = ConceptMapper()
    test_queries = [
        "Is it ethical to use AI surveillance to prevent crime?",
        "Should euthanasia be permitted for terminal patients?",
        "Is it right to lie to protect someone's feelings?",
        "How should a government balance individual privacy and collective security?",
        "Is cryptocurrency ethical from a dharmic perspective?",
        "What is the dharmic view on climate change and environmental destruction?",
    ]
    for q in test_queries:
        result = mapper.map(q)
        print(f"\n{'='*60}")
        print(result.describe())
