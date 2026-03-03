"""
school_classifier.py — School Intent Detector for HinduMind

Detects which philosophical school (darśana) a query aligns with,
and extracts key concepts mentioned.

Two modes:
  1. Keyword-based (always available, no dependencies)
  2. Transformer-based (requires fine-tuned model; falls back to keyword)
"""

import re
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Keyword-based classifier (zero-dependency fallback)
# ─────────────────────────────────────────────────────────────────

SCHOOL_KEYWORDS: dict[str, list[str]] = {
    "advaita": [
        "advaita", "non-dual", "nondual", "non dual", "shankara", "sankara", "shankaracharya",
        "maya", "brahman", "nirguna", "atman", "vivekachudamani", "avidya", "vivarta",
        "tat tvam asi", "neti neti", "jnana", "jñāna", "consciousness", "turiya",
        "mandukya", "gaudapada", "ajati", "saguna", "nirakara", "illusion", "moksha advaita"
    ],
    "dvaita": [
        "dvaita", "madhva", "madhvacharya", "madhvachary", "tatvavada", "panchabheda",
        "vishnu", "vaishnava", "vaikuntha", "dvaitin", "anuvyakhyana", "distinction",
        "eternally distinct", "jiva distinct", "grace of vishnu", "jayatirtha", "bheda"
    ],
    "vishishtadvaita": [
        "vishishtadvaita", "ramanuja", "ramanujacharya", "sri vaishnava", "sri bhashya",
        "body of god", "jiva body", "parinama", "qualified non-dualism", "qualified nondualism",
        "vedanta desika", "pillai lokacharya", "chit achit", "prapatti", "surrender"
    ],
    "nyaya": [
        "nyaya", "vaisheshika", "vaiseshika", "logic", "inference", "anumana", "pratyaksha",
        "pramana", "syllogism", "kanada", "gautama", "nyayasutra", "nyaya sutra",
        "atomic", "paramanu", "categories", "padaartha", "natural theology", "nyaya proof god",
        "gangesa", "navya nyaya", "udayana"
    ],
    "mimamsa": [
        "mimamsa", "jaimini", "vedic ritual", "dharma mimamsa", "karma kanda", "sabda pramana",
        "vedic injunction", "eternal veda", "kumarila", "prabhakara", "svabhavika dharma",
        "manusmriti", "yajnavalkya", "arthashastra", "ritual", "sacrifice", "yajna",
        "dharmashastra", "dharmasastra"
    ],
    "bhakti": [
        "bhakti", "devotion", "devotional", "caitanya", "chaitanya", "krishna", "rama bhakti",
        "narada", "narada bhakti", "bhagavata", "bhagavatam", "surrender", "divine love",
        "nama japa", "kirtan", "tukaram", "mirabai", "ramananda", "divine grace", "prema"
    ]
}

CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "atman": ["atman", "ātman", "self", "soul", "individual self"],
    "brahman": ["brahman", "brahma", "absolute", "ultimate reality"],
    "karma": ["karma", "action", "kamma", "deed"],
    "moksha": ["moksha", "liberation", "mukti", "nirvana", "freedom"],
    "dharma": ["dharma", "duty", "righteousness", "law", "cosmic order"],
    "maya": ["maya", "illusion", "cosmic illusion", "māyā"],
    "ahimsa": ["ahimsa", "non-violence", "nonviolence", "harmlessness"],
    "jnana": ["jnana", "jñāna", "knowledge", "wisdom", "gnana"],
    "bhakti": ["bhakti", "devotion", "love of god"],
    "karma_yoga": ["karma yoga", "selfless action", "nishkama karma"],
    "pramana": ["pramana", "valid knowledge", "epistemology", "means of knowledge"],
    "samsara": ["samsara", "rebirth", "cycle", "reincarnation"],
    "viveka": ["viveka", "discrimination", "discernment"],
    "yoga": ["yoga", "meditation", "dhyana", "samadhi"]
}


def classify_keywords(text: str) -> dict:
    """
    Keyword-based classification.
    Returns: {school, confidence, all_scores, concepts}
    """
    text_lower = text.lower()

    scores: dict[str, float] = {s: 0.0 for s in SCHOOL_KEYWORDS}
    for school, keywords in SCHOOL_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                scores[school] += 1.0
            elif kw in text_lower:
                scores[school] += 0.5

    # Normalise
    total = sum(scores.values()) or 1.0
    norm_scores = {s: round(v / total, 3) for s, v in scores.items()}

    best_school = max(scores, key=scores.get)
    best_conf   = norm_scores[best_school]

    # Detect concepts
    found_concepts = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found_concepts.append(concept)
                break

    return {
        "school": best_school if best_conf > 0 else "advaita",  # fallback
        "confidence": best_conf if best_conf > 0 else 0.1,
        "all_scores": norm_scores,
        "concepts": list(set(found_concepts)),
        "method": "keyword"
    }


# ─────────────────────────────────────────────────────────────────
# Transformer-based classifier (optional, requires fine-tuning or
# zero-shot with a suitable HF model)
# ─────────────────────────────────────────────────────────────────

SCHOOL_LABELS = ["advaita", "dvaita", "vishishtadvaita", "nyaya", "mimamsa", "bhakti"]


class TransformerClassifier:
    """
    Zero-shot classifier using HuggingFace zero-shot-classification pipeline.
    Hypothesis template: "This text is about {label} philosophy."
    Requires: transformers, torch
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1  # CPU; change to 0 for GPU
            )

    def classify(self, text: str) -> dict:
        self._load()
        result = self._pipeline(
            text,
            candidate_labels=SCHOOL_LABELS,
            hypothesis_template="This text discusses {} Vedanta or Hindu philosophy."
        )
        scores_dict = dict(zip(result["labels"], result["scores"]))
        best_school = result["labels"][0]
        best_conf   = result["scores"][0]

        # Combine with keyword concept detection
        concept_result = classify_keywords(text)

        return {
            "school": best_school,
            "confidence": round(best_conf, 3),
            "all_scores": {k: round(v, 3) for k, v in scores_dict.items()},
            "concepts": concept_result["concepts"],
            "method": "transformer"
        }


# ─────────────────────────────────────────────────────────────────
# Main SchoolClassifier (facade with auto-fallback)
# ─────────────────────────────────────────────────────────────────

class SchoolClassifier:
    """
    Facade: tries transformer, falls back to keyword classifier gracefully.
    Usage:
        clf = SchoolClassifier()
        result = clf.classify("What is the nature of atman in Advaita?")
        print(result["school"], result["confidence"])
    """

    def __init__(self, use_transformer: bool = True):
        self._transformer: Optional[TransformerClassifier] = None
        if use_transformer:
            try:
                self._transformer = TransformerClassifier()
            except Exception:
                pass

    def classify(self, text: str) -> dict:
        if self._transformer:
            try:
                return self._transformer.classify(text)
            except Exception as e:
                print(f"[Classifier] Transformer failed ({e}), falling back to keyword.")
        return classify_keywords(text)


# ─────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "What is the relationship between atman and brahman?",
    "How does Madhva understand the distinction between jiva and Vishnu?",
    "What are the valid means of knowledge (pramanas) according to Nyaya?",
    "Is Vedic ritual sufficient for moksha according to Jaimini?",
    "How does bhakti lead to liberation in the Bhagavata Purana?",
    "Explain the concept of maya in Advaita Vedanta.",
    "Does Ramanuja accept the world as real transformation of Brahman?",
]

if __name__ == "__main__":
    clf = SchoolClassifier(use_transformer=False)  # keyword only for fast test

    print("=" * 60)
    print("  HinduMind — School Classifier Test")
    print("=" * 60)
    for q in TEST_QUERIES:
        result = clf.classify(q)
        print(f"\nQ: {q}")
        print(f"   School: {result['school']} (conf={result['confidence']:.2f})")
        print(f"   Concepts: {result['concepts']}")
        print(f"   Method: {result['method']}")
