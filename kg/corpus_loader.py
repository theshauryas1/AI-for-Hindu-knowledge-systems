"""
corpus_loader.py — Master Corpus Controller for HinduMind

Orchestrates all text loaders and:
  1. Downloads/loads all text corpora
  2. Merges into unified verse-level JSON DB
  3. Indexes verses by concept → used for live verse retrieval
  4. Injects real verse data into the HPO Knowledge Graph

Usage:
  python kg/corpus_loader.py               # Full download (scrape all)
  python kg/corpus_loader.py --offline     # Embedded fallback only (no network)
  python kg/corpus_loader.py --gita-only   # Just the Gītā
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.sources.gita_loader    import load_gita, GITA_CORE_VERSES
from kg.sources.sacred_texts_loader import load_upanishads, UPANISHAD_CORE_PASSAGES
from kg.sources.gretil_loader  import load_gretil_texts, GRETIL_EMBEDDED_PASSAGES


PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VERSE_DB_PATH    = PROCESSED_DIR / "verse_db.json"
CONCEPT_IDX_PATH = PROCESSED_DIR / "concept_verse_index.json"


# ─────────────────────────────────────────────────────────────────
# Verse DB Builder
# ─────────────────────────────────────────────────────────────────

def build_verse_db(gita: list[dict],
                   upanishads: list[dict],
                   gretil: list[dict]) -> list[dict]:
    """Merge all corpora into a single indexed verse list."""
    all_verses = gita + upanishads + gretil

    # Deduplicate on verse_id
    seen = set()
    unique = []
    for v in all_verses:
        vid = v.get("verse_id", "")
        if vid and vid not in seen:
            seen.add(vid)
            unique.append(v)
        elif not vid:
            unique.append(v)  # passthrough if no verse_id

    print(f"[CorpusLoader] Total verses: {len(all_verses)} → "
          f"{len(unique)} after dedup")
    return unique


def build_concept_index(verse_db: list[dict]) -> dict[str, list[str]]:
    """
    Build reverse index: concept → [verse_id, ...]
    Used by VerseRetriever for live lookup.
    """
    index: dict[str, list] = defaultdict(list)
    for verse in verse_db:
        vid   = verse.get("verse_id", "")
        for concept in verse.get("concepts", []):
            if vid:
                index[concept].append(vid)
    return dict(index)


def save_db(verse_db: list[dict], concept_index: dict):
    """Persist verse DB + concept index to disk."""
    VERSE_DB_PATH.write_text(
        json.dumps(verse_db, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    CONCEPT_IDX_PATH.write_text(
        json.dumps(concept_index, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[CorpusLoader] ✓ verse_db.json ({len(verse_db)} verses)")
    print(f"[CorpusLoader] ✓ concept_verse_index.json ({len(concept_index)} concepts)")


# ─────────────────────────────────────────────────────────────────
# KG Integration
# ─────────────────────────────────────────────────────────────────

def inject_into_kg(verse_db: list[dict], kg=None):
    """
    Add real verse data back into HPOGraph as text nodes
    and enhance existing concept-text edges with verse IDs.
    """
    if kg is None:
        from kg.hpo_graph import HPOGraph
        from kg.populate_kg import populate
        kg = HPOGraph()
        populate(kg)

    added = 0
    for verse in verse_db:
        tid     = verse["text_id"]
        verse_id = verse.get("verse_id", "")
        concepts = verse.get("concepts", [])
        src      = verse.get("source", verse_id)

        # Add source verse as text relation to each concept
        for concept in concepts:
            cn = kg.get_node(concept)
            tn = kg.get_node(tid)
            if cn and tn:
                # Check if edge already exists
                neighbors = kg.get_neighbors(concept, predicate="is_defined_in")
                if not any(n["id"] == tid for n in neighbors):
                    kg.add_relation(concept, "is_defined_in", tid,
                                    source=src, confidence=0.9)
                    added += 1

    print(f"[CorpusLoader] ✓ Injected {added} new concept-text edges into KG")
    return kg


# ─────────────────────────────────────────────────────────────────
# Loading Report
# ─────────────────────────────────────────────────────────────────

def print_report(verse_db: list[dict], concept_index: dict):
    by_text = defaultdict(int)
    for v in verse_db:
        by_text[v.get("text_id", "unknown")] += 1

    print()
    print("=" * 55)
    print(f"  📚 HinduMind Corpus Report")
    print("=" * 55)
    print(f"  Total verses/passages : {len(verse_db)}")
    print(f"  Concepts indexed      : {len(concept_index)}")
    print()
    print(f"  By text:")
    for text_id, count in sorted(by_text.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 5, 30)
        print(f"    {text_id:<30}  {count:>4}  {bar}")

    print()
    print(f"  Top concept coverage:")
    top_concepts = sorted(concept_index.items(), key=lambda x: -len(x[1]))[:10]
    for concept, verses in top_concepts:
        print(f"    {concept:<25}  {len(verses):>3} verses")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run(scrape: bool = True,
        gita_api_key: str = None,
        inject_kg: bool = False) -> tuple[list, dict]:
    """
    Full corpus ingestion run.

    Parameters
    ----------
    scrape       : Whether to scrape the web sources
    gita_api_key : Optional bhagavadgita.io API key
    inject_kg    : Whether to also inject into HPOGraph
    """
    print("=" * 55)
    print("  🕉️  HinduMind Corpus Ingestion")
    print("=" * 55)

    # 1. Bhagavad Gītā
    print("\n── Bhagavad Gītā ──────────────────────────────────────")
    gita = load_gita(
        api_key=gita_api_key,
        scrape=scrape,
        use_embedded_fallback=True
    )

    # 2. Upanishads
    print("\n── Upanishads ─────────────────────────────────────────")
    upanishads = load_upanishads(
        scrape=scrape,
        use_embedded_fallback=True
    )

    # 3. GRETIL texts
    print("\n── GRETIL (Dharmaśāstra / Sūtras) ────────────────────")
    gretil = load_gretil_texts(
        scrape=scrape,
        use_embedded_fallback=True
    )

    # 4. Build + save DB
    print("\n── Building Verse DB ──────────────────────────────────")
    verse_db = build_verse_db(gita, upanishads, gretil)
    concept_index = build_concept_index(verse_db)
    save_db(verse_db, concept_index)

    # 5. Inject into KG (optional)
    if inject_kg:
        print("\n── Injecting into HPO KG ──────────────────────────────")
        inject_into_kg(verse_db)

    # 6. Report
    print_report(verse_db, concept_index)

    return verse_db, concept_index


def main():
    parser = argparse.ArgumentParser(description="HinduMind Corpus Loader")
    parser.add_argument("--offline", action="store_true",
                        help="Use embedded fallback only (no network)")
    parser.add_argument("--gita-only", action="store_true",
                        help="Load Bhagavad Gītā only")
    parser.add_argument("--inject-kg", action="store_true",
                        help="Inject corpus into HPO KG after download")
    parser.add_argument("--gita-api-key", type=str, default=None,
                        help="bhagavadgita.io API key (optional)")
    args = parser.parse_args()

    import os
    api_key = args.gita_api_key or os.getenv("BHAGAVAD_GITA_API_KEY")
    scrape  = not args.offline

    if args.gita_only:
        from kg.sources.gita_loader import load_gita
        gita = load_gita(api_key=api_key, scrape=scrape, use_embedded_fallback=True)
        print(f"\nLoaded {len(gita)} Gītā verses.")
    else:
        run(scrape=scrape, gita_api_key=api_key, inject_kg=args.inject_kg)


if __name__ == "__main__":
    main()
