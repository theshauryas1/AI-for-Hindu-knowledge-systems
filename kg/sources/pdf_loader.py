"""
pdf_loader.py — Local PDF Text Extractor for HinduMind

Reads the user-provided PDF files:
  - The Bhagavad Gita.pdf   (Raghavendra Teertha commentary, 447 pages)
  - 18 Puranas.pdf          (Condensed 18 Mahāpurāṇas, 1348 pages)

Extraction strategy:
  - Gīta  : Sliding-window chunking + verse pattern detection (Ch/Verse/Shloka)
  - Puranas: Purana-section detection + narrative paragraph chunking

Output:
  - kg/processed/gita_from_pdf.json
  - kg/processed/puranas_from_pdf.json
  - kg/processed/verse_db.json  (merged with existing corpus)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("[PDFLoader] pymupdf not installed — run: pip install pymupdf")


PROCESSED_DIR = Path(__file__).parent.parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# PDFs are in the project root  (d:\AI for preserving hindu philosophy\)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
GITA_PDF    = _PROJECT_ROOT / "The Bhagavad Gita.pdf"
PURANAS_PDF = _PROJECT_ROOT / "18 Puranas.pdf"



# ─────────────────────────────────────────────────────────────────
# Auto-Concept Tagger
# ─────────────────────────────────────────────────────────────────

CONCEPT_KEYWORD_TAGGER = {
    "atman":           ["atman", "self", "soul", "jivatma", "individual self"],
    "brahman":         ["brahman", "absolute", "brahma", "ultimate reality", "brahm"],
    "dharma":          ["dharma", "duty", "righteousness", "right conduct", "law"],
    "karma":           ["karma", "action", "deeds", "fruit of action", "karmic"],
    "karma_yoga":      ["karma yoga", "nishkama", "selfless action", "without attachment"],
    "jnana":           ["jnana", "knowledge", "wisdom", "vidya", "enlightenment"],
    "bhakti":          ["bhakti", "devotion", "love of god", "devotee", "worship"],
    "moksha":          ["moksha", "liberation", "mukti", "salvation", "freedom"],
    "maya":            ["maya", "illusion", "delusion", "unreal", "appearance"],
    "ahimsa":          ["ahimsa", "non-violence", "harmlessness", "do not harm", "nonviolence"],
    "satya":           ["satya", "truth", "truthfulness", "honesty"],
    "varnashrama":     ["varna", "ashrama", "caste", "duty of", "kshatriya", "brahmin",
                        "householder", "sannyasi", "grihastha", "brahmacharya"],
    "ishvara":         ["ishvara", "god", "lord", "vishnu", "krishna", "shiva", "divine"],
    "prakriti":        ["prakriti", "nature", "material nature", "creation"],
    "purusha":         ["purusha", "consciousness", "spirit", "witness"],
    "gunas":           ["gunas", "sattva", "rajas", "tamas", "qualities of nature"],
    "chitta":          ["chitta", "mind", "mental", "consciousness", "intellect", "buddhi"],
    "raja_yoga":       ["yoga", "meditation", "dhyana", "samadhi", "eight-fold"],
    "sannyasa":        ["renunciation", "sannyasa", "renounce", "monk", "ascetic"],
    "asteya":          ["asteya", "non-stealing", "not stealing"],
    "tapas":           ["tapas", "austerity", "discipline", "penance"],
    "pramana":         ["pramana", "valid knowledge", "proof", "inference", "testimony"],
}

# Purana names for section detection
PURANA_NAMES = {
    "Brahma Purāṇa":       ["brahma purana", "brahma puranam"],
    "Padma Purāṇa":        ["padma purana"],
    "Viṣṇu Purāṇa":        ["vishnu purana", "vishnu puranam"],
    "Vāyu Purāṇa":         ["vayu purana"],
    "Bhāgavata Purāṇa":    ["bhagavata", "bhagavata purana", "srimad bhagavatam"],
    "Nārada Purāṇa":       ["narada purana"],
    "Mārkaṇḍeya Purāṇa":  ["markandeya purana"],
    "Agni Purāṇa":         ["agni purana"],
    "Bhaviṣya Purāṇa":    ["bhavishya purana"],
    "Brahma Vaivarta":     ["brahma vaivarta"],
    "Liṅga Purāṇa":        ["linga purana"],
    "Varāha Purāṇa":       ["varaha purana"],
    "Skanda Purāṇa":       ["skanda purana"],
    "Vāmana Purāṇa":       ["vamana purana"],
    "Kūrma Purāṇa":        ["kurma purana"],
    "Matsya Purāṇa":       ["matsya purana"],
    "Garuḍa Purāṇa":      ["garuda purana"],
    "Brahmāṇḍa Purāṇa":   ["brahmanda purana"],
}

# Purana → concept auto-mapping
PURANA_CONCEPT_MAP = {
    "Viṣṇu Purāṇa":     ["ishvara", "bhakti", "dharma", "karma"],
    "Bhāgavata Purāṇa": ["bhakti", "ishvara", "atman", "dharma", "karma"],
    "Agni Purāṇa":      ["dharma", "karma", "tapas", "varnashrama"],
    "Garuḍa Purāṇa":   ["karma", "moksha", "dharma"],
    "Skanda Purāṇa":    ["bhakti", "ishvara", "dharma"],
    "Brahma Purāṇa":    ["brahman", "creation", "dharma"],
    "Mārkaṇḍeya Purāṇa": ["dharma", "karma", "maya"],
    "Nārada Purāṇa":    ["bhakti", "jnana", "dharma"],
}

GITA_CHAPTER_CONCEPTS = {
    1: ["dharma", "karma", "varnashrama"],
    2: ["atman", "karma_yoga", "jnana", "sankhya"],
    3: ["karma_yoga", "dharma", "varnashrama"],
    4: ["jnana", "karma_yoga", "bhakti"],
    5: ["sannyasa", "karma_yoga"],
    6: ["raja_yoga", "chitta", "dhyana"],
    7: ["maya", "jnana", "bhakti"],
    8: ["brahman", "atman", "moksha"],
    9: ["bhakti", "ishvara"],
    10: ["ishvara", "brahman"],
    11: ["ishvara", "bhakti"],
    12: ["bhakti", "karma_yoga"],
    13: ["prakriti", "purusha", "atman"],
    14: ["gunas", "moksha"],
    15: ["atman", "brahman", "maya"],
    16: ["dharma"],
    17: ["gunas", "tapas"],
    18: ["karma_yoga", "bhakti", "moksha", "dharma"],
}


def _tag_concepts(text: str, extra: list[str] = None) -> list[str]:
    """Auto-tag concepts from passage text."""
    text_l = text.lower()
    found = set(extra or [])
    for concept, keywords in CONCEPT_KEYWORD_TAGGER.items():
        if any(kw in text_l for kw in keywords):
            found.add(concept)
    return list(found)[:5]  # cap at 5 per passage


def _detect_purana(text: str) -> Optional[str]:
    """Return Purana name if detected in text."""
    text_l = text.lower()
    for name, aliases in PURANA_NAMES.items():
        if any(alias in text_l for alias in aliases):
            return name
    return None


def _clean_text(text: str) -> str:
    """Remove PDF artifacts and normalize whitespace."""
    # Remove stray whitespace and noise
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers (isolated digits)
    text = re.sub(r'^\d+\s+', '', text.strip())
    text = re.sub(r'\s+\d+\s*$', '', text.strip())
    # Remove header/footer noise
    text = re.sub(r'(The Bhagavad Gita|18 Mahāpurāṇas|T\.T\.D\.|Sacred Texts)', '', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# Bhagavad Gīta PDF Parser
# ─────────────────────────────────────────────────────────────────

# Patterns for Gita verse markers
# Matches: "Chapter 2 Verse 47", "Ch.2 V.47", "2.47", "Shloka 47"
GITA_CHAPTER_RE = re.compile(
    r'(?:Chapter|Ch\.?|CHAPTER)\s*(\d{1,2})', re.IGNORECASE)
GITA_VERSE_RE = re.compile(
    r'(?:Verse|Shloka|Sloka|V\.?)\s*(\d{1,3})|(?<!\d)(\d{1,2})\.(\d{1,3})(?!\d)',
    re.IGNORECASE)


def _infer_chapter_from_page(page_num: int, total_pages: int = 447) -> int:
    """Rough chapter inference based on page proportion (18 chapters)."""
    # Skip first ~10 pages (title/preface)
    content_page = max(0, page_num - 10)
    content_total = total_pages - 10
    return min(18, max(1, round(content_page / content_total * 18) + 1))


def parse_gita_pdf(pdf_path: str = None,
                   chunk_size: int = 4,
                   skip_pages: int = 8) -> list[dict]:
    """
    Parse Bhagavad Gita PDF into passage records.

    Strategy:
      - Merge every `chunk_size` pages into one passage block
      - Detect chapter/verse markers with regex
      - Auto-tag concepts
      - Record page range + school relevance from chapter map
    """
    if not fitz:
        print("[GitaPDF] PyMuPDF not available")
        return []

    path = Path(pdf_path or GITA_PDF)
    if not path.exists():
        print(f"[GitaPDF] File not found: {path}")
        return []

    print(f"[GitaPDF] Opening {path.name} ({path.stat().st_size // 1024} KB)...")
    doc = fitz.open(str(path))
    total = len(doc)
    print(f"[GitaPDF] {total} pages — extracting passages...")

    passages = []
    chapter = 1
    current_text = []
    current_pages = []
    chunk_start = skip_pages

    for pg_num in range(skip_pages, total):
        page_text = doc[pg_num].get_text()
        cleaned = _clean_text(page_text)

        if not cleaned or len(cleaned) < 30:
            continue

        # Chapter detection
        ch_match = GITA_CHAPTER_RE.search(cleaned)
        if ch_match:
            ch_num = int(ch_match.group(1))
            if 1 <= ch_num <= 18:
                chapter = ch_num

        current_text.append(cleaned)
        current_pages.append(pg_num + 1)

        # Flush at chunk boundary
        if len(current_pages) >= chunk_size:
            merged = " ".join(current_text)
            if len(merged) > 80:
                # Detect verse markers
                verse_match = GITA_VERSE_RE.search(merged[:50])
                verse_num = int(verse_match.group(1) or verse_match.group(3) or 0) \
                            if verse_match else 0
                verse_id  = f"BG {chapter}.{verse_num}" if verse_num else \
                            f"BG Ch{chapter} p{current_pages[0]}"

                concepts = _tag_concepts(
                    merged,
                    extra=GITA_CHAPTER_CONCEPTS.get(chapter, ["dharma"])
                )
                # School relevance by chapter
                CHAPTER_SCHOOL_LOOKUP = {
                    2:["advaita","nyaya"], 3:["mimamsa","advaita","dvaita"],
                    4:["dvaita","bhakti"], 6:["nyaya","advaita"],
                    7:["vishishtadvaita","dvaita","bhakti"],
                    9:["bhakti","vishishtadvaita"], 12:["bhakti"],
                    13:["advaita","vishishtadvaita"], 18:["advaita","dvaita","bhakti","mimamsa"]
                }
                schools = CHAPTER_SCHOOL_LOOKUP.get(chapter, ["advaita","dvaita","bhakti"])

                passages.append({
                    "text_id":        "bhagavad_gita",
                    "chapter":        chapter,
                    "verse":          verse_num,
                    "verse_id":       verse_id,
                    "sanskrit":       "",
                    "translation_en": merged[:1200],
                    "page_range":     f"{current_pages[0]}-{current_pages[-1]}",
                    "concepts":       concepts,
                    "school_relevance": schools,
                    "source":         verse_id,
                    "loader":         "pdf_gita",
                    "commentary":     "HH Sri Raghavendra Teertha's Gita Vivṛti"
                })

            current_text = []
            current_pages = []
            chunk_start = pg_num + 1

    # Flush remaining
    if current_text:
        merged = " ".join(current_text)
        if len(merged) > 80:
            concepts = _tag_concepts(merged, extra=GITA_CHAPTER_CONCEPTS.get(chapter, ["dharma"]))
            passages.append({
                "text_id":        "bhagavad_gita",
                "chapter":        chapter,
                "verse":          0,
                "verse_id":       f"BG Ch{chapter} tail",
                "sanskrit":       "",
                "translation_en": merged[:1200],
                "page_range":     f"{current_pages[0]}-{current_pages[-1]}",
                "concepts":       concepts,
                "school_relevance": ["advaita","dvaita","bhakti"],
                "source":         f"Gita Ch{chapter}",
                "loader":         "pdf_gita",
                "commentary":     "HH Sri Raghavendra Teertha's Gita Vivṛti"
            })

    doc.close()
    print(f"[GitaPDF] ✓ Extracted {len(passages)} passages from {path.name}")
    return passages


# ─────────────────────────────────────────────────────────────────
# 18 Puranas PDF Parser
# ─────────────────────────────────────────────────────────────────

def parse_puranas_pdf(pdf_path: str = None,
                      chunk_size: int = 6,
                      skip_pages: int = 3) -> list[dict]:
    """
    Parse 18 Puranas PDF into passage records.

    Strategy:
      - Detect section headers to identify which Purana we're in
      - Chunk every `chunk_size` pages into one narrative block
      - Auto-tag concepts and Purana-specific themes
    """
    if not fitz:
        return []

    path = Path(pdf_path or PURANAS_PDF)
    if not path.exists():
        print(f"[PuranasPDF] File not found: {path}")
        return []

    print(f"[PuranasPDF] Opening {path.name} ({path.stat().st_size // 1024 // 1024} MB)...")
    doc = fitz.open(str(path))
    total = len(doc)
    print(f"[PuranasPDF] {total} pages — extracting passages (this may take ~30s)...")

    passages = []
    current_purana = "Bhāgavata Purāṇa"  # default
    current_text = []
    current_pages = []
    passage_num = 0

    for pg_num in range(skip_pages, total):
        page_text = doc[pg_num].get_text()
        cleaned = _clean_text(page_text)

        if not cleaned or len(cleaned) < 30:
            continue

        # Detect Purana section
        detected = _detect_purana(cleaned)
        if detected:
            current_purana = detected

        current_text.append(cleaned)
        current_pages.append(pg_num + 1)

        if len(current_pages) >= chunk_size:
            merged = " ".join(current_text)
            if len(merged) > 100:
                passage_num += 1
                purana_id = current_purana.lower().replace(" ", "_").replace(".", "").replace("ā","a").replace("ṇ","n").replace("ḍ","d")
                purana_short = current_purana.split()[0]  # e.g. "Bhāgavata"

                extra_concepts = PURANA_CONCEPT_MAP.get(current_purana, ["dharma", "bhakti"])
                concepts = _tag_concepts(merged, extra=extra_concepts)

                passages.append({
                    "text_id":        purana_id,
                    "purana_name":    current_purana,
                    "passage_num":    passage_num,
                    "verse_id":       f"{purana_short} P{passage_num}",
                    "sanskrit":       "",
                    "translation_en": merged[:1500],
                    "page_range":     f"{current_pages[0]}-{current_pages[-1]}",
                    "concepts":       concepts,
                    "school_relevance": ["dvaita", "bhakti", "vishishtadvaita"],
                    "source":         f"{current_purana} (summary p.{current_pages[0]})",
                    "loader":         "pdf_puranas"
                })

            current_text = []
            current_pages = []

        # Progress indicator every 200 pages
        if pg_num % 200 == 0:
            print(f"  ... page {pg_num}/{total} | current: {current_purana[:30]}")

    # Flush remaining
    if current_text:
        merged = " ".join(current_text)
        if len(merged) > 100:
            passage_num += 1
            purana_short = current_purana.split()[0]
            passages.append({
                "text_id":        current_purana.lower().replace(" ", "_"),
                "purana_name":    current_purana,
                "passage_num":    passage_num,
                "verse_id":       f"{purana_short} P{passage_num}",
                "sanskrit":       "",
                "translation_en": merged[:1500],
                "page_range":     f"{current_pages[0]}-{current_pages[-1]}",
                "concepts":       _tag_concepts(merged, extra=["dharma"]),
                "school_relevance": ["bhakti", "dvaita"],
                "source":         f"{current_purana} (tail)",
                "loader":         "pdf_puranas"
            })

    doc.close()
    print(f"[PuranasPDF] ✓ Extracted {len(passages)} passages from {path.name}")
    return passages


# ─────────────────────────────────────────────────────────────────
# Merge into verse_db + concept index
# ─────────────────────────────────────────────────────────────────

def merge_into_verse_db(gita_passages: list[dict],
                         purana_passages: list[dict]) -> tuple[list, dict]:
    """Load existing verse_db + merge new passages, rebuild concept index."""
    from collections import defaultdict

    # Load existing
    verse_db_path = PROCESSED_DIR / "verse_db.json"
    existing = []
    if verse_db_path.exists():
        with open(verse_db_path, encoding="utf-8") as f:
            existing = json.load(f)

    # Merge — keep all pdf passages + existing embedded
    all_verses = existing + gita_passages + purana_passages

    # Deduplicate on verse_id
    seen = set()
    unique = []
    for v in all_verses:
        vid = v.get("verse_id", "")
        key = f"{v.get('text_id','')}__{vid}"
        if key not in seen:
            seen.add(key)
            unique.append(v)

    # Concept index
    concept_index: dict[str, list] = defaultdict(list)
    for verse in unique:
        vid = verse.get("verse_id", "")
        for concept in verse.get("concepts", []):
            if vid:
                concept_index[concept].append(vid)

    # Save
    (PROCESSED_DIR / "verse_db.json").write_text(
        json.dumps(unique, indent=2, ensure_ascii=False), encoding="utf-8")
    (PROCESSED_DIR / "concept_verse_index.json").write_text(
        json.dumps(dict(concept_index), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[PDFLoader] ✓ verse_db: {len(unique)} total passages ({len(existing)} existing + "
          f"{len(gita_passages)} gita + {len(purana_passages)} puranas)")
    print(f"[PDFLoader] ✓ concept index: {len(concept_index)} concepts")
    return unique, dict(concept_index)


def print_report(gita: list, puranas: list, verse_db: list, concept_idx: dict):
    from collections import Counter
    purana_counts = Counter(p.get("purana_name", "unknown") for p in puranas)

    print()
    print("=" * 60)
    print("  📚 PDF Corpus Ingestion Report")
    print("=" * 60)
    print(f"  Gītā passages extracted   : {len(gita)}")
    print(f"  Purāṇa passages extracted : {len(puranas)}")
    print(f"  Total verse_db size       : {len(verse_db)}")
    print(f"  Concepts indexed          : {len(concept_idx)}")
    print()
    if purana_counts:
        print("  Purāṇa coverage:")
        for name, count in sorted(purana_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name:<35} {count:>4} passages")
    print()
    top = sorted(concept_idx.items(), key=lambda x: -len(x[1]))[:8]
    print("  Top concepts by passage count:")
    for concept, vids in top:
        print(f"    {concept:<25} {len(vids):>4}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run(gita_path: str = None, puranas_path: str = None,
        inject_kg: bool = False) -> tuple[list, dict]:
    """Full PDF ingestion run."""
    print("=" * 60)
    print("  🕉️  HinduMind PDF Corpus Ingestion")
    print("=" * 60)

    # 1. Gita
    print("\n── Bhagavad Gītā PDF ──────────────────────────────────")
    gita = parse_gita_pdf(gita_path)
    if gita:
        (PROCESSED_DIR / "gita_from_pdf.json").write_text(
            json.dumps(gita, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Puranas
    print("\n── 18 Purāṇas PDF ─────────────────────────────────────")
    puranas = parse_puranas_pdf(puranas_path)
    if puranas:
        (PROCESSED_DIR / "puranas_from_pdf.json").write_text(
            json.dumps(puranas, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Merge into verse DB
    print("\n── Merging into Verse DB ───────────────────────────────")
    verse_db, concept_idx = merge_into_verse_db(gita, puranas)

    # 4. Optional KG injection
    if inject_kg:
        print("\n── Injecting into HPO KG ───────────────────────────────")
        from kg.corpus_loader import inject_into_kg
        inject_into_kg(verse_db)

    # 5. Report
    print_report(gita, puranas, verse_db, concept_idx)
    return verse_db, concept_idx


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gita",    default=None, help="Path to Gita PDF")
    parser.add_argument("--puranas", default=None, help="Path to Puranas PDF")
    parser.add_argument("--inject-kg", action="store_true")
    parser.add_argument("--gita-only", action="store_true")
    parser.add_argument("--puranas-only", action="store_true")
    args = parser.parse_args()

    if args.gita_only:
        gita = parse_gita_pdf(args.gita)
        verse_db, idx = merge_into_verse_db(gita, [])
        print_report(gita, [], verse_db, idx)
    elif args.puranas_only:
        puranas = parse_puranas_pdf(args.puranas)
        verse_db, idx = merge_into_verse_db([], puranas)
        print_report([], puranas, verse_db, idx)
    else:
        run(args.gita, args.puranas, args.inject_kg)
