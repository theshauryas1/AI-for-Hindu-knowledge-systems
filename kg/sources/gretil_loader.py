"""
gretil_loader.py — GRETIL Sanskrit Text Repository Loader

Downloads texts from the Göttingen Register of Electronic Texts
in Indian Languages (GRETIL): http://gretil.sub.uni-goettingen.de

Texts targeted:
  - Manusmṛti            (Dharmaśāstra)
  - Brahma Sūtras        (Vedānta foundation)
  - Yoga Sūtras          (Patañjali)
  - Arthaśāstra          (Kauṭilya — governance)
  - Nyāya Sūtras         (Gautama)
  - Bṛhaspati Sūtra      (Cārvāka — for contrast)

Output: kg/processed/gretil_texts.json
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import requests
except ImportError:
    requests = None


OUT_DIR = Path(__file__).parent.parent / "processed"
RAW_DIR = Path(__file__).parent.parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

GRETIL_BASE = "http://gretil.sub.uni-goettingen.de/gretil"


# ─────────────────────────────────────────────────────────────────
# GRETIL Text Registry
# ─────────────────────────────────────────────────────────────────

GRETIL_TEXTS = [
    {
        "id": "manusmriti",
        "label": "Manusmṛti",
        "abbrev": "MS",
        "url": f"{GRETIL_BASE}/1_sanskr/6_sastra/3_dharma/manusmrtu.htm",
        "school_relevance": ["mimamsa", "nyaya", "dvaita"],
        "key_concepts": ["dharma", "varnashrama", "karma", "ahimsa"],
        "description": "Primary Dharmaśāstra text — rules for dharmic life by varna and ashrama"
    },
    {
        "id": "brahma_sutras",
        "label": "Brahma Sūtras",
        "abbrev": "BS",
        "url": f"{GRETIL_BASE}/1_sanskr/6_sastra/1_vedanta/brahmsuu.htm",
        "school_relevance": ["advaita", "vishishtadvaita", "dvaita"],
        "key_concepts": ["brahman", "atman", "vedanta", "moksha"],
        "description": "Systematizes Upanishadic teaching; commented on by all Vedānta schools"
    },
    {
        "id": "yoga_sutras",
        "label": "Yoga Sūtras",
        "abbrev": "YS",
        "url": f"{GRETIL_BASE}/1_sanskr/6_sastra/4_yoga/yogasutu.htm",
        "school_relevance": ["nyaya", "advaita"],
        "key_concepts": ["chitta", "raja_yoga", "dharana", "samadhi", "ahimsa", "satya"],
        "description": "Patañjali's systematic yoga — 8-limb path including yamas/niyamas"
    },
    {
        "id": "arthashastra",
        "label": "Arthaśāstra",
        "abbrev": "AS",
        "url": f"{GRETIL_BASE}/1_sanskr/6_sastra/8_nitisastra/arthashu.htm",
        "school_relevance": ["nyaya", "mimamsa"],
        "key_concepts": ["varnashrama", "dharma", "governance", "raja_dharma"],
        "description": "Kauṭilya's treatise on statecraft — king's duties, taxation, punishment"
    },
    {
        "id": "nyaya_sutras",
        "label": "Nyāya Sūtras",
        "abbrev": "NS",
        "url": f"{GRETIL_BASE}/1_sanskr/6_sastra/5_nyaya/nyayasuu.htm",
        "school_relevance": ["nyaya"],
        "key_concepts": ["pramana", "anumana", "pratyaksha", "ishvara", "atman"],
        "description": "Gautama's foundational text on logic, epistemology, and valid inference"
    },
    {
        "id": "mimamsa_sutras",
        "label": "Mīmāṃsā Sūtras",
        "abbrev": "MiS",
        "url": f"{GRETIL_BASE}/1_sanskr/6_sastra/6_mimamsa/mimsuu1u.htm",
        "school_relevance": ["mimamsa"],
        "key_concepts": ["dharma", "vedic_authority", "karma", "ritual"],
        "description": "Jaiminī's systematization of Vedic hermeneutics and ritual dharma"
    },
]


# ─────────────────────────────────────────────────────────────────
# Embedded Key Passages (offline fallback)
# ─────────────────────────────────────────────────────────────────

GRETIL_EMBEDDED_PASSAGES = [
    # Manusmriti
    {"text_id":"manusmriti","section":"1.88","verse_id":"MS 1.88",
     "sanskrit":"adhyāpanaṃ brahmayajño pitṛyajñaś ca tarpaṇam",
     "translation_en":"Teaching, studying the Vedas, performing sacrifices, giving gifts — these are the duties of the Brahmin.",
     "concepts":["dharma","varnashrama"],"school_relevance":["mimamsa","nyaya"],
     "source":"MS 1.88","loader":"embedded"},

    {"text_id":"manusmriti","section":"5.45","verse_id":"MS 5.45",
     "sanskrit":"nākṛtvā prāṇināṃ hiṃsāṃ māṃsam utpadyate kvacit",
     "translation_en":"Meat cannot be obtained without injury to living beings; killing living beings is against dharmic prosperity; therefore one should abstain from meat.",
     "concepts":["ahimsa","dharma","karma"],"school_relevance":["advaita","mimamsa"],
     "source":"MS 5.45","loader":"embedded"},

    {"text_id":"manusmriti","section":"7.2","verse_id":"MS 7.2",
     "sanskrit":"kṣatriyasya paraṃ dharmaṃ prajānāṃ ca parirakṣaṇam",
     "translation_en":"The highest dharma of the Kṣatriya is the protection of his subjects.",
     "concepts":["dharma","varnashrama","karma_yoga"],"school_relevance":["mimamsa","nyaya"],
     "source":"MS 7.2","loader":"embedded"},

    {"text_id":"manusmriti","section":"8.15","verse_id":"MS 8.15",
     "sanskrit":"yas tu doṣam ajānānaḥ sāhasaṃ kurute naraḥ",
     "translation_en":"A king who, without investigation, punishes innocent men, quickly departs from fame and goes to hell.",
     "concepts":["dharma","karma","varnashrama","satya"],"school_relevance":["nyaya","mimamsa"],
     "source":"MS 8.15","loader":"embedded"},

    {"text_id":"manusmriti","section":"8.349","verse_id":"MS 8.349",
     "translation_en":"A man who has committed an illegal act, wounded, or has been struck, and kills his attacker — is not guilty of killing.",
     "concepts":["dharma","ahimsa","karma"],"school_relevance":["nyaya","mimamsa"],
     "source":"MS 8.349","loader":"embedded"},

    # Yoga Sutras
    {"text_id":"yoga_sutras","section":"1.2","verse_id":"YS 1.2",
     "sanskrit":"yogaś citta-vṛtti-nirodhaḥ",
     "translation_en":"Yoga is the cessation of the modifications (vṛttis) of the mind (citta).",
     "concepts":["chitta","raja_yoga","vairagya"],"school_relevance":["nyaya","advaita"],
     "source":"YS 1.2","loader":"embedded"},

    {"text_id":"yoga_sutras","section":"2.30","verse_id":"YS 2.30",
     "sanskrit":"ahiṃsā satyam asteyam brahmacaryam aparigrahaḥ yamāḥ",
     "translation_en":"Non-violence, truthfulness, non-stealing, celibacy, non-possessiveness — these are the five yamas (restraints).",
     "concepts":["ahimsa","satya","asteya","brahmacharya","aparigraha"],
     "school_relevance":["advaita","nyaya","bhakti"],
     "source":"YS 2.30","loader":"embedded"},

    {"text_id":"yoga_sutras","section":"2.31","verse_id":"YS 2.31",
     "sanskrit":"jāti deśa kāla samayānavacchinnāḥ sārvabhaumā mahāvratam",
     "translation_en":"These (yamas), not conditioned by class, place, time, or consequence, are the great vow (mahāvrata), universal in scope.",
     "concepts":["ahimsa","dharma","satya"],"school_relevance":["advaita","nyaya"],
     "source":"YS 2.31","loader":"embedded"},

    # Brahma Sutras
    {"text_id":"brahma_sutras","section":"1.1.1","verse_id":"BS 1.1.1",
     "sanskrit":"athāto brahmajijñāsā",
     "translation_en":"Now, therefore, the inquiry into Brahman.",
     "concepts":["brahman","jnana","vedanta"],"school_relevance":["advaita","vishishtadvaita","dvaita"],
     "source":"BS 1.1.1","loader":"embedded"},

    {"text_id":"brahma_sutras","section":"1.1.2","verse_id":"BS 1.1.2",
     "sanskrit":"janmādy asya yataḥ",
     "translation_en":"Brahman is that from which the origin, sustenance, and dissolution of this universe proceed.",
     "concepts":["brahman","ishvara","prakriti"],"school_relevance":["advaita","dvaita","vishishtadvaita"],
     "source":"BS 1.1.2","loader":"embedded"},

    # Nyaya Sutras
    {"text_id":"nyaya_sutras","section":"1.1.1","verse_id":"NS 1.1.1",
     "sanskrit":"pramāṇa prameya saṃśaya prayojana",
     "translation_en":"The means of right knowledge, the objects of right knowledge, doubt, purpose (are the sixteen categories of Nyāya).",
     "concepts":["pramana","jnana","anumana"],"school_relevance":["nyaya"],
     "source":"NS 1.1.1","loader":"embedded"},

    # Arthashastra
    {"text_id":"arthashastra","section":"1.19","verse_id":"AS 1.19",
     "translation_en":"The king shall consider as good anything that pleases his subjects. The root of happiness is virtue (dharma); the root of virtue is wealth (artha); the root of wealth is an effective administration.",
     "concepts":["dharma","karma","varnashrama"],"school_relevance":["nyaya","mimamsa"],
     "source":"AS 1.19","loader":"embedded"},
]


# ─────────────────────────────────────────────────────────────────
# Downloader (one text at a time)
# ─────────────────────────────────────────────────────────────────

def _parse_gretil_html(html: str, meta: dict) -> list[dict]:
    """
    Parse raw GRETIL HTML into passage list.
    GRETIL texts vary in structure — this does best-effort parsing.
    """
    try:
        from bs4 import BeautifulSoup as BS
    except ImportError:
        return []

    soup = BS(html, "html.parser")
    passages = []
    paras = soup.find_all(["p", "div"])

    verse_num = 0
    for para in paras:
        text = para.get_text(separator=" ").strip()
        if len(text) < 30 or len(text) > 2000:
            continue
        if any(skip in text.lower() for skip in ["gretil", "©", "encoded by", "prepared by"]):
            continue

        verse_num += 1
        # Try to detect section numbers (common pattern: "1.1" or "[1.1]")
        m = re.search(r"\[?(\d+\.\d+(?:\.\d+)?)\]?", text[:30])
        section = m.group(1) if m else str(verse_num)

        passages.append({
            "text_id": meta["id"],
            "section": section,
            "verse_id": f"{meta['abbrev']} {section}",
            "sanskrit": text if _is_sanskrit(text) else "",
            "translation_en": text if not _is_sanskrit(text) else "",
            "concepts": meta["key_concepts"][:2],
            "school_relevance": meta["school_relevance"],
            "source": f"{meta['abbrev']} {section}",
            "loader": "gretil_scrape"
        })

    return passages


def _is_sanskrit(text: str) -> bool:
    """Rough heuristic: check for IAST diacritic characters."""
    iast_chars = "āīūṛṝḷṃḥśṣṭḍṇ"
    return sum(1 for c in text if c in iast_chars) > 2


def download_gretil_text(meta: dict, delay: float = 2.0) -> list[dict]:
    """Download and parse a single GRETIL text."""
    if not requests:
        return []
    try:
        print(f"  Downloading {meta['label']}...", end=" ", flush=True)
        r = requests.get(meta["url"], timeout=20,
                         headers={"User-Agent": "HinduMind-Research/1.0"})
        r.raise_for_status()

        # Cache raw HTML
        raw_path = RAW_DIR / f"{meta['id']}.html"
        raw_path.write_text(r.text, encoding="utf-8", errors="replace")

        passages = _parse_gretil_html(r.text, meta)
        print(f"✓ ({len(passages)} passages)")
        time.sleep(delay)
        return passages

    except Exception as e:
        print(f"\n  [WARN] {meta['id']}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# Master load function
# ─────────────────────────────────────────────────────────────────

def load_gretil_texts(scrape: bool = True,
                      texts: list[str] = None,
                      use_embedded_fallback: bool = True) -> list[dict]:
    """
    Load texts from GRETIL.

    texts : list of text IDs to load (default: all)
    """
    print("[GRETILLoader] Starting GRETIL ingestion...")
    all_passages = []

    targets = [m for m in GRETIL_TEXTS
               if texts is None or m["id"] in texts]

    if scrape and requests:
        for meta in targets:
            passages = download_gretil_text(meta)
            all_passages.extend(passages)

    if not all_passages and use_embedded_fallback:
        print("[GRETILLoader] Using embedded core passage fallback...")
        if texts:
            all_passages = [p for p in GRETIL_EMBEDDED_PASSAGES
                            if p["text_id"] in texts]
        else:
            all_passages = GRETIL_EMBEDDED_PASSAGES

    if all_passages:
        out_path = OUT_DIR / "gretil_texts.json"
        out_path.write_text(
            json.dumps(all_passages, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[GRETILLoader] ✓ {len(all_passages)} passages → {out_path}")

    return all_passages


if __name__ == "__main__":
    load_gretil_texts(scrape=True, use_embedded_fallback=True)
