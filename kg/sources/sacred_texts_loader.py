"""
sacred_texts_loader.py — Upanishad Corpus Loader

Scrapes the 10 Principal Upanishads from sacred-texts.com
(Max Müller translations — public domain).

Texts targeted:
  1. Chāndogya Upaniṣad   (CU)
  2. Bṛhadāraṇyaka Upaniṣad (BU)
  3. Aitareya Upaniṣad     (AU)
  4. Taittirīya Upaniṣad   (TU)
  5. Kena Upaniṣad         (KU)
  6. Kaṭha Upaniṣad        (KaU)
  7. Praśna Upaniṣad       (PU)
  8. Māṇḍūkya Upaniṣad     (ManU)
  9. Muṇḍaka Upaniṣad      (MuU)
  10. Śvetāśvatara Upaniṣad (SU)

Output: kg/processed/upanishads.json
"""

import json
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


OUT_DIR = Path(__file__).parent.parent / "processed"
RAW_DIR = Path(__file__).parent.parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Upanishad source URLs + metadata
# ─────────────────────────────────────────────────────────────────

UPANISHADS = [
    {
        "id":  "chandogya_up",
        "label": "Chāndogya Upaniṣad",
        "abbrev": "CU",
        "url":  "https://sacred-texts.com/hin/sbe01/index.htm",
        "school_relevance": ["advaita", "vishishtadvaita"],
        "key_concepts": ["atman", "brahman", "prana", "satya"],
        "mahavakya": "Tat tvam asi (CU 6.8.7)"
    },
    {
        "id":  "brihadaranyaka_up",
        "label": "Bṛhadāraṇyaka Upaniṣad",
        "abbrev": "BU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["advaita"],
        "key_concepts": ["atman", "brahman", "neti neti"],
        "mahavakya": "Ahaṃ brahmāsmi (BU 1.4.10)"
    },
    {
        "id":  "katha_up",
        "label": "Kaṭha Upaniṣad",
        "abbrev": "KaU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["advaita", "nyaya"],
        "key_concepts": ["atman", "moksha", "nachiketa"],
        "mahavakya": "nāyam ātmā pravacanena labhyaḥ (KaU 1.2.23)"
    },
    {
        "id":  "mandukya_up",
        "label": "Māṇḍūkya Upaniṣad",
        "abbrev": "ManU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["advaita"],
        "key_concepts": ["atman", "brahman", "aum", "turiya"],
        "mahavakya": "Ayam ātmā brahma (ManU 1.2)"
    },
    {
        "id":  "aitareya_up",
        "label": "Aitareya Upaniṣad",
        "abbrev": "AU",
        "url":  "https://sacred-texts.com/hin/sbe01/index.htm",
        "school_relevance": ["advaita"],
        "key_concepts": ["brahman", "prajnana"],
        "mahavakya": "Prajñānaṃ brahma (AU 3.3)"
    },
    {
        "id":  "taittiriya_up",
        "label": "Taittirīya Upaniṣad",
        "abbrev": "TU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["advaita", "vishishtadvaita"],
        "key_concepts": ["brahman", "ananda", "atman", "panchakosa"],
        "mahavakya": "satyaṃ jñānam anantaṃ brahma (TU 2.1.1)"
    },
    {
        "id":  "mundaka_up",
        "label": "Muṇḍaka Upaniṣad",
        "abbrev": "MuU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["advaita"],
        "key_concepts": ["para_vidya", "apara_vidya", "brahman", "atman"]
    },
    {
        "id":  "prashna_up",
        "label": "Praśna Upaniṣad",
        "abbrev": "PU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["advaita", "nyaya"],
        "key_concepts": ["prana", "atman", "brahman"]
    },
    {
        "id":  "kena_up",
        "label": "Kena Upaniṣad",
        "abbrev": "KU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["advaita"],
        "key_concepts": ["brahman", "atman", "prana"]
    },
    {
        "id":  "shvetashvatara_up",
        "label": "Śvetāśvatara Upaniṣad",
        "abbrev": "SU",
        "url":  "https://sacred-texts.com/hin/sbe15/index.htm",
        "school_relevance": ["dvaita", "vishishtadvaita"],
        "key_concepts": ["ishvara", "purusha", "prakriti", "brahman"]
    },
]


# ─────────────────────────────────────────────────────────────────
# Embedded Core Passages (offline usable immediately)
# ─────────────────────────────────────────────────────────────────

UPANISHAD_CORE_PASSAGES = [
    {"text_id":"chandogya_up","section":"6.8.7","verse_id":"CU 6.8.7",
     "sanskrit":"tat tvam asi śvetaketo",
     "translation_en":"That thou art, O Śvetaketu (= you are Brahman, the Self of all).",
     "concepts":["atman","brahman","advaita"],"school_relevance":["advaita"],
     "source":"CU 6.8.7","loader":"embedded",
     "significance":"Mahāvākya: establishes ātman-brahman identity (Advaita)"},

    {"text_id":"brihadaranyaka_up","section":"1.4.10","verse_id":"BU 1.4.10",
     "sanskrit":"ahaṃ brahmāsmi",
     "translation_en":"I am Brahman.",
     "concepts":["atman","brahman","advaita"],"school_relevance":["advaita"],
     "source":"BU 1.4.10","loader":"embedded",
     "significance":"Mahāvākya: first-person declaration of ātman-brahman identity"},

    {"text_id":"mandukya_up","section":"1.2","verse_id":"ManU 1.2",
     "sanskrit":"ayam ātmā brahma",
     "translation_en":"This ātman is Brahman.",
     "concepts":["atman","brahman","advaita"],"school_relevance":["advaita"],
     "source":"ManU 1.2","loader":"embedded",
     "significance":"Mahāvākya: third-person affirmation of non-dual identity"},

    {"text_id":"aitareya_up","section":"3.3","verse_id":"AU 3.3",
     "sanskrit":"prajñānaṃ brahma",
     "translation_en":"Consciousness (prajñāna) is Brahman.",
     "concepts":["brahman","jnana","consciousness"],"school_relevance":["advaita"],
     "source":"AU 3.3","loader":"embedded",
     "significance":"Mahāvākya: identifies pure awareness with Brahman"},

    {"text_id":"taittiriya_up","section":"2.1.1","verse_id":"TU 2.1.1",
     "sanskrit":"satyaṃ jñānam anantaṃ brahma",
     "translation_en":"Brahman is Truth, Knowledge, and Infinity.",
     "concepts":["brahman","satya","jnana","ananda"],"school_relevance":["advaita","vishishtadvaita"],
     "source":"TU 2.1.1","loader":"embedded"},

    {"text_id":"katha_up","section":"1.2.23","verse_id":"KaU 1.2.23",
     "sanskrit":"nāyam ātmā pravacanena labhyaḥ na medhayā na bahunā śrutena",
     "translation_en":"The ātman is not attained through discourse, nor through intellect, nor through much learning.",
     "concepts":["atman","moksha","jnana"],"school_relevance":["advaita","bhakti"],
     "source":"KaU 1.2.23","loader":"embedded"},

    {"text_id":"katha_up","section":"1.3.3","verse_id":"KaU 1.3.3",
     "sanskrit":"ātmānaṃ rathinaṃ viddhi śarīraṃ ratham eva ca",
     "translation_en":"Know the ātman as the lord of the chariot, and the body as the chariot.",
     "concepts":["atman","prakriti","raja_yoga","chitta"],"school_relevance":["advaita","nyaya"],
     "source":"KaU 1.3.3","loader":"embedded",
     "significance":"Chariot analogy: mind=reins, intellect=charioteer, senses=horses"},

    {"text_id":"chandogya_up","section":"8.7.1","verse_id":"CU 8.7.1",
     "sanskrit":"ya ātmā apahatapāpmā vijaro vimṛtyur viśoko vijighatso'pipāsaḥ",
     "translation_en":"The ātman which is free from sin, old age, death, grief, hunger, and thirst — that should be sought after and understood.",
     "concepts":["atman","moksha","brahman"],"school_relevance":["advaita"],
     "source":"CU 8.7.1","loader":"embedded"},

    {"text_id":"shvetashvatara_up","section":"6.11","verse_id":"SU 6.11",
     "sanskrit":"eko devaḥ sarvabhūteṣu gūḍhaḥ sarvavyāpī sarvabhūtāntarātmā",
     "translation_en":"The one God, hidden in all beings, pervading all, the inner soul of all.",
     "concepts":["ishvara","brahman","atman"],"school_relevance":["vishishtadvaita","dvaita"],
     "source":"SU 6.11","loader":"embedded"},

    {"text_id":"mundaka_up","section":"2.2.5","verse_id":"MuU 2.2.5",
     "sanskrit":"brahmaiva idaṃ amṛtaṃ purastad brahma paścād brahma dakṣiṇataś cottareṇa",
     "translation_en":"Brahman, indeed, is all this immortal; in front, Brahman behind, to the right and to the left, above and below.",
     "concepts":["brahman","atman","advaita"],"school_relevance":["advaita"],
     "source":"MuU 2.2.5","loader":"embedded"},
]


# ─────────────────────────────────────────────────────────────────
# Scraper
# ─────────────────────────────────────────────────────────────────

def scrape_upanishad(meta: dict, delay: float = 2.0) -> list[dict]:
    """Scrape a single Upanishad from sacred-texts.com."""
    if not requests or not BeautifulSoup:
        return []
    passages = []
    try:
        r = requests.get(meta["url"], timeout=15,
                         headers={"User-Agent": "HinduMind-Research/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        paras = soup.find_all(["p", "blockquote"])
        section = 0
        passage_num = 0
        for para in paras:
            text = para.get_text(separator=" ").strip()
            if len(text) < 40:
                continue
            # Skip navigation/header noise
            if any(skip in text.lower() for skip in ["sacred-texts", "index", "copyright"]):
                continue

            passage_num += 1
            passages.append({
                "text_id": meta["id"],
                "section": str(passage_num),
                "verse_id": f"{meta['abbrev']} {passage_num}",
                "sanskrit": "",
                "translation_en": text[:800],
                "concepts": meta["key_concepts"][:2],
                "school_relevance": meta["school_relevance"],
                "source": f"{meta['abbrev']} {passage_num}",
                "loader": "sacred_texts_scrape"
            })
        time.sleep(delay)
    except Exception as e:
        print(f"  [WARN] {meta['id']}: {e}")
    return passages


def load_upanishads(scrape: bool = True,
                    use_embedded_fallback: bool = True) -> list[dict]:
    """
    Load the 10 principal Upanishads.

    Priority:
      1. Scrape sacred-texts.com
      2. Embedded core passages (offline fallback)
    """
    print("[UpanishadLoader] Starting Upanishad ingestion...")
    passages = []

    if scrape and requests and BeautifulSoup:
        for meta in UPANISHADS:
            print(f"  Scraping {meta['label']}...", end=" ", flush=True)
            scraped = scrape_upanishad(meta)
            passages.extend(scraped)
            print(f"✓ ({len(scraped)} passages)")

    if not passages and use_embedded_fallback:
        print("[UpanishadLoader] Using embedded core passages fallback...")
        passages = UPANISHAD_CORE_PASSAGES

    if passages:
        out_path = OUT_DIR / "upanishads.json"
        out_path.write_text(
            json.dumps(passages, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[UpanishadLoader] ✓ {len(passages)} passages → {out_path}")

    return passages


if __name__ == "__main__":
    load_upanishads(scrape=True, use_embedded_fallback=True)
