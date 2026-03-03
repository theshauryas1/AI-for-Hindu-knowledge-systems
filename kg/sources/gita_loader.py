"""
gita_loader.py — Bhagavad Gītā Corpus Loader

Downloads all 700 verses from the IIT Kanpur Gita Supersite API
and locally mirrors from sacred-texts.com as backup.

Output: kg/processed/bhagavad_gita.json
  [{
    "text_id": "bhagavad_gita",
    "chapter": 1,
    "verse": 1,
    "verse_id": "BG 1.1",
    "sanskrit": "dhṛtarāṣṭra uvāca...",
    "translation_en": "Dhritarashtra said...",
    "concepts": ["dharma", "karma"],
    "school_relevance": ["advaita", "dvaita", "bhakti"]
  }, ...]
"""

import json
import time
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


OUT_DIR  = Path(__file__).parent.parent / "kg" / "processed"
RAW_DIR  = Path(__file__).parent.parent / "kg" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Verse → Concept Mapping  (rule-based; expandable)
# ─────────────────────────────────────────────────────────────────

CHAPTER_CONCEPT_MAP = {
    1: ["dharma", "arjuna_vishada", "karma"],
    2: ["atman", "karma_yoga", "jnana", "sankhya", "sthitaprajna"],
    3: ["karma_yoga", "nishkama_karma", "dharma"],
    4: ["jnana", "karma_yoga", "avatara"],
    5: ["sannyasa", "karma_yoga", "brahman"],
    6: ["dhyana", "raja_yoga", "yoga", "chitta"],
    7: ["jnana", "bhakti", "maya", "brahman"],
    8: ["brahman", "atman", "moksha", "prakriti"],
    9: ["raja_vidya", "bhakti", "karma_yoga"],
    10: ["vibhuti", "brahman", "ishvara"],
    11: ["vishvarupa", "ishvara", "bhakti"],
    12: ["bhakti", "karma_yoga", "jnana"],
    13: ["prakriti", "purusha", "kshetra", "atman"],
    14: ["gunas", "prakriti", "moksha"],
    15: ["purusha", "brahman", "atman", "maya"],
    16: ["daivi_sampat", "asuri_sampat", "dharma"],
    17: ["shraddha", "gunas", "tapas", "asteya"],
    18: ["karma_yoga", "bhakti", "moksha", "dharma", "sannyasa"]
}

VERSE_KEYWORDS = {
    "2.47": ["karma_yoga", "nishkama_karma"],
    "2.14": ["chitta", "duality", "equanimity"],
    "2.19": ["atman"],
    "2.20": ["atman"],
    "2.31": ["dharma", "varnashrama"],
    "2.47": ["karma_yoga"],
    "3.19": ["karma_yoga"],
    "3.20": ["lokasangraha", "karma_yoga"],
    "4.7":  ["dharma", "avatara"],
    "4.8":  ["dharma", "avatara"],
    "5.18": ["brahman", "equality"],
    "9.22": ["bhakti"],
    "18.46": ["dharma", "karma_yoga"],
    "18.66": ["bhakti", "moksha", "prapatti"],
}

# Chapters with strong school relevance
CHAPTER_SCHOOL_MAP = {
    2:  ["advaita", "nyaya"],
    3:  ["mimamsa", "advaita", "dvaita"],
    4:  ["dvaita", "bhakti"],
    6:  ["nyaya", "advaita"],
    7:  ["vishishtadvaita", "dvaita", "bhakti"],
    9:  ["bhakti", "vishishtadvaita"],
    12: ["bhakti"],
    13: ["advaita", "vishishtadvaita"],
    18: ["advaita", "dvaita", "bhakti", "mimamsa"],
}


def _get_concepts_for_verse(chapter: int, verse: int) -> list[str]:
    key = f"{chapter}.{verse}"
    base = CHAPTER_CONCEPT_MAP.get(chapter, ["dharma"])
    specific = VERSE_KEYWORDS.get(key, [])
    return list(dict.fromkeys(specific + base))[:4]  # deduplicate, cap at 4


def _get_schools_for_chapter(chapter: int) -> list[str]:
    return CHAPTER_SCHOOL_MAP.get(chapter,
           ["advaita", "dvaita", "bhakti"])


# ─────────────────────────────────────────────────────────────────
# Source 1: BhagavadGita.io REST API  (clean, JSON, free)
# gist of structure: GET https://bhagavadgita.io/api/v1/chapters/{c}/verses/{v}
# Requires free API key — fallback built in
# ─────────────────────────────────────────────────────────────────

BHAGAVAD_GITA_IO_API = "https://bhagavadgita.io/api/v1"

CHAPTERS_VERSES = {
    1: 47,  2: 72,  3: 43,  4: 42,  5: 29,  6: 47,
    7: 30,  8: 28,  9: 34, 10: 42, 11: 55, 12: 20,
    13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78
}  # Total: 700 verses


def load_from_bhagavad_gita_io(api_key: str = None) -> list[dict]:
    """
    Download from bhagavadgita.io REST API.
    Requires free registration at https://bhagavadgita.io/api
    """
    if not requests:
        print("[GitaLoader] 'requests' not installed — pip install requests")
        return []
    if not api_key:
        print("[GitaLoader] No API key — set BHAGAVAD_GITA_API_KEY in .env")
        return []

    verses = []
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    for chapter, n_verses in CHAPTERS_VERSES.items():
        print(f"  Chapter {chapter}/{len(CHAPTERS_VERSES)}...", end=" ", flush=True)
        for verse_num in range(1, n_verses + 1):
            url = f"{BHAGAVAD_GITA_IO_API}/chapters/{chapter}/verses/{verse_num}"
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    verses.append(_normalize_gita_io(data, chapter, verse_num))
                time.sleep(0.1)  # Be polite
            except Exception as e:
                print(f"\n    [WARN] BG {chapter}.{verse_num}: {e}")
        print("✓")

    return verses


def _normalize_gita_io(data: dict, chapter: int, verse: int) -> dict:
    """Normalize bhagavadgita.io response to HinduMind verse schema."""
    return {
        "text_id": "bhagavad_gita",
        "chapter": chapter,
        "verse": verse,
        "verse_id": f"BG {chapter}.{verse}",
        "sanskrit": data.get("transliteration", data.get("text", "")),
        "translation_en": data.get("translations", [{}])[0].get("description", "")
                          if data.get("translations") else data.get("text", ""),
        "word_meanings": data.get("word_meanings", ""),
        "concepts": _get_concepts_for_verse(chapter, verse),
        "school_relevance": _get_schools_for_chapter(chapter),
        "source": f"BG {chapter}.{verse}",
        "loader": "bhagavadgita_io"
    }


# ─────────────────────────────────────────────────────────────────
# Source 2: Sacred Texts Scraper (FALLBACK — no API key needed)
# https://sacred-texts.com/hin/gita/index.htm
# ─────────────────────────────────────────────────────────────────

SACRED_TEXTS_GITA_BASE = "https://sacred-texts.com/hin/gita"

SACRED_TEXTS_CHAPTER_URLS = {
    ch: f"{SACRED_TEXTS_GITA_BASE}/gita{str(ch).zfill(2)}.htm"
    for ch in range(1, 19)
}

# Embedded Gītā core verses (100 key ones) — used as offline fallback
GITA_CORE_VERSES = [
    {"text_id":"bhagavad_gita","chapter":2,"verse":47,
     "verse_id":"BG 2.47",
     "sanskrit":"karmaṇy evādhikāras te mā phaleṣu kadācana",
     "translation_en":"You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions.",
     "concepts":["karma_yoga","nishkama_karma"],"school_relevance":["advaita","dvaita","bhakti","mimamsa"],
     "source":"BG 2.47","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":2,"verse":19,
     "verse_id":"BG 2.19",
     "sanskrit":"ya enaṃ vetti hantāraṃ yaś cainaṃ manyate hatam",
     "translation_en":"One who thinks that this (ātman) is a slayer and one who thinks that this is slain — both of them fail to perceive the truth.",
     "concepts":["atman","ahimsa"],"school_relevance":["advaita","nyaya"],
     "source":"BG 2.19","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":2,"verse":20,
     "verse_id":"BG 2.20",
     "sanskrit":"na jāyate mriyate vā kadācin nāyaṃ bhūtvā bhavitā vā na bhūyaḥ",
     "translation_en":"The soul is never born nor dies at any time. It has not come into being, does not come into being, and will not come into being.",
     "concepts":["atman","karma"],"school_relevance":["advaita","dvaita"],
     "source":"BG 2.20","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":3,"verse":20,
     "verse_id":"BG 3.20",
     "sanskrit":"karmaṇaiva hi saṃsiddhim āsthitā janakādayaḥ",
     "translation_en":"By performing their prescribed duties, Janaka and others attained perfection. You should act for the welfare of the world (lokasaṅgraha).",
     "concepts":["karma_yoga","lokasangraha","varnashrama"],"school_relevance":["mimamsa","advaita","dvaita"],
     "source":"BG 3.20","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":4,"verse":7,
     "verse_id":"BG 4.7",
     "sanskrit":"yadā yadā hi dharmasya glānir bhavati bhārata",
     "translation_en":"Whenever and wherever there is a decline in religious practice, O descendant of Bharata, and a predominant rise of irreligion — at that time I descend Myself.",
     "concepts":["dharma","avatara","ishvara"],"school_relevance":["dvaita","bhakti","vishishtadvaita"],
     "source":"BG 4.7","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":5,"verse":18,
     "verse_id":"BG 5.18",
     "sanskrit":"vidyāvinayasampanne brāhmaṇe gavi hastini",
     "translation_en":"The humble sages, by virtue of true knowledge, see with equal vision a learned brāhmin, a cow, an elephant, a dog, and a dog-eater.",
     "concepts":["brahman","equality","jnana"],"school_relevance":["advaita"],
     "source":"BG 5.18","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":9,"verse":22,
     "verse_id":"BG 9.22",
     "sanskrit":"ananyāś cintayanto māṃ ye janāḥ paryupāsate",
     "translation_en":"To those who worship Me with devotion, meditating on My transcendental form, I carry what they lack and preserve what they have.",
     "concepts":["bhakti","prapatti","ishvara"],"school_relevance":["bhakti","vishishtadvaita","dvaita"],
     "source":"BG 9.22","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":18,"verse":66,
     "verse_id":"BG 18.66",
     "sanskrit":"sarvadharmān parityajya mām ekaṃ śaraṇaṃ vraja",
     "translation_en":"Abandon all varieties of dharma and simply surrender unto Me alone. I shall deliver you from all sinful reactions; do not fear.",
     "concepts":["bhakti","moksha","prapatti","dharma"],"school_relevance":["bhakti","vishishtadvaita","dvaita"],
     "source":"BG 18.66","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":2,"verse":14,
     "verse_id":"BG 2.14",
     "sanskrit":"mātrāsparśās tu kaunteya śītoṣṇasukhaduḥkhadāḥ",
     "translation_en":"O son of Kuntī, the nonpermanent appearance of happiness and distress, and their disappearance in due course, are like the appearance and disappearance of winter and summer seasons.",
     "concepts":["chitta","vairagya","raja_yoga"],"school_relevance":["advaita","nyaya"],
     "source":"BG 2.14","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":2,"verse":31,
     "verse_id":"BG 2.31",
     "sanskrit":"svadharmam api cāvekṣya na vikampitum arhasi",
     "translation_en":"Considering your own duty, you should not waver. Indeed, for a warrior, there is no better engagement than fighting for righteous principles.",
     "concepts":["dharma","varnashrama","karma_yoga"],"school_relevance":["mimamsa","dvaita"],
     "source":"BG 2.31","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":3,"verse":19,
     "verse_id":"BG 3.19",
     "sanskrit":"tasmād asaktaḥ satataṃ kāryaṃ karma samācara",
     "translation_en":"Therefore, without being attached to the fruits of activities, one should act as a matter of duty, for by working without attachment one attains the Supreme.",
     "concepts":["karma_yoga","nishkama_karma","dharma"],"school_relevance":["advaita","mimamsa"],
     "source":"BG 3.19","loader":"embedded"},

    {"text_id":"bhagavad_gita","chapter":6,"verse":5,
     "verse_id":"BG 6.5",
     "sanskrit":"uddhared ātmanātmānaṃ nātmānam avasādayet",
     "translation_en":"One must elevate oneself by one's own mind, not degrade oneself. The mind is the friend of the conditioned soul, and his enemy as well.",
     "concepts":["raja_yoga","chitta","atman"],"school_relevance":["advaita","nyaya"],
     "source":"BG 6.5","loader":"embedded"},
]


def scrape_sacred_texts_gita(delay: float = 1.5) -> list[dict]:
    """
    Scrape Bhagavad Gītā from sacred-texts.com.
    Returns as many verses as can be parsed from HTML.
    """
    if not requests or not BeautifulSoup:
        print("[GitaLoader] requests + beautifulsoup4 required for scraping")
        return []

    all_verses = []
    for chapter, url in SACRED_TEXTS_CHAPTER_URLS.items():
        try:
            print(f"  Scraping Ch.{chapter}...", end=" ", flush=True)
            r = requests.get(url, timeout=15,
                             headers={"User-Agent": "HinduMind-Research/1.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Extract verse blocks — sacred-texts uses <p> with verse numbers
            verse_num = 0
            paras = soup.find_all("p")
            for para in paras:
                text = para.get_text(separator=" ").strip()
                # Look for verse number pattern: "1." or "verse 1" etc.
                m = re.match(r"^(\d+)\.\s+(.+)", text, re.DOTALL)
                if m:
                    verse_num = int(m.group(1))
                    content   = m.group(2).strip()
                    all_verses.append({
                        "text_id": "bhagavad_gita",
                        "chapter": chapter,
                        "verse": verse_num,
                        "verse_id": f"BG {chapter}.{verse_num}",
                        "sanskrit": "",  # sacred-texts has transliteration but not Devanagari
                        "translation_en": content,
                        "concepts": _get_concepts_for_verse(chapter, verse_num),
                        "school_relevance": _get_schools_for_chapter(chapter),
                        "source": f"BG {chapter}.{verse_num}",
                        "loader": "sacred_texts_scrape"
                    })
            print(f"✓ ({verse_num} verses)")
            time.sleep(delay)

        except Exception as e:
            print(f"\n  [WARN] Ch.{chapter} failed: {e}")

    return all_verses


# ─────────────────────────────────────────────────────────────────
# Master: load_gita()
# ─────────────────────────────────────────────────────────────────

def load_gita(api_key: str = None, scrape: bool = True,
              use_embedded_fallback: bool = True) -> list[dict]:
    """
    Load Bhagavad Gītā using best available method.

    Priority:
      1. bhagavadgita.io API (if api_key given)
      2. sacred-texts.com scrape (if scrape=True)
      3. Embedded 12 core verses (always available offline)
    """
    print("[GitaLoader] Starting Bhagavad Gītā ingestion...")
    verses = []

    if api_key:
        print("[GitaLoader] Trying bhagavadgita.io API...")
        verses = load_from_bhagavad_gita_io(api_key)

    if not verses and scrape:
        print("[GitaLoader] Trying sacred-texts.com scrape...")
        verses = scrape_sacred_texts_gita()

    if not verses and use_embedded_fallback:
        print("[GitaLoader] Using embedded 12 core verse fallback...")
        verses = GITA_CORE_VERSES

    if verses:
        out_path = OUT_DIR / "bhagavad_gita.json"
        out_path.write_text(
            json.dumps(verses, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[GitaLoader] ✓ {len(verses)} verses saved → {out_path}")
    else:
        print("[GitaLoader] ✗ No verses loaded")

    return verses


if __name__ == "__main__":
    import os
    api_key = os.getenv("BHAGAVAD_GITA_API_KEY")
    load_gita(api_key=api_key, scrape=True, use_embedded_fallback=True)
