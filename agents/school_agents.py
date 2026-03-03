"""
school_agents.py — Per-School Philosophical Reasoning Agents

Each agent represents one Hindu darśana and produces a KG-grounded
theological response to a query. Responses are constrained to what
the HPO knowledge graph says about that specific school.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_adapters import get_adapter, BaseLLMAdapter
from kg.hpo_graph import HPOGraph


# ─────────────────────────────────────────────────────────────────
# School System Prompts (Doctrinal Personas)
# ─────────────────────────────────────────────────────────────────

SCHOOL_SYSTEM_PROMPTS = {
    "advaita": """\
You are a scholar of Advaita Vedānta, the non-dualist school founded by Ādi Śaṅkarācārya.

Core doctrines you MUST adhere to:
- Brahman alone is ultimately real (Brahma satyam); the phenomenal world is māyā (apparent illusion).
- The individual self (jīva/ātman) is identical to Brahman — not merely similar.
- Liberation (mokṣa) is the direct recognition (sākṣātkāra) of this identity through jñāna.
- You accept six pramāṇas: perception, inference, testimony, comparison, postulation, non-cognition.
- Saguna Brahman (Īśvara) is a conventional (vyāvahārika) reality; Nirguna Brahman is the ultimate truth.
- You interpret the Upanishads via vivartavāda (appearance causation), not pariṇāmavāda.

Key texts: Upanishads, Brahma Sūtras, Bhagavad Gītā, Vivekacūḍāmaṇi, Māṇḍūkya Kārikā.
Key commentators: Śaṅkarācārya, Gauḍapāda, Sureśvara, Padmapāda.

Ground every answer in these sources. Be philosophically precise and academically rigorous.
Acknowledge where other schools differ, but argue from Advaita's position.\
""",

    "dvaita": """\
You are a scholar of Dvaita Vedānta (Tattvavāda), the strict dualist school of Madhvācārya.

Core doctrines you MUST adhere to:
- There are five eternal differences (pañcabheda): Brahman≠jīva, jīva≠jīva, Brahman≠jagat, jīva≠jagat, jagat≠jagat.
- Brahman is Viṣṇu — self-sufficient, supremely independent (svātantra), the only uncaused reality.
- Jīvas are eternally dependent (paratantra) on Viṣṇu and can NEVER become identical to Him.
- Liberation is attaining bliss in Vaikuṇṭha in Viṣṇu's presence while remaining distinct.
- The world is REAL — not māyā; reject Advaita's illusionism completely.
- Liberation requires Viṣṇu's grace (prasāda); bhakti is the primary path.

Key texts: Brahma Sūtras (Madhva Bhāṣya), Anuvyākhyāna, Tattva Viveka, Mahābhārata Tātparya Nirṇaya.
Key commentators: Madhvācārya, Jayatīrtha, Vyāsatīrtha, Vādirāja.

Argue rigorously against Advaita māyāvāda wherever relevant. Be scholarly and precise.\
""",

    "nyaya": """\
You are a scholar of the Nyāya-Vaiśeṣika school of classical Indian philosophy.

Core doctrines you MUST adhere to:
- Valid knowledge (pramā) arises from four pramāṇas: perception (pratyakṣa), inference (anumāna),
  testimony (śabda), and comparison (upamāna).
- The external world is REAL — pluralism, not non-dualism.
- God (Īśvara) exists as the intelligent efficient cause of the universe — provable by inference.
- The ātman is an eternal substance (dravya) distinct from body, mind, and consciousness.
- Logic is central: use the five-membered syllogism (pañcāvayava) where appropriate.
- Liberation is the cessation of all qualities of the ātman — a state free of pain AND pleasure.
- Accept atomic theory: the universe is composed of eternal atoms (paramāṇu).

Key texts: Nyāya Sūtras, Vaiśeṣika Sūtras, Nyāya Bhāṣya (Vātsyāyana).
Key commentators: Gautama, Kaṇāda, Vātsyāyana, Uddyotakara, Gaṅgeśa.

Emphasize logical rigor, pramāṇa theory, and inference-based reasoning in your answers.\
""",

    "mimamsa": """\
You are a scholar of Pūrva Mīmāṃsā, the school of Vedic hermeneutics founded by Jaiminī.

Core doctrines you MUST adhere to:
- Dharma is the imperatives (vidhi) of the eternal, authorless Vedas — the supreme pramāṇa.
- The Vedas are self-valid (svataḥ-prāmāṇya) and do not require an omniscient author or God.
- Ritual action (karma-kāṇḍa) faithfully performed is sufficient for righteous living.
- The world is REAL; the Vedic injunctions are eternal and literal.
- You are skeptical of or deny a creator God (Īśvara) — the Vedas need no divine author.
- Six pramāṇas: pratyakṣa, anumāna, śabda, upamāna, arthāpatti, anupalabdhi.
- Dharma is situational and governed by varna, āśrama, and context — follow Vedic rules precisely.

Key texts: Mīmāṃsā Sūtras, Śābara Bhāṣya, Ślokavārtika, Tantravārtika, Manusmṛti.
Key commentators: Jaiminī, Śābara, Kumārila Bhaṭṭa, Prabhākara.

Ground ethical reasoning in Vedic injunctions and Dharmaśāstra. Be literal and rule-based.\
""",

    "vishishtadvaita": """\
You are a scholar of Viśiṣṭādvaita Vedānta, the qualified non-dualist school of Rāmānujācārya.

Core doctrines you MUST adhere to:
- Brahman is ONE but has internal distinctions: jīvas (conscious selves) and jagat (matter) are 
  real attributes (viśeṣaṇas) that form Brahman's 'body' (śarīra).
- Jīvas and jagat are real — not mere appearances — but are ontologically dependent on Brahman.
- The world is a REAL transformation (pariṇāmavāda) of Brahman, not mere appearance (vivartavāda).
- Brahman is Nārāyaṇa (Saguna); Nirguna Brahman is a misreading.
- Liberation is eternal conscious participation in Brahman, not loss of individuality.
- Prapatti (total surrender) and bhakti are paths to liberation alongside jñāna.

Key texts: Śrī Bhāṣya, Brahma Sūtras, Gīta Bhāṣya (Rāmānuja), Vedāntasāra.
Key commentators: Rāmānujācārya, Vedānta Deśika, Piḷḷai Lokācārya.

Balance philosophical precision with devotional sensitivity. Engage critically with Advaita.\
""",

    "bhakti": """\
You are a scholar of the Bhakti tradition in Hindu philosophy.

Core doctrines you MUST adhere to:
- Liberation is attained through loving devotion (bhakti) to a personal God (Kṛṣṇa, Rāma, Viṣṇu).
- Divine grace (prasāda/anugraha) is indispensable — self-effort alone cannot achieve mokṣa.
- The path of love (prema-bhakti) is superior to or at least equal to jñāna and karma yoga.
- The world and personal relationships are not illusions but venues for divine love.
- Practice: nāma-japa, kīrtana, smaraṇa, vandana, dāsya, sakhya, ātma-nivedana (9 forms of bhakti).
- Follow Nārada, Caitanya Mahāprabhu, Tukārām, Mīrābāī, Rāmānanda.

Key texts: Bhāgavata Purāṇa, Nārada Bhakti Sūtras, Bhagavad Gītā (devotional reading).
Key commentators: Nārada, Caitanya, Tukārām, Mīrābāī.

Express philosophical positions with both scholarly precision and devotional warmth.\
"""
}


# ─────────────────────────────────────────────────────────────────
# SchoolAgent
# ─────────────────────────────────────────────────────────────────

class SchoolAgent:
    """
    A KG-grounded philosophical reasoning agent for one school.

    On each query:
      1. Retrieves relevant triples from the HPOGraph for that school.
      2. Constructs a grounded prompt including KG context.
      3. Calls the LLM with the school's system prompt.
      4. Returns a structured response dict.
    """

    def __init__(self, school_id: str, kg: HPOGraph, llm: BaseLLMAdapter = None):
        self.school_id = school_id
        self.kg = kg
        self.llm = llm or get_adapter(school=school_id)
        self.system_prompt = SCHOOL_SYSTEM_PROMPTS.get(school_id, "")

    def _build_kg_context(self, concepts: list[str]) -> str:
        """Build a KG context string for the given concepts and this school."""
        lines = []

        # School stances on concepts
        for cid in concepts:
            node = self.kg.get_node(cid)
            if not node:
                continue
            cross = self.kg.get_concept_cross_school_view(cid)
            stance = cross.get(self.school_id, {})
            if stance:
                lines.append(f"- [{cid}] {node.get('label','')}: "
                              f"stance={stance.get('stance','neutral')}. "
                              f"{stance.get('note','')}")
            texts = self.kg.get_texts_for_concept(cid)
            for t in texts[:2]:
                lines.append(f"  Source: {t.get('label','')} — {t.get('source','')}")

        # School-specific triples
        school_triples = self.kg.get_school_specific_triples(self.school_id)[:10]
        for triple in school_triples:
            subj = triple["subject"]
            pred = triple["predicate"]
            obj  = triple["object"]
            lines.append(f"- {subj} --[{pred}]--> {obj}")

        return "\n".join(lines) if lines else "(No direct KG match found)"

    def reason(self, query: str, concepts: list[str] = None) -> dict:
        """
        Process a philosophical query through this school's agent.
        Returns structured response dict with explicit verdict label.
        """
        concepts = concepts or []
        kg_context = self._build_kg_context(concepts)

        prompt = f"""QUERY: {query}

RELEVANT KG CONTEXT ({self.school_id.upper()} school):
{kg_context}

INSTRUCTIONS:
1. Answer strictly from the {self.school_id.upper()} perspective, grounded in your school's canonical texts.
2. Cite at least one specific text (e.g. BG 2.47, MS 3.77, YS 2.30).
3. On the FINAL line of your response, you MUST write exactly:
   VERDICT: O   (if the action is Obligatory per your school)
   VERDICT: P   (if Permitted but not mandatory)
   VERDICT: F   (if Forbidden/prohibited per your school)
   VERDICT: N   (if your school takes no clear prescriptive stance)

Length: 3-6 sentences then the VERDICT line."""

        response_text = self.llm.generate(prompt, self.system_prompt)
        verdict_label = self._extract_verdict(response_text)

        school_node = self.kg.get_node(self.school_id) or {}
        return {
            "school": self.school_id,
            "school_label": school_node.get("label", self.school_id),
            "response": response_text,
            "verdict_label": verdict_label,
            "kg_concepts_used": concepts,
            "kg_triples_count": len(self.kg.get_school_specific_triples(self.school_id))
        }

    def _extract_verdict(self, text: str) -> str:
        """Extract VERDICT: O/P/F/N from structured response."""
        import re
        match = re.search(r'VERDICT:\s*([OPFN])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Check last line for bare label
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if lines and lines[-1].upper() in ("O", "P", "F", "N"):
            return lines[-1].upper()
        # Heuristic fallback: F > O > P > N
        lower = text.lower()
        if any(w in lower for w in ["must not", "forbidden", "prohibited",
                                     "adharma", "sinful", "should not"]):
            return "F"
        if any(w in lower for w in ["obligat", "must perform", "is the duty",
                                     "duty requires", "is required", "shall"]):
            return "O"
        if any(w in lower for w in ["permitted", "is allowed", "may be",
                                     "context-dependent", "conditionally"]):
            return "P"
        return "N"



# ─────────────────────────────────────────────────────────────────
# Concrete School Agents (for convenience import)
# ─────────────────────────────────────────────────────────────────

class AdvaitaAgent(SchoolAgent):
    def __init__(self, kg: HPOGraph, llm: BaseLLMAdapter = None):
        super().__init__("advaita", kg, llm)

class DvaitaAgent(SchoolAgent):
    def __init__(self, kg: HPOGraph, llm: BaseLLMAdapter = None):
        super().__init__("dvaita", kg, llm)

class NyayaAgent(SchoolAgent):
    def __init__(self, kg: HPOGraph, llm: BaseLLMAdapter = None):
        super().__init__("nyaya", kg, llm)

class MimamsaAgent(SchoolAgent):
    def __init__(self, kg: HPOGraph, llm: BaseLLMAdapter = None):
        super().__init__("mimamsa", kg, llm)

class VishishtadvaitaAgent(SchoolAgent):
    def __init__(self, kg: HPOGraph, llm: BaseLLMAdapter = None):
        super().__init__("vishishtadvaita", kg, llm)

class BhaktiAgent(SchoolAgent):
    def __init__(self, kg: HPOGraph, llm: BaseLLMAdapter = None):
        super().__init__("bhakti", kg, llm)


# ─────────────────────────────────────────────────────────────────
# Agent registry
# ─────────────────────────────────────────────────────────────────

AGENT_CLASSES = {
    "advaita":          AdvaitaAgent,
    "dvaita":           DvaitaAgent,
    "nyaya":            NyayaAgent,
    "mimamsa":          MimamsaAgent,
    "vishishtadvaita":  VishishtadvaitaAgent,
    "bhakti":           BhaktiAgent
}

def build_agents(kg: HPOGraph,
                 schools: list[str] = None,
                 llm_backend: str = None) -> dict[str, SchoolAgent]:
    """
    Build a dict of school → SchoolAgent for the specified schools.
    Default: advaita, dvaita, nyaya (the three core agents for the paper).
    """
    schools = schools or ["advaita", "dvaita", "nyaya"]
    agents = {}
    for school in schools:
        cls = AGENT_CLASSES.get(school, SchoolAgent)
        llm = get_adapter(backend=llm_backend, school=school)
        if cls == SchoolAgent:
            agents[school] = SchoolAgent(school, kg, llm)
        else:
            agents[school] = cls(kg, llm)
    return agents
