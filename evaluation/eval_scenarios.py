"""
eval_scenarios.py — 34 gold evaluation scenarios with ground truth.

4 groups:
  Group A — Direct Textual (10): exact verse exists
  Group B — Contextual Extension (10): related principle
  Group C — Modern Analog (10): no direct verse
  Group D — Ambiguity Stress Test (4): conflicting verses

Each scenario has:
  - id, group, question
  - expected_verdict_per_school: {school: verdict}  ('O'=obligatory, 'P'=permitted, 'F'=forbidden, 'N'=neutral, 'X'=not applicable)
  - supporting_verses: list of valid verse citations
  - conflict_expected: bool
  - conflict_schools: which schools disagree
  - prescriptive_strength: 'absolute' | 'strong' | 'moderate' | 'contextual'
  - valid_concepts: list of KG concept IDs that must appear in answer
  - hallucination_traps: phrases that would indicate hallucination
"""

SCENARIOS = [

    # ═══════════════════════════════════════════════════════════
    # GROUP A — DIRECT TEXTUAL (10 scenarios)
    # Exact verse exists in canon
    # ═══════════════════════════════════════════════════════════

    {
        "id": "A01",
        "group": "direct_textual",
        "question": "Is a Kshatriya obligated to fight in a just war according to the Bhagavad Gita?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "O",
            "nyaya": "O",
            "bhakti": "O"
        },
        "supporting_verses": ["BG 2.31", "BG 2.33", "BG 3.35", "MS 1.89", "MS 7.87"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "absolute",
        "valid_concepts": ["kshatriya_dharma", "dharma", "karma_yoga"],
        "hallucination_traps": ["BG 5.12", "MS 3.77", "nonexistent verse", "Arjuna refused"]
    },
    {
        "id": "A02",
        "group": "direct_textual",
        "question": "What does BG 2.47 say about one's right to action versus results?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "O",
            "nyaya": "O",
            "bhakti": "O"
        },
        "supporting_verses": ["BG 2.47", "BG 3.19"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "absolute",
        "valid_concepts": ["karma_yoga", "nishkama_karma", "dharma"],
        "hallucination_traps": ["BG 2.45", "BG 4.18", "results belong to the doer"]
    },
    {
        "id": "A03",
        "group": "direct_textual",
        "question": "What is the Advaita Vedanta teaching on the identity of Atman and Brahman?",
        "expected_verdict_per_school": {
            "advaita": "O",         # Identity affirmed
            "dvaita": "F",          # Explicitly rejected
            "vishishtadvaita": "P", # Part-whole, not full identity
            "mimamsa": "N",
            "nyaya": "N",
            "bhakti": "F"
        },
        "supporting_verses": ["BAU 1.4.10", "CU 6.8.7", "MU 2.2.5"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "dvaita"],
        "prescriptive_strength": "absolute",
        "valid_concepts": ["atman", "brahman", "avidya", "maya"],
        "hallucination_traps": ["Shankara disagreed", "atman is separate", "Buddhist teaching"]
    },
    {
        "id": "A04",
        "group": "direct_textual",
        "question": "According to Manusmriti, what are the daily obligatory duties of a householder?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "O",
            "nyaya": "P",
            "bhakti": "P"
        },
        "supporting_verses": ["MS 3.69", "MS 3.70", "MS 3.77", "MS 2.101"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["grihastha_ashrama", "panchayajna", "nitya_karma", "sandhyavandana"],
        "hallucination_traps": ["Manusmriti Chapter 5", "daily bath only", "MS 7.104"]
    },
    {
        "id": "A05",
        "group": "direct_textual",
        "question": "What are the five yamas prescribed in Patanjali's Yoga Sutras?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "P",
            "nyaya": "O",
            "bhakti": "O"
        },
        "supporting_verses": ["YS 2.30", "YS 2.31"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["ahimsa", "satya", "asteya", "brahmacharya", "aparigraha"],
        "hallucination_traps": ["six yamas", "YS 3.1", "pranayama is first", "Manusmriti 6.92"]
    },
    {
        "id": "A06",
        "group": "direct_textual",
        "question": "What is Rama's teaching on moksha in Ramanuja's Vishishtadvaita?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "P",
            "vishishtadvaita": "O",
            "mimamsa": "N",
            "nyaya": "N",
            "bhakti": "O"
        },
        "supporting_verses": ["Ramanuja Sri Bhashya 3.3.53", "Bhagavata 3.29.13"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "vishishtadvaita"],
        "prescriptive_strength": "strong",
        "valid_concepts": ["sayujya", "bhakti_yoga", "prapatti", "moksha"],
        "hallucination_traps": ["Ramanuja rejected devotion", "same as advaita", "individual ceases to exist"]
    },
    {
        "id": "A07",
        "group": "direct_textual",
        "question": "What does Madhvacharya teach about the relationship between the individual soul and God?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "O",
            "vishishtadvaita": "P",
            "mimamsa": "N",
            "nyaya": "P",
            "bhakti": "O"
        },
        "supporting_verses": ["Madhva Tattvaviveka 1", "Madhva Anuvyakhyana"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "dvaita", "vishishtadvaita"],
        "prescriptive_strength": "absolute",
        "valid_concepts": ["vishesa", "jiva", "ishvara", "dvaita"],
        "hallucination_traps": ["Madhva agreed with Shankara", "no distinction", "souls merge with God"]
    },
    {
        "id": "A08",
        "group": "direct_textual",
        "question": "According to Mimamsa, what is the epistemic status of the Vedas?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "O",
            "nyaya": "P",
            "bhakti": "O"
        },
        "supporting_verses": ["Mimamsa Sutras 1.1.2", "Shabara Bhashya 1.1.2"],
        "conflict_expected": True,
        "conflict_schools": ["mimamsa", "nyaya"],
        "prescriptive_strength": "absolute",
        "valid_concepts": ["vedas", "sabda", "svatah_pramanya", "paratah_pramanya"],
        "hallucination_traps": ["Vedas were written by sages", "Nyaya rejects Vedas", "Mimamsa admits fallibility"]
    },
    {
        "id": "A09",
        "group": "direct_textual",
        "question": "What does the Bhagavad Gita say about the three gunas and their effect on human action?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "P",
            "nyaya": "O",
            "bhakti": "O"
        },
        "supporting_verses": ["BG 14.5", "BG 14.6", "BG 14.7", "BG 14.8", "BG 18.40"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["trigunas", "sattva", "rajas", "tamas", "prakriti"],
        "hallucination_traps": ["four gunas", "BG 7.13", "gunas are permanent states"]
    },
    {
        "id": "A10",
        "group": "direct_textual",
        "question": "How does the Nyaya school define valid inference (anumana) and its requirements?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "P",
            "vishishtadvaita": "P",
            "mimamsa": "P",
            "nyaya": "O",
            "bhakti": "N"
        },
        "supporting_verses": ["Nyaya Sutras 1.1.5", "Nyaya Sutras 1.1.32-38"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["anumana", "vyapti", "pratyaksha", "yukti"],
        "hallucination_traps": ["Nyaya has three members", "vyapti means definition", "Nyaya Sutras 2.1"]
    },

    # ═══════════════════════════════════════════════════════════
    # GROUP B — CONTEXTUAL EXTENSION (10 scenarios)
    # Text addresses related principle but not exact wording
    # ═══════════════════════════════════════════════════════════

    {
        "id": "B01",
        "group": "contextual_extension",
        "question": "Is lying to protect an innocent person from a murderer ethically justified according to Hindu texts?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "P",
            "vishishtadvaita": "P",
            "mimamsa": "F",     # Mimamsa takes satya as absolute
            "nyaya": "P",
            "bhakti": "P"
        },
        "supporting_verses": ["MS 4.138", "YS 2.30", "Mahabharata Vana Parva 297"],
        "conflict_expected": True,
        "conflict_schools": ["mimamsa", "advaita"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["satya", "ahimsa", "dharma"],
        "hallucination_traps": ["all schools allow lying freely", "lying is always forbidden", "BG 16.7"]
    },
    {
        "id": "B02",
        "group": "contextual_extension",
        "question": "Can a renunciate (sannyasi) accumulate wealth for charitable purposes?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "P",
            "vishishtadvaita": "P",
            "mimamsa": "N",
            "nyaya": "P",
            "bhakti": "P"
        },
        "supporting_verses": ["MS 6.33", "MS 6.38", "BG 12.13"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "dvaita"],
        "prescriptive_strength": "moderate",
        "valid_concepts": ["sannyasa_ashrama", "aparigraha", "dana", "dharma"],
        "hallucination_traps": ["all renunciates can own property", "Manusmriti allows this explicitly"]
    },
    {
        "id": "B03",
        "group": "contextual_extension",
        "question": "Is it dharmic for a king to execute a brahmin who commits murder?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "O",
            "nyaya": "O",
            "bhakti": "P"
        },
        "supporting_verses": ["MS 8.380", "MS 7.14", "Arthashastra 3.17", "MS 8.270"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "mimamsa"],
        "prescriptive_strength": "strong",
        "valid_concepts": ["raja_dharma", "danda_niti", "brahmin_dharma", "ahimsa"],
        "hallucination_traps": ["brahmins are always exempt", "MS 8.380 says no punishment", "king has no authority over brahmins"]
    },
    {
        "id": "B04",
        "group": "contextual_extension",
        "question": "Should a student prioritize worldly success (artha) over scriptural study during the brahmacharya ashrama?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "F",
            "bhakti": "F"
        },
        "supporting_verses": ["MS 2.36", "MS 2.69", "BG 4.34"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["brahmacharya_ashrama", "svadhyaya", "artha", "brahmacharya"],
        "hallucination_traps": ["artha is primary", "student may pursue both equally", "MS 2.1"]
    },
    {
        "id": "B05",
        "group": "contextual_extension",
        "question": "Is it ethically permissible to eat meat in Hindu dharmic thought?",
        "expected_verdict_per_school": {
            "advaita": "P",     # Contextual — tapas and sattvic diet preferred
            "dvaita": "P",
            "vishishtadvaita": "P",
            "mimamsa": "P",     # Permitted if ritually consecrated
            "nyaya": "P",
            "bhakti": "F"       # Strong preference for vegetarianism in Vaishnava bhakti
        },
        "supporting_verses": ["MS 5.47", "MS 5.51", "MS 5.56", "BG 17.8-9"],
        "conflict_expected": True,
        "conflict_schools": ["bhakti", "mimamsa"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["ahimsa", "dharma", "karma", "shaucha"],
        "hallucination_traps": ["all Hindus are vegetarian", "meat is always forbidden", "Vedas ban meat"]
    },
    {
        "id": "B06",
        "group": "contextual_extension",
        "question": "Can a woman take up sannyasa and pursue moksha independently?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "P",
            "vishishtadvaita": "P",
            "mimamsa": "F",
            "nyaya": "P",
            "bhakti": "O"
        },
        "supporting_verses": ["MS 5.147", "BG 9.32", "Narada Bhakti Sutras 73"],
        "conflict_expected": True,
        "conflict_schools": ["mimamsa", "advaita", "bhakti"],
        "prescriptive_strength": "moderate",
        "valid_concepts": ["sannyasa_ashrama", "moksha", "bhakti_yoga", "sadharana_dharma"],
        "hallucination_traps": ["all texts prohibit women's renunciation", "women need husband's permission for moksha"]
    },
    {
        "id": "B07",
        "group": "contextual_extension",
        "question": "Is anger (krodha) ever justified in dharmic action, such as a ruler punishing wrongdoers?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "P",
            "vishishtadvaita": "P",
            "mimamsa": "P",
            "nyaya": "P",
            "bhakti": "F"
        },
        "supporting_verses": ["BG 2.62-63", "MS 7.14", "Arthashastra 1.6", "Mahabharata Udyoga 33.60"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "mimamsa"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["krodha", "raja_dharma", "danda_niti", "dharma"],
        "hallucination_traps": ["anger is always permitted", "BG explicitly permits anger", "ruler must be angry"]
    },
    {
        "id": "B08",
        "group": "contextual_extension",
        "question": "What is the dharmic obligation of a soldier who is ordered to kill civilians?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "F",
            "bhakti": "F"
        },
        "supporting_verses": ["MS 7.87", "Arthashastra 10.3", "MS 7.90"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "absolute",
        "valid_concepts": ["kshatriya_dharma", "ahimsa", "dharma", "raja_dharma"],
        "hallucination_traps": ["following orders is dharma", "MS explicitly permits this", "non-combatants have no status"]
    },
    {
        "id": "B09",
        "group": "contextual_extension",
        "question": "Is it permissible to perform nishkama karma (desireless action) even in commercial contexts?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "P",
            "nyaya": "O",
            "bhakti": "O"
        },
        "supporting_verses": ["BG 2.47", "BG 3.19", "BG 18.9"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["nishkama_karma", "karma_yoga", "artha", "dharma"],
        "hallucination_traps": ["commercial work violates karma yoga", "Gita only for warriors"]
    },
    {
        "id": "B10",
        "group": "contextual_extension",
        "question": "Does the doctrine of karma mean that suffering victims deserve their fate and should not be helped?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "F",
            "bhakti": "F"
        },
        "supporting_verses": ["MS 4.180", "YS 2.30", "Narada Bhakti Sutras 61", "BG 16.1-3"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["daya", "ahimsa", "karma", "dharma", "sadharana_dharma"],
        "hallucination_traps": ["karma means do not help", "victims are responsible only", "Gita says let karma play out"]
    },

    # ═══════════════════════════════════════════════════════════
    # GROUP C — MODERN ANALOG (10 scenarios)
    # No direct verse — requires contextual synthesis
    # ═══════════════════════════════════════════════════════════

    {
        "id": "C01",
        "group": "modern_analog",
        "question": "Is mass AI surveillance of citizens justified for public safety according to Hindu dharmic principles?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "P",     # May be permitted if king has just cause
            "nyaya": "F",
            "bhakti": "F"
        },
        "supporting_verses": ["MS 7.14", "Arthashastra 1.19.34", "YS 2.30", "MS 4.138"],
        "conflict_expected": True,
        "conflict_schools": ["mimamsa", "advaita"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["ahimsa", "satya", "raja_dharma", "danda_niti"],
        "hallucination_traps": ["Kautilya approves all surveillance", "no relevant verse", "BG explicitly discusses AI"]
    },
    {
        "id": "C02",
        "group": "modern_analog",
        "question": "Is physician-assisted euthanasia (assisted dying) permissible under dharmic ethics?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "P",
            "nyaya": "P",
            "bhakti": "F"
        },
        "supporting_verses": ["YS 2.30", "MS 5.47", "BG 3.19", "Narada Bhakti Sutras 61"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "dvaita", "bhakti"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["ahimsa", "daya", "karma_yoga", "nishkama_karma"],
        "hallucination_traps": ["Vedas explicitly discuss euthanasia", "all schools permit it", "karma yoga means killing is fine"]
    },
    {
        "id": "C03",
        "group": "modern_analog",
        "question": "Is collecting personal data without informed consent a violation of dharma?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "F",
            "bhakti": "F"
        },
        "supporting_verses": ["YS 2.30", "MS 4.85", "MS 4.164", "MS 4.138"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["asteya", "satya", "ahimsa", "dharma"],
        "hallucination_traps": ["no relevant principle", "data is not property", "dharma has nothing to say about data"]
    },
    {
        "id": "C04",
        "group": "modern_analog",
        "question": "Is genetic modification of crops and animals to reduce starvation ethically justified?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "P",
            "vishishtadvaita": "P",
            "mimamsa": "P",
            "nyaya": "P",
            "bhakti": "P"
        },
        "supporting_verses": ["YS 2.30", "MS 5.47", "Arthashastra 2.24"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["ahimsa", "daya", "dharma", "karma"],
        "hallucination_traps": ["Vedas explicitly permit GMO", "Hindu texts discuss genetics", "forbidden in all texts"]
    },
    {
        "id": "C05",
        "group": "modern_analog",
        "question": "Is environmental destruction for economic development a violation of dharmic duty?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "F",
            "bhakti": "F"
        },
        "supporting_verses": ["MS 4.46", "MS 5.45", "YS 2.30", "Arthashastra 2.1"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["ahimsa", "dharma", "karma", "raja_dharma"],
        "hallucination_traps": ["no dharmic view on environment", "Arthashastra promotes deforestation"]
    },
    {
        "id": "C06",
        "group": "modern_analog",
        "question": "Is cryptocurrency speculation driven by greed (lobha) a form of adharma?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "P",     # Artha acquisition is permitted within limits
            "nyaya": "P",
            "bhakti": "F"
        },
        "supporting_verses": ["BG 3.37", "MS 4.164", "Arthashastra 3.9", "BG 16.21"],
        "conflict_expected": True,
        "conflict_schools": ["mimamsa", "advaita", "bhakti"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["lobha", "artha", "karma", "asteya", "satya"],
        "hallucination_traps": ["all financial activity is adharma", "Arthashastra bans investments", "Gita discusses cryptocurrency"]
    },
    {
        "id": "C07",
        "group": "modern_analog",
        "question": "Does a software engineer have a svadharma (personal duty) in the way Kshatriyas have battle duty?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "P",
            "nyaya": "O",
            "bhakti": "O"
        },
        "supporting_verses": ["BG 3.35", "BG 18.41-44", "MS 1.87-91"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "moderate",
        "valid_concepts": ["kshatriya_dharma", "dharma", "karma_yoga", "varnashrama"],
        "hallucination_traps": ["software engineers have no dharma", "Gita only applies to warriors", "modern professions are not in Vedas"]
    },
    {
        "id": "C08",
        "group": "modern_analog",
        "question": "Is it dharmic for a doctor to break patient confidentiality to prevent harm to a third party?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "F",     # Satya is more categorical; but danda niti permits
            "nyaya": "O",
            "bhakti": "P"
        },
        "supporting_verses": ["MS 4.138", "YS 2.30", "MS 7.14", "Caraka Samhita Sutrasthana 1"],
        "conflict_expected": True,
        "conflict_schools": ["mimamsa", "nyaya"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["satya", "ahimsa", "daya", "dharma"],
        "hallucination_traps": ["doctors have no dharmic role", "confidentiality is always overridden", "no relevant verse"]
    },
    {
        "id": "C09",
        "group": "modern_analog",
        "question": "Does the use of autonomous weapons in war violate dharmic principles of just war?",
        "expected_verdict_per_school": {
            "advaita": "F",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "F",
            "bhakti": "F"
        },
        "supporting_verses": ["MS 7.87", "MS 7.90", "Arthashastra 10.3"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "strong",
        "valid_concepts": ["kshatriya_dharma", "ahimsa", "dharma", "danda_niti"],
        "hallucination_traps": ["machines can perform dharmic war", "no rule about autonomous weapons"]
    },
    {
        "id": "C10",
        "group": "modern_analog",
        "question": "Is organ donation after death dharmic, or does it violate the body's sanctity?",
        "expected_verdict_per_school": {
            "advaita": "O",
            "dvaita": "O",
            "vishishtadvaita": "O",
            "mimamsa": "P",
            "nyaya": "O",
            "bhakti": "P"
        },
        "supporting_verses": ["MS 4.229", "YS 2.30", "BG 16.1-3"],
        "conflict_expected": False,
        "conflict_schools": [],
        "prescriptive_strength": "moderate",
        "valid_concepts": ["dana", "ahimsa", "daya", "dharma"],
        "hallucination_traps": ["body must be preserved intact always", "organ donation is forbidden in all texts"]
    },

    # ═══════════════════════════════════════════════════════════
    # GROUP D — AMBIGUITY STRESS TEST (4 scenarios)
    # Multiple conflicting verses — system must detect conflict
    # ═══════════════════════════════════════════════════════════

    {
        "id": "D01",
        "group": "ambiguity_stress",
        "question": "When non-violence (ahimsa) conflicts with the duty to protect innocents through force, which takes precedence?",
        "expected_verdict_per_school": {
            "advaita": "P",      # Ahimsa preferred but role-duty matters
            "dvaita": "O",       # Duty to protect clearly O
            "vishishtadvaita": "O",
            "mimamsa": "O",      # Vidhi (injunction) of protection overrides general principle
            "nyaya": "O",
            "bhakti": "P"
        },
        "supporting_verses": ["BG 2.31-33", "MS 7.87", "YS 2.30", "MS 5.47"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "mimamsa", "bhakti"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["ahimsa", "kshatriya_dharma", "dharma", "karma_yoga"],
        "hallucination_traps": ["both principles are identical", "conflict does not exist in Hinduism"]
    },
    {
        "id": "D02",
        "group": "ambiguity_stress",
        "question": "Is renunciation (sannyasa) possible or advisable while still having family responsibilities?",
        "expected_verdict_per_school": {
            "advaita": "P",
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "F",
            "bhakti": "P"
        },
        "supporting_verses": ["MS 6.33-35", "MS 3.77", "BG 3.4-5", "Narada Bhakti Sutras"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "mimamsa", "bhakti"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["sannyasa_ashrama", "grihastha_ashrama", "karma_yoga", "vairagya"],
        "hallucination_traps": ["all schools permit early renunciation", "family duties can always be abandoned"]
    },
    {
        "id": "D03",
        "group": "ambiguity_stress",
        "question": "Does the impermanence of worldly life (maya, samsara) mean humans should avoid relationships and attachments entirely?",
        "expected_verdict_per_school": {
            "advaita": "P",     # Viveka points to detachment but not avoidance
            "dvaita": "F",
            "vishishtadvaita": "F",
            "mimamsa": "F",
            "nyaya": "P",
            "bhakti": "F"       # Bhakti values loving relationships with God and devotees
        },
        "supporting_verses": ["BG 2.47", "BG 18.66", "MS 3.77", "Narada Bhakti Sutras 2"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "dvaita", "bhakti"],
        "prescriptive_strength": "contextual",
        "valid_concepts": ["maya", "samsara", "vairagya", "karma_yoga", "grihastha_ashrama"],
        "hallucination_traps": ["all schools say avoid relationships", "Gita teaches full withdrawal", "maya means relationships are evil"]
    },
    {
        "id": "D04",
        "group": "ambiguity_stress",
        "question": "Can a devotee of Vishnu worship Shiva equally, or does this violate their dharmic loyalty?",
        "expected_verdict_per_school": {
            "advaita": "O",     # All deities are Brahman
            "dvaita": "F",      # Vishnu alone is supreme; Shiva is a jiva
            "vishishtadvaita": "P",  # Shiva worships Vishnu; ambiguous
            "mimamsa": "P",
            "nyaya": "P",
            "bhakti": "F"       # Vaishnava bhakti: exclusive devotion preferred
        },
        "supporting_verses": ["BG 9.23-25", "BG 7.20-23", "Bhagavata 4.2.29"],
        "conflict_expected": True,
        "conflict_schools": ["advaita", "dvaita", "bhakti"],
        "prescriptive_strength": "moderate",
        "valid_concepts": ["bhakti_yoga", "ishvara", "saguna_brahman", "nirguna_brahman"],
        "hallucination_traps": ["all sects are identical", "Dvaita permits Shiva worship freely", "Gita says worship other gods is equivalent"]
    },
]

# ───────────────────────────────────────────────────────────────
# VALID VERSE CITATION REGISTRY
# Used to check hallucination — these are REAL citations
# ───────────────────────────────────────────────────────────────
VALID_CITATIONS = {
    # Bhagavad Gita
    "BG 2.31", "BG 2.33", "BG 2.47", "BG 3.4", "BG 3.5", "BG 3.9",
    "BG 3.19", "BG 3.35", "BG 4.7", "BG 4.8", "BG 4.34",
    "BG 7.13", "BG 7.20", "BG 7.21", "BG 7.22", "BG 7.23",
    "BG 9.7", "BG 9.23", "BG 9.24", "BG 9.25", "BG 9.32",
    "BG 12.13", "BG 14.5", "BG 14.6", "BG 14.7", "BG 14.8",
    "BG 16.1", "BG 16.21", "BG 17.3", "BG 17.8", "BG 17.9",
    "BG 18.9", "BG 18.40", "BG 18.41", "BG 18.66", "BG 18.73",
    # Manusmriti
    "MS 1.87", "MS 1.88", "MS 1.89", "MS 1.90", "MS 1.91",
    "MS 2.6", "MS 2.36", "MS 2.69", "MS 2.101",
    "MS 3.55", "MS 3.69", "MS 3.70", "MS 3.77", "MS 3.78",
    "MS 4.46", "MS 4.85", "MS 4.138", "MS 4.164", "MS 4.180", "MS 4.229",
    "MS 5.45", "MS 5.47", "MS 5.51", "MS 5.56",
    "MS 6.1", "MS 6.33", "MS 6.35", "MS 6.38", "MS 6.92",
    "MS 7.1", "MS 7.14", "MS 7.87", "MS 7.90", "MS 7.104",
    "MS 8.270", "MS 8.380",
    # Yoga Sutras
    "YS 1.1", "YS 1.2", "YS 1.12", "YS 1.41", "YS 1.50",
    "YS 2.29", "YS 2.30", "YS 2.31", "YS 2.32", "YS 3.3",
    # Upanishads
    "BAU 1.4.10", "BAU 4.4.5", "CU 6.8.7", "MU 2.2.5",
    "TU 2.1.1", "MandU 7",
    # Brahma Sutras
    "BS 1.1.1", "BS 2.1.9", "BS 2.1.14", "BS 3.3.53", "BS 4.1.15",
    # Arthashastra
    "AS 1.4", "AS 1.6", "AS 1.19.34", "AS 2.1", "AS 2.24",
    "AS 3.9", "AS 3.17", "AS 10.3",
    # Bhagavata Purana
    "BP 1.1.1", "BP 3.29.13", "BP 4.2.29", "BP 7.5.23",
    "BP 10.33", "BP 12.3.52",
    # Mimamsa Sutras
    "MimS 1.1.2", "MimS 4.1", "MimS 4.2", "MimS 4.3", "MimS 4.4",
    # Nyaya Sutras
    "NS 1.1.4", "NS 1.1.5", "NS 1.1.6", "NS 1.1.7",
    "NS 1.1.22", "NS 1.1.32",
    # Narada Bhakti Sutras
    "NBS 1.2", "NBS 2", "NBS 61", "NBS 73",
    # Vivekachudamani
    "VC 19", "VC 20", "VC 22", "VC 23", "VC 27", "VC 48", "VC 139", "VC 248",
    "VC 339", "VC 453", "VC 454",
}

# Groups and counts
SCENARIO_SUMMARY = {
    "total": len(SCENARIOS),
    "direct_textual": sum(1 for s in SCENARIOS if s["group"] == "direct_textual"),
    "contextual_extension": sum(1 for s in SCENARIOS if s["group"] == "contextual_extension"),
    "modern_analog": sum(1 for s in SCENARIOS if s["group"] == "modern_analog"),
    "ambiguity_stress": sum(1 for s in SCENARIOS if s["group"] == "ambiguity_stress"),
    "conflict_expected": sum(1 for s in SCENARIOS if s["conflict_expected"]),
}

if __name__ == "__main__":
    import json
    print("Evaluation Scenario Summary:")
    for k, v in SCENARIO_SUMMARY.items():
        print(f"  {k}: {v}")
    print(f"\nSample scenario:")
    print(json.dumps(SCENARIOS[0], indent=2))
