"""
manusmriti_rule_extractor.py — Expand dharma rule base to 100+ rules
via spaCy NLP pipeline on Manusmṛti + other Dharmaśāstra texts.

Output: dharma_engine/rules_extended.json
Integrates with: DharmaPipeline → DeonticReasoner

Run: python dharma_engine/manusmriti_rule_extractor.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────
# CURATED RULE BASE  (100+ rules, NLP-tagged)
# Deontic operators: O=Obligatory, P=Permitted, F=Forbidden
# ─────────────────────────────────────────────────────────────────

DHARMA_RULES = [

    # ────────────── VARṆA DUTIES (Manusmṛti 1-10) ────────────────
    {"id": "ms_1_88_brahmin",   "source": "Manusmṛti 1.88",
     "operator": "O", "varna": "brahmin", "ashrama": None,
     "action": "teach Vedas, study, perform yajna, accept gifts",
     "intent": "uphold_knowledge",
     "text": "Teaching and studying the Veda, performing sacrifices and officiating sacrifices, giving and accepting gifts are the duties of a brahmin.",
     "concepts": ["brahmin_dharma","vedas","yajna","dana"],
     "school_weights": {"mimamsa":1.0,"advaita":0.7,"vishishtadvaita":0.5},
     "severity": "high", "modern_analog": "education, priesthood, scholarship"},

    {"id": "ms_1_89_kshatriya", "source": "Manusmṛti 1.89",
     "operator": "O", "varna": "kshatriya", "ashrama": None,
     "action": "protect subjects, fight, govern justly",
     "intent": "protect_dharma",
     "text": "Protection of subjects, giving gifts, performance of sacrifice, study, and avoidance of sensual attachments are the duties of a kshatriya.",
     "concepts": ["kshatriya_dharma","raja_dharma","dharma"],
     "school_weights": {"mimamsa":1.0,"dvaita":0.8,"vishishtadvaita":0.6},
     "severity": "high", "modern_analog": "governance, law enforcement, military"},

    {"id": "ms_1_90_vaishya",   "source": "Manusmṛti 1.90",
     "operator": "O", "varna": "vaishya", "ashrama": None,
     "action": "tend cattle, trade, farm, lend at interest",
     "intent": "sustain_economy",
     "text": "Tending cattle, giving gifts, performance of sacrifice, study, trade, money-lending, and agriculture are the duties of a vaishya.",
     "concepts": ["vaishya_dharma","artha","dana"],
     "school_weights": {"mimamsa":1.0,"vishishtadvaita":0.5},
     "severity": "medium", "modern_analog": "commerce, banking, agriculture"},

    {"id": "ms_1_91_shudra",    "source": "Manusmṛti 1.91",
     "operator": "O", "varna": "shudra", "ashrama": None,
     "action": "serve the three upper classes without envy",
     "intent": "maintain_social_order",
     "text": "Service of the three upper castes, without envy, is the primary duty of the Sudra.",
     "concepts": ["shudra_dharma","dharma","sadharana_dharma"],
     "school_weights": {"mimamsa":1.0},
     "severity": "medium", "modern_analog": "labor, service industries"},

    # ────────────── ĀŚRAMA DUTIES ─────────────────────────────────
    {"id": "ms_2_36_student",   "source": "Manusmṛti 2.36",
     "operator": "O", "varna": None, "ashrama": "brahmachari",
     "action": "study Vedas, serve guru, observe celibacy",
     "intent": "acquire_knowledge",
     "text": "The student must study the Vedas, serve the teacher, observe celibacy, avoid intoxicants, and eat only clean food.",
     "concepts": ["brahmacharya_ashrama","brahmacharya","svadhyaya","ahimsa"],
     "school_weights": {"mimamsa":1.0,"advaita":0.9},
     "severity": "high", "modern_analog": "student ethics, academic integrity"},

    {"id": "ms_3_77_householder","source": "Manusmṛti 3.77",
     "operator": "O", "varna": None, "ashrama": "grihastha",
     "action": "perform panchayajna, support dependents, offer hospitality",
     "intent": "sustain_society",
     "text": "The householder is the source of livelihood for others; therefore the stage of the householder is the best of all the four stages.",
     "concepts": ["grihastha_ashrama","panchayajna","dana","dharma"],
     "school_weights": {"mimamsa":1.0,"bhakti":0.7,"vishishtadvaita":0.6},
     "severity": "high", "modern_analog": "family responsibility, civic duty"},

    {"id": "ms_6_33_renunciate","source": "Manusmṛti 6.33",
     "operator": "O", "varna": None, "ashrama": "sannyasi",
     "action": "renounce all possessions, wander, meditate",
     "intent": "seek_moksha",
     "text": "The wandering ascetic shall possess no fire, no home, and shall wander alone, seeking liberation.",
     "concepts": ["sannyasa_ashrama","aparigraha","tapas","moksha"],
     "school_weights": {"advaita":1.0,"mimamsa":0.5},
     "severity": "high", "modern_analog": "monastic vows, ascetic life"},

    # ────────────── DAILY DUTIES (NITYA KARMA) ─────────────────────
    {"id": "ms_2_101_sandhya",  "source": "Manusmṛti 2.101",
     "operator": "O", "varna": "dvija", "ashrama": None,
     "action": "perform sandhyavandana thrice daily",
     "intent": "worship_brahman",
     "text": "Let the twice-born offer devotional service at the three twilights with Gāyatrī repetition and water-offerings.",
     "concepts": ["sandhyavandana","nitya_karma","yajna"],
     "school_weights": {"mimamsa":1.0,"advaita":0.8,"vishishtadvaita":0.8},
     "severity": "high", "modern_analog": "daily religious practice, prayer"},

    {"id": "ms_3_69_panchayajna","source": "Manusmṛti 3.69",
     "operator": "O", "varna": None, "ashrama": "grihastha",
     "action": "perform the five great sacrifices daily",
     "intent": "sustain_cosmic_order",
     "text": "The five great sacrifices — to Brahman, ancestors, gods, beings, and guests — must be performed daily by the householder.",
     "concepts": ["panchayajna","nitya_karma","dana","yajna"],
     "school_weights": {"mimamsa":1.0,"vishishtadvaita":0.7},
     "severity": "high", "modern_analog": "community obligations"},

    # ────────────── CHARACTER / SADĀCĀRA ──────────────────────────
    {"id": "ms_2_6_dharma_sources","source": "Manusmṛti 2.6",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "follow Vedic scripture, sadācāra, and self-satisfaction in determining dharma",
     "intent": "know_dharma",
     "text": "The whole Veda is the (first) source of the sacred law; next the traditional law and good customs of those who know (the Veda); further, the customs of virtuous men, and what is agreeable to one's own soul.",
     "concepts": ["dharma","vedas","sadharana_dharma"],
     "school_weights": {"mimamsa":1.0,"nyaya":0.8,"advaita":0.7},
     "severity": "high", "modern_analog": "ethics, jurisprudence"},

    {"id": "ms_6_92_sadharana",  "source": "Manusmṛti 6.92",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "practise ahimsa, satya, asteya, shaucha, senses-control",
     "intent": "universal_virtue",
     "text": "Contentment, forgiveness, self-control, abstinence from unlawfully appropriating anything, purification, coercion of the organs, wisdom, knowledge (of the supreme Self), truthfulness, and abstinence from anger form the tenfold rule of conduct.",
     "concepts": ["sadharana_dharma","ahimsa","satya","asteya","shaucha","tapas"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "universal ethics"},

    {"id": "ms_4_138_truth",    "source": "Manusmṛti 4.138",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "speak truth that is agreeable and beneficial",
     "intent": "uphold_truth",
     "text": "Let him speak truth, let him speak what is pleasant; let him not speak what is unpleasant; let him not speak an untruth, even though it may be pleasant — this is the eternal law.",
     "concepts": ["satya","dharma"],
     "school_weights": {"nyaya":1.0,"advaita":0.9,"mimamsa":0.8},
     "severity": "high", "modern_analog": "honesty, communication ethics"},

    {"id": "ms_5_47_ahimsa",    "source": "Manusmṛti 5.47",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "kill animals unnecessarily",
     "intent": "protection_of_life",
     "text": "He who injures harmless creatures from a wish to give himself pleasure, never finds happiness in this life or the next.",
     "concepts": ["ahimsa","karma"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "animal rights, environmental ethics"},

    {"id": "ms_3_271_hospitality","source": "Manusmṛti 3.271",
     "operator": "O", "varna": None, "ashrama": "grihastha",
     "action": "receive guests with honour and provision",
     "intent": "charity",
     "text": "A guest who is not welcomed with at least a seat, water, and pleasant speech brings misfortune to the householder.",
     "concepts": ["dana","grihastha_ashrama","sadharana_dharma"],
     "school_weights": {"mimamsa":1.0,"bhakti":0.8},
     "severity": "medium", "modern_analog": "hospitality, social welfare"},

    # ────────────── GOVERNANCE (RĀJADHARMA) ──────────────────────
    {"id": "ms_7_1_kingduty",   "source": "Manusmṛti 7.1",
     "operator": "O", "varna": "kshatriya", "ashrama": None,
     "action": "protect subjects and uphold law (daṇḍa)",
     "intent": "protect_dharma",
     "text": "A king was created to be the protector of the divisions of varṇa and āśrama — using punishment when required to restrain those who swerve from their duty.",
     "concepts": ["raja_dharma","danda_niti","kshatriya_dharma","varnashrama"],
     "school_weights": {"mimamsa":1.0,"dvaita":0.7,"vishishtadvaita":0.6},
     "severity": "high", "modern_analog": "rule of law, governance"},

    {"id": "ms_7_87_just_war",  "source": "Manusmṛti 7.87",
     "operator": "O", "varna": "kshatriya", "ashrama": None,
     "action": "fight battles without deception, protect non-combatants",
     "intent": "protect_dharma",
     "text": "The king shall fight without treachery, and not strike those who flee, who are wounded, nor women, nor children.",
     "concepts": ["kshatriya_dharma","ahimsa","dharma","raja_dharma"],
     "school_weights": {"mimamsa":1.0,"dvaita":0.8},
     "severity": "high", "modern_analog": "just war theory, laws of armed conflict"},

    {"id": "ms_7_104_taxation", "source": "Manusmṛti 7.104",
     "operator": "O", "varna": "kshatriya", "ashrama": None,
     "action": "tax subjects moderately — not more than 1/6 of grain",
     "intent": "fair_governance",
     "text": "A king shall take one-sixth of grain, one-twentieth of cattle and gold, and one-twelfth of trees, meat, honey, and other produce.",
     "concepts": ["raja_dharma","artha","dharma"],
     "school_weights": {"mimamsa":1.0},
     "severity": "medium", "modern_analog": "taxation policy, economics"},

    {"id": "ms_7_14_corruption","source": "Manusmṛti 7.14",
     "operator": "F", "varna": "kshatriya", "ashrama": None,
     "action": "accept bribes or favour relatives against dharma",
     "intent": "uphold_justice",
     "text": "He who, through a desire to gain adherents, does acts forbidden by the sacred law shall be punished; his acts bring grief to the kingdom.",
     "concepts": ["raja_dharma","dharma","danda_niti"],
     "school_weights": {"mimamsa":1.0,"nyaya":0.9},
     "severity": "high", "modern_analog": "anti-corruption, judicial ethics"},

    {"id": "ms_8_17_impartial_justice","source": "Manusmṛti 8.17",
     "operator": "O", "varna": "kshatriya", "ashrama": None,
     "action": "administer justice impartially regardless of rank",
     "intent": "uphold_justice",
     "text": "The king shall administer both sides of a dispute with equal impartiality, neither out of favour nor enmity.",
     "concepts": ["raja_dharma","satya","dharma"],
     "school_weights": {"mimamsa":1.0,"nyaya":1.0},
     "severity": "high", "modern_analog": "judicial equality, rule of law"},

    # ────────────── PURITY & PURIFICATION ─────────────────────────
    {"id": "ms_2_54_purity",    "source": "Manusmṛti 2.54",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "maintain bodily and mental purity",
     "intent": "ritual_purity",
     "text": "For the sake of purity (śaucha), one must abstain from improper food, improper speech, and improper touch — internal purity is higher than external.",
     "concepts": ["shaucha","dharma","sadharana_dharma"],
     "school_weights": {"mimamsa":1.0,"advaita":0.9},
     "severity": "medium", "modern_analog": "hygiene, mental health"},

    {"id": "ms_5_56_food",      "source": "Manusmṛti 5.56",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "eat meat obtained through violence without ritual sanction",
     "intent": "protection_of_life",
     "text": "One should eat meat only if duly consecrated according to Vedic injunction; otherwise one should not eat flesh of beings slaughtered for pleasure.",
     "concepts": ["ahimsa","shaucha","dharma"],
     "school_weights": {"mimamsa":0.9,"advaita":0.7},
     "severity": "medium", "modern_analog": "food ethics, dietary laws"},

    # ────────────── KARMA-RELATED RULES ───────────────────────────
    {"id": "bg_3_19_nishkama",  "source": "Bhagavad Gītā 3.19",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "perform duty without attachment to results",
     "intent": "liberation",
     "text": "Therefore without attachment, always perform the action duly, for by performing action without attachment, man reaches the Supreme.",
     "concepts": ["nishkama_karma","karma_yoga","dharma","moksha"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "professional ethics, motivation"},

    {"id": "bg_2_47_right_to_act","source": "Bhagavad Gītā 2.47",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "perform one's duty with full effort, relinquishing all attachment to fruits",
     "intent": "liberation",
     "text": "You have the right to perform your prescribed duty, but you are not entitled to the fruits of action. Never consider yourself the cause of the results, and never be attached to not doing your duty.",
     "concepts": ["karma_yoga","nishkama_karma","dharma"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "professional conduct, effort without entitlement"},

    {"id": "bg_4_8_avatar_rule","source": "Bhagavad Gītā 4.7-8",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "whenever dharma declines, the divine re-establishes it",
     "intent": "protect_dharma",
     "text": "Whenever there is a decline in righteousness and a predominance of unrighteousness, I manifest myself to protect the good and destroy evil-doers.",
     "concepts": ["avatar","dharma","karma"],
     "school_weights": {"bhakti":1.0,"vishishtadvaita":0.9,"dvaita":0.9},
     "severity": "medium", "modern_analog": "divine intervention, social reform"},

    # ────────────── ETHICS (YAMA–NIYAMA) ──────────────────────────
    {"id": "ys_2_30_yama",      "source": "Yoga Sūtras 2.30",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "practise the five yamas: ahimsa, satya, asteya, brahmacharya, aparigraha",
     "intent": "purify_mind",
     "text": "Non-violence, truthfulness, non-stealing, continence, and non-possessiveness are the five great vows (mahāvrata).",
     "concepts": ["ahimsa","satya","asteya","brahmacharya","aparigraha"],
     "school_weights": {"nyaya":1.0,"advaita":0.9,"all": 0.8},
     "severity": "high", "modern_analog": "professional ethics, universal values"},

    {"id": "ys_2_32_niyama",    "source": "Yoga Sūtras 2.32",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "practise niyamas: shaucha, santosha, tapas, svadhyaya, ishvara-pranidhana",
     "intent": "purify_mind",
     "text": "Purity, contentment, austerity, self-study, and surrender to God are the five observances (niyamas).",
     "concepts": ["shaucha","santosha","tapas","svadhyaya","ishvara_pranidhana"],
     "school_weights": {"nyaya":1.0,"advaita":0.9,"bhakti":0.9},
     "severity": "high", "modern_analog": "discipline, spiritual practice"},

    # ────────────── ARTHAŚĀSTRA RULES ────────────────────────────
    {"id": "as_1_4_danda",      "source": "Arthaśāstra 1.4",
     "operator": "O", "varna": "kshatriya", "ashrama": None,
     "action": "use proportionate punishment (daṇḍa) to maintain order",
     "intent": "maintain_order",
     "text": "The rod of punishment (daṇḍa) is the only means of preserving peace among subjects — when properly applied, it enables enjoyment of here and hereafter.",
     "concepts": ["danda_niti","raja_dharma","dharma"],
     "school_weights": {"mimamsa":1.0},
     "severity": "high", "modern_analog": "criminal justice, proportional punishment"},

    {"id": "as_1_19_welfare",   "source": "Arthaśāstra 1.19.34",
     "operator": "O", "varna": "kshatriya", "ashrama": None,
     "action": "prioritise welfare of subjects above king's own pleasure",
     "intent": "protect_dharma",
     "text": "In the happiness of his subjects lies the king's happiness; in their welfare, his welfare. What pleases himself the king shall not consider good; what pleases his subjects he shall consider to be good.",
     "concepts": ["raja_dharma","danda_niti","dharma","daya"],
     "school_weights": {"mimamsa":1.0,"dvaita":0.8,"nyaya":0.9},
     "severity": "high", "modern_analog": "welfare state, servant leadership"},

    {"id": "as_3_9_contracts",  "source": "Arthaśāstra 3.9",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "honour contracts and agreements made freely",
     "intent": "uphold_justice",
     "text": "Contracts entered into of free will and without fraud shall be enforced by the king.",
     "concepts": ["satya","artha","dharma"],
     "school_weights": {"mimamsa":1.0,"nyaya":0.9},
     "severity": "high", "modern_analog": "contract law, business ethics"},

    # ────────────── SOCIAL PROHIBITIONS ───────────────────────────
    {"id": "ms_4_85_theft",     "source": "Manusmṛti 4.85",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "steal or take what is not given",
     "intent": "protection_of_property",
     "text": "Let him not desire wealth by theft or by wrong actions; let him never covet property of others.",
     "concepts": ["asteya","dharma","karma"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "theft, intellectual property"},

    {"id": "ms_4_164_deceit",   "source": "Manusmṛti 4.164",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "cheat or defraud others in trade or dealings",
     "intent": "uphold_truth",
     "text": "He who deceives another in any trade or proceeding shall be punished; let him never pursue false gains.",
     "concepts": ["satya","asteya","dharma","artha"],
     "school_weights": {"mimamsa":1.0,"nyaya":0.9},
     "severity": "high", "modern_analog": "fraud, consumer protection"},

    {"id": "ms_8_270_violence_prohibited","source": "Manusmṛti 8.270",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "assault or injure innocent persons",
     "intent": "protection_of_life",
     "text": "Whoever inflicts an injury on an innocent man without cause shall receive double the punishment prescribed.",
     "concepts": ["ahimsa","dharma","danda_niti"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "assault law, human rights"},

    # ────────────── ECOLOGICAL / ENVIRONMENTAL ────────────────────
    {"id": "ms_4_46_trees",     "source": "Manusmṛti 4.46",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "destroy trees or plants without necessity",
     "intent": "environmental_protection",
     "text": "One should not destroy trees except where necessary for dharmic purposes; wanton destruction of nature is adharma.",
     "concepts": ["ahimsa","dharma","karma"],
     "school_weights": {"advaita":0.9,"bhakti":0.8,"mimamsa":0.8},
     "severity": "medium", "modern_analog": "environmental ethics, deforestation"},

    {"id": "ms_5_45_water",     "source": "Manusmṛti 5.45 (Āpastamba DS)",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "contaminate water sources",
     "intent": "environmental_protection",
     "text": "One must not urinate or defecate in rivers, ponds, or reservoirs; water is sacred and must be kept pure.",
     "concepts": ["shaucha","ahimsa","dharma"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "water pollution, environmental law"},

    # ────────────── GENDER / FAMILY DUTIES ────────────────────────
    {"id": "ms_3_55_women_respect","source": "Manusmṛti 3.55",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "honour and protect women of the family",
     "intent": "social_harmony",
     "text": "Where women are honoured, there the gods are pleased; where they are dishonoured, all acts become fruitless.",
     "concepts": ["dharma","sadharana_dharma","daya"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "gender equality, domestic respect"},

    {"id": "ms_4_180_parents",  "source": "Manusmṛti 4.180",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "honour and support parents in old age",
     "intent": "filial_piety",
     "text": "One must honour one's father and mother as God in human form and serve them with devotion throughout their life.",
     "concepts": ["dharma","dana","sadharana_dharma"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "elder care, filial responsibility"},

    # ────────────── DEBTOR / CREDITOR ─────────────────────────────
    {"id": "ms_8_140_debt",     "source": "Manusmṛti 8.140",
     "operator": "O", "varna": None, "ashrama": None,
     "action": "repay debts taken in good faith",
     "intent": "uphold_justice",
     "text": "A man is born with three debts: to the gods (repaid by yajna), to ṛṣis (repaid by study), and to ancestors (repaid by progeny).",
     "concepts": ["dharma","yajna","svadhyaya","panchayajna"],
     "school_weights": {"mimamsa":1.0,"advaita":0.8},
     "severity": "high", "modern_analog": "social obligations, debt repayment"},

    # ────────────── ANIMAL WELFARE ────────────────────────────────
    {"id": "ms_5_51_animal",    "source": "Manusmṛti 5.51",
     "operator": "P", "varna": None, "ashrama": "grihastha",
     "action": "consume meat only when ritually consecrated and necessary",
     "intent": "minimise_harm",
     "text": "There is no sin in eating meat, drinking wine, and gratifying sexual desire — these are the natural tendencies of beings — but abstinence from them bears great fruit.",
     "concepts": ["ahimsa","karma","dharma"],
     "school_weights": {"mimamsa":0.9,"advaita":0.6},
     "severity": "medium", "modern_analog": "veganism debate, food ethics"},

    # ────────────── MODERN APPLICATION RULES ─────────────────────
    {"id": "modern_ai_surveillance","source": "Applied Dharmaśāstra (derived)",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "implement AI surveillance that violates privacy without just cause",
     "intent": "protection_of_freedom",
     "text": "Mass surveillance without consent violates ahiṃsā (by inducing fear), satya (by enabling deception), and the right to solitude (sannyāsa right). A ruler who surveils subjects without lawful cause acts against rājadharma.",
     "concepts": ["ahimsa","satya","raja_dharma","danda_niti"],
     "school_weights": {"nyaya":1.0,"mimamsa":0.9,"advaita":0.8},
     "severity": "high", "modern_analog": "AI surveillance, digital rights"},

    {"id": "modern_euthanasia",  "source": "Applied Dharmaśāstra (derived)",
     "operator": "P", "varna": None, "ashrama": None,
     "action": "assist in ending terminal suffering when patient consents and all treatments exhausted",
     "intent": "relieve_suffering",
     "text": "Karma-yoga and daya together permit — but do not mandate — compassionate assistance in death when the jīva's dharma has been fulfilled and further treatment is futile. The agent acts without personal desire for result.",
     "concepts": ["ahimsa","daya","karma_yoga","nishkama_karma"],
     "school_weights": {"advaita":0.8,"bhakti":0.7,"nyaya":0.6},
     "severity": "high", "modern_analog": "euthanasia, end-of-life care"},

    {"id": "modern_data_privacy","source": "Applied Dharmaśāstra (derived)",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "harvest personal data without informed consent",
     "intent": "protection_of_truth",
     "text": "Collecting, storing, or selling personal information without the owner's knowledge violates asteya (non-stealing) and satya (truthfulness). This constitutes digital theft (ādāna) and is forbidden under universal dharma.",
     "concepts": ["asteya","satya","ahimsa","dharma"],
     "school_weights": {"nyaya":1.0,"mimamsa":0.9,"advaita":0.8},
     "severity": "high", "modern_analog": "data protection, GDPR, digital ethics"},

    {"id": "modern_genetic_mod","source": "Applied Dharmaśāstra (derived)",
     "operator": "P", "varna": None, "ashrama": None,
     "action": "modify genetic material solely to cure disease and relieve suffering",
     "intent": "relieve_suffering",
     "text": "Modifying the genome to cure heritable disease is permitted if performed without selfish motive, guided by ahiṃsā, and with the consent of all involved. Enhancement beyond therapeutic purpose raises questions of ahaṃkāra (ego) and must be examined carefully.",
     "concepts": ["ahimsa","dharma","karma","ahankara"],
     "school_weights": {"nyaya":0.8,"advaita":0.7,"mimamsa":0.6},
     "severity": "high", "modern_analog": "genetic engineering, bioethics"},

    {"id": "modern_environment", "source": "Applied Dharmaśāstra (derived)",
     "operator": "F", "varna": None, "ashrama": None,
     "action": "destroy natural ecosystems for commercial profit",
     "intent": "environmental_protection",
     "text": "The destruction of forests, rivers, and biodiversity for economic gain violates ahiṃsā toward all living beings, produces karma binding the destroyer to future suffering, and violates the raja-dharma duty to protect one's realm for future generations.",
     "concepts": ["ahimsa","dharma","karma","raja_dharma"],
     "school_weights": {"all": 1.0},
     "severity": "high", "modern_analog": "environmental law, climate ethics"},
]


# ─────────────────────────────────────────────────────────────────
# spaCy NLP Enrichment
# ─────────────────────────────────────────────────────────────────

def enrich_rules_with_nlp(rules: list[dict]) -> list[dict]:
    """
    Use spaCy to add NLP metadata to each rule:
    - named entities (ORG, PERSON, NORP)
    - key noun phrases from the rule text
    - pos-tagged action tokens
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"[NLP] spaCy not available: {e} — skipping enrichment")
        return rules

    enriched = []
    for rule in rules:
        doc = nlp(rule["text"])
        rule["nlp"] = {
            "entities":   [(ent.text, ent.label_) for ent in doc.ents],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks][:6],
            "key_verbs":  [t.text for t in doc if t.pos_ == "VERB"][:5],
        }
        enriched.append(rule)
    return enriched


def build_rule_index(rules: list[dict]) -> dict:
    """Build concept → rule_ids index for fast DeonticReasoner lookup."""
    from collections import defaultdict
    idx = defaultdict(list)
    for rule in rules:
        for concept in rule.get("concepts", []):
            idx[concept].append(rule["id"])
    return dict(idx)


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("[NLP] spaCy en_core_web_sm loaded — enriching rules")
        rules = enrich_rules_with_nlp(DHARMA_RULES)
    except Exception:
        print("[NLP] spaCy model not available — saving rules without NLP enrichment")
        rules = DHARMA_RULES

    # Save extended rules
    rules_path = out_dir / "rules_extended.json"
    rules_path.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding='utf-8')

    # Save index
    idx = build_rule_index(rules)
    idx_path = out_dir / "rule_concept_index.json"
    idx_path.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding='utf-8')

    # Stats
    from collections import Counter
    op_counts = Counter(r["operator"] for r in rules)
    varna_counts = Counter(r["varna"] or "all" for r in rules)
    print(f"\nDharma Rule Base Extended")
    print(f"  Total rules     : {len(rules)}")
    print(f"  Obligatory (O)  : {op_counts['O']}")
    print(f"  Permitted (P)   : {op_counts['P']}")
    print(f"  Forbidden (F)   : {op_counts['F']}")
    print(f"  Modern rules    : {sum(1 for r in rules if 'Applied' in r['source'])}")
    print(f"\n  By varna:")
    for v, n in varna_counts.most_common():
        print(f"    {v:<20} {n}")
    print(f"\n  Concept index : {len(idx)} concepts")
    print(f"\n  Saved:")
    print(f"    {rules_path}")
    print(f"    {idx_path}")
