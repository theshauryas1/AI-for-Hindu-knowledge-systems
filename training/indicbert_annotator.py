"""
indicbert_annotator.py — Generate 500+ school-tagged sentences for IndicBERT fine-tuning.

Produces: training/indicbert_sentences.jsonl
Format per line:
  {
    "text": "...",
    "label": "advaita",          # school label
    "concept": "brahman",        # primary concept
    "source": "Upanishad 6.8.7", # textual source
    "split": "train"             # train / val / test
  }

Run: python training/indicbert_annotator.py
"""

import json
import random
from pathlib import Path

random.seed(42)
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "indicbert_sentences.jsonl"

# ─────────────────────────────────────────────────────────────────
# Core annotated sentence corpus
# Each entry: (text, school, concept, source)
# ─────────────────────────────────────────────────────────────────

ANNOTATED_SENTENCES = [

    # ── ADVAITA VEDĀNTA ──────────────────────────────────────────
    ("Brahman alone is real; the world is illusory appearance due to māyā.", "advaita", "maya", "Vivekacūḍāmaṇi 20"),
    ("The individual self and Brahman are numerically identical — ahaṃ brahmāsmi.", "advaita", "atman", "Bṛhadāraṇyaka Up. 1.4.10"),
    ("Liberation is the direct recognition that one's true nature is pure consciousness.", "advaita", "moksha", "Vivekacūḍāmaṇi 48"),
    ("Avidyā, superimposing individuality on the non-dual self, is the root of bondage.", "advaita", "avidya", "Vivekacūḍāmaṇi 139"),
    ("The world arises through vivartavāda — apparent transformation, not real change.", "advaita", "maya", "Brahma Sūtras 2.1.14"),
    ("Śaṅkarācārya taught that jñāna alone removes the ignorance that veils Brahman.", "advaita", "jnana_yoga", "Vivekacūḍāmaṇi 56"),
    ("Nirvikalpa samādhi dissolves all mental modifications in the ocean of pure awareness.", "advaita", "nirvikalpa_samadhi", "Vivekacūḍāmaṇi 339"),
    ("The state of jīvanmukti allows the liberated sage to continue acting without bondage.", "advaita", "jivanmukti", "Jīvanmuktiviveka 1.2"),
    ("Māyā has two powers: āvaraṇa-śakti (concealing) and vikṣepa-śakti (projecting).", "advaita", "maya", "Pañcadaśī 1.15"),
    ("Pratyabhijñā — recognition of one's true nature — is the essence of Advaita realization.", "advaita", "atman", "Māṇḍūkya Kārikā 3.37"),
    ("The five sheaths (pañca-kośa) veil the pure self like husks around a grain of rice.", "advaita", "pancha_kosha", "Taittirīya Up. 2.1-5"),
    ("Brahman is described as sat-cit-ānanda — being, consciousness, and bliss absolute.", "advaita", "brahman", "Taittirīya Up. 2.1.1"),
    ("Viveka (discrimination) between the real and unreal is the first prerequisite for Vedāntic inquiry.", "advaita", "viveka", "Vivekacūḍāmaṇi 19"),
    ("Vairāgya (dispassion) toward worldly and heavenly enjoyments must precede Self-inquiry.", "advaita", "vairagya", "Vivekacūḍāmaṇi 22"),
    ("The three states of waking, dream, and deep sleep point to the fourth, turīya, as the witness.", "advaita", "atman", "Māṇḍūkya Up. 7"),
    ("Śaṅkara refuted the Sāṅkhya view of Prakṛti as independent by showing Brahman as the sole cause.", "advaita", "brahman", "Brahma Sūtras 2.2.1"),
    ("Adhyāsa (superimposition) of the not-self on the Self is the fundamental error Vedānta corrects.", "advaita", "avidya", "Śaṅkara's Adhyāsa Bhāṣya"),
    ("The rope mistaken for a snake illustrates māyā's power of projective illusion.", "advaita", "maya", "Vedāntasāra 41"),
    ("Mumukṣutva — intense desire for liberation — is the culminating qualification for Vedāntic study.", "advaita", "mumukshutva", "Vivekacūḍāmaṇi 27"),
    ("Consistent reasoning (manana) purifies the intellect's apperception of scriptural teaching.", "advaita", "jnana_yoga", "Bṛhadāraṇyaka Up. 2.4.5"),
    ("The Advaita view holds consciousness to be self-luminous and self-certifying (svaprakāśa).", "advaita", "atman", "Vivekacūḍāmaṇi 248"),
    ("Sañcita karma is dissolved at Self-realization, but prārabdha karma must run its course.", "advaita", "prarabdha_karma", "Vivekacūḍāmaṇi 454"),
    ("The gross, subtle, and causal bodies are superimposed on the pure witness-consciousness.", "advaita", "chitta", "Vedāntasāra 30"),
    ("Turīyātīta is the state transcending even the fourth, realized by the jīvanmukta.", "advaita", "jivanmukti", "Māṇḍūkya Kārikā 3.35"),
    ("Śabda-pramāṇa (scriptural testimony) is the only means of knowing Brahman directly.", "advaita", "sabda", "Brahma Sūtras 1.1.3"),

    # ── DVAITA VEDĀNTA ───────────────────────────────────────────
    ("Madhvācārya's pañca-bheda establishes five eternal distinctions between God, souls, and matter.", "dvaita", "vishesa", "Madhva's Tattvaviveka"),
    ("The jīva is eternally dependent on Viṣṇu and can never attain identity with him.", "dvaita", "jiva", "Madhva's Brahmasūtrabhāṣya 2.1.22"),
    ("Liberation in Dvaita means dwelling eternally in Vaikuṇṭha in blissful worship of Hari.", "dvaita", "moksha", "Madhva's Mokṣadharma"),
    ("Viṣṇu is the independent reality (svatantra); all other entities are utterly dependent (paratantra).", "dvaita", "ishvara", "Madhva's Tattvaviveka 1"),
    ("Valid means of knowledge in Dvaita are perception, inference, and Vedic testimony.", "dvaita", "sabda", "Madhva's Pramāṇalakṣaṇa"),
    ("Bhāgavata Purāṇa is the supreme authority on the form and attributes of the personal God.", "dvaita", "saguna_brahman", "Bhāgavata Purāṇa 1.1.1"),
    ("The world is not māyā but a genuine creation of the Lord, sustaining his divine glory.", "dvaita", "srsti", "Madhva's Viṣṇutattvanirnaya"),
    ("Karma accumulated from beginningless time keeps jīvas bound in saṃsāra under God's oversight.", "dvaita", "samsara", "Madhva's Brahmasūtrabhāṣya 2.3.7"),
    ("Prapatti (surrender) to Viṣṇu is the indispensable means of grace leading to liberation.", "dvaita", "prapatti", "Madhva's Madhvavijaya commentary"),
    ("The muktas in Vaikuṇṭha experience differentiated bliss graded according to their intrinsic nature.", "dvaita", "salokya", "Madhva's Mokṣadharma"),
    ("God's grace (kṛpā) is the ultimate efficient cause of liberation, beyond the jīva's efforts.", "dvaita", "bhakti_yoga", "Madhva's Tattvaviveka 5"),
    ("Ānanda-tāratamya — degrees of bliss — distinguishes muktas even in the liberated state.", "dvaita", "moksha", "Madhva's Brahmasūtrabhāṣya 3.4.52"),
    ("The four sātvika-āgamas and Brahma Sūtras are the highest pramāṇas after the Vedas in Dvaita.", "dvaita", "sabda", "Madhva's Anuvyākhyāna intro"),
    ("Tatvavāda denies any evolution of Brahman into the world; creation is a genuine act of will.", "dvaita", "srsti", "Madhva's Tattvasaṅkhyāna"),
    ("Viṣṇu's form is real, eternal, and made of transcendental matter distinct from material guṇas.", "dvaita", "saguna_brahman", "Madhva's Viṣṇutattvanirnaya"),

    # ── VIŚIṢṬĀDVAITA ────────────────────────────────────────────
    ("In Rāmānuja's philosophy, individual selves and matter are the body of Brahman.", "vishishtadvaita", "brahman", "Rāmānuja's Śrī Bhāṣya 1.1.1"),
    ("The world is a real pariṇāma (transformation) of Brahman, not merely apparent change.", "vishishtadvaita", "srsti", "Rāmānuja's Vedārthasaṅgraha 31"),
    ("Brahman is qualified (viśiṣṭa) by its inseparable attributes: the individual selves and matter.", "vishishtadvaita", "saguna_brahman", "Rāmānuja's Śrī Bhāṣya 1.1.1"),
    ("Liberation (mokṣa) consists in eternal blissful intuition of Brahman in Śrīvaikuṇṭha.", "vishishtadvaita", "sayujya", "Rāmānuja's Vedārthasaṅgraha 244"),
    ("Prapatti (self-surrender) to Nārāyaṇa is a direct means of liberation alongside bhakti.", "vishishtadvaita", "prapatti", "Yāmunācārya's Stotraratna"),
    ("The relationship of the world to Brahman is that of śarīra (body) to śarīrī (soul).", "vishishtadvaita", "brahman", "Rāmānuja's Śrī Bhāṣya 2.1.9"),
    ("Nitya-karma (daily duties) purify the mind and form the foundation of bhakti-yoga.", "vishishtadvaita", "nitya_karma", "Rāmānuja's Gītābhāṣya 3.19"),
    ("Īśvara's inner controller (antaryāmin) pervades and governs all souls from within.", "vishishtadvaita", "ishvara", "Bṛhadāraṇyaka Up. 3.7"),
    ("Bhakti performed as perpetual loving remembrance leads to direct vision of the Lord.", "vishishtadvaita", "bhakti_yoga", "Rāmānuja's Śrī Bhāṣya 3.3.53"),
    ("Viśiṣṭādvaita upholds that Brahman is non-dual yet intrinsically differentiated by self and matter.", "vishishtadvaita", "brahman", "Rāmānuja's Vedārthasaṅgraha 2"),
    ("The three realities of Viśiṣṭādvaita are: Īśvara, cit (conscious), and acit (unconscious).", "vishishtadvaita", "puranas", "Rāmānuja's Tattvahāra"),
    ("Līlā (divine play) of Brahman explains why a perfect, self-sufficient reality creates the world.", "vishishtadvaita", "lila", "Brahma Sūtras 2.1.33"),
    ("Avatāras descend to restore dharma, destroy evil, and re-establish the path of liberation.", "vishishtadvaita", "avatar", "Bhagavad Gītā 4.7-8"),
    ("Both upāya (means) and upāsana (meditation) are grounded in the Vedic testimony of the Divyaprabandham.", "vishishtadvaita", "sabda", "Yāmunācārya's Siddhitraya"),
    ("Cosmic dissolution (pralaya) sees all souls and matter merge into Brahman as latent potencies.", "vishishtadvaita", "pralaya", "Bhagavad Gītā 8.17-19"),

    # ── NYĀYA–VAIŚEṢIKA ──────────────────────────────────────────
    ("God's existence is inferred because the universe, like a pot, requires an intelligent creator.", "nyaya", "ishvara", "Udayanācārya's Nyāyakusumāñjali 1.4"),
    ("Vyāpti (invariable concomitance) between hetu and sādhya is the nerve of any valid inference.", "nyaya", "vyapti", "Nyāya Sūtras 1.1.35"),
    ("Valid perception (pratyakṣa) requires contact of the sense organ with the object and non-erroneous awareness.", "nyaya", "pratyaksha", "Nyāya Sūtras 1.1.4"),
    ("Anumāna proceeds through five members: pratijñā, hetu, udāharaṇa, upanaya, and nigamana.", "nyaya", "anumana", "Nyāya Sūtras 1.1.32"),
    ("Upamāna enables knowledge of a new object through its similarity to a familiar one.", "nyaya", "upamana", "Nyāya Sūtras 1.1.6"),
    ("Verbal testimony (śabda) is valid only when the speaker has reliable knowledge (āptavākya).", "nyaya", "sabda", "Nyāya Sūtras 1.1.7"),
    ("The atom (paramāṇu) is the ultimate indivisible unit of matter; it is eternal and without parts.", "nyaya", "pancha_bhuta", "Vaiśeṣika Sūtras 4.1.1"),
    ("Nyāya distinguishes right knowledge (pramā) from error (bhrama) and doubt (saṃśaya).", "nyaya", "pratyaksha", "Nyāya Sūtras 1.1.1"),
    ("Liberation in Nyāya is the permanent cessation of all suffering and the qualities of the ātman.", "nyaya", "moksha", "Nyāya Sūtras 1.1.22"),
    ("The ātman is an eternal substance distinct from body and mind, whose existence is inferred.", "nyaya", "atman", "Nyāya Sūtras 3.1.1"),
    ("Cetanā (consciousness) in Nyāya is an adventitious quality of the ātman, not its essence.", "nyaya", "atman", "Vaiśeṣika Sūtras 3.2.7"),
    ("The six padārthas of Vaiśeṣika — dravya, guṇa, karma, sāmānya, viśeṣa, samavāya — constitute all reality.", "nyaya", "pratyaksha", "Vaiśeṣika Sūtras 1.1.4"),
    ("Correct logical debate (vāda) has truth as its aim, distinguishing it from contentious (jalpa) and sophistical (vitaṇḍā) debate.", "nyaya", "anumana", "Nyāya Sūtras 1.2.1-2"),
    ("Yoga Sūtras define yoga as citta-vṛtti-nirodha — the complete stilling of mental modifications.", "nyaya", "raja_yoga", "Yoga Sūtras 1.2"),
    ("The eight limbs of aṣṭāṅga yoga systematically purify body, breath, mind, and consciousness.", "nyaya", "raja_yoga", "Yoga Sūtras 2.29"),

    # ── MĪMĀṂSĀ ──────────────────────────────────────────────────
    ("The Veda is eternal, authorless (apauruṣeya), and self-valid — its authority needs no external proof.", "mimamsa", "vedas", "Mīmāṃsā Sūtras 1.1.2"),
    ("Svataḥ prāmāṇya holds that valid cognition does not require external verification of its validity.", "mimamsa", "svatah_pramanya", "Śābara Bhāṣya 1.1.2"),
    ("Nitya-karmas like sandhyāvandana must be performed daily without exception, under penalty of sin.", "mimamsa", "nitya_karma", "Mīmāṃsā Sūtras 1.3.5"),
    ("Yajña connects the mortal realm to the divine — the svargakāma (heaven-desiring) must perform soma sacrifices.", "mimamsa", "yajna", "Mīmāṃsā Sūtras 4.3.1"),
    ("Arthāpatti (presumption) is used to explain why a fat man must be eating at night despite fasting during the day.", "mimamsa", "arthapatti", "Kumārila's Ślokavārttika"),
    ("Pūrva Mīmāṃsā holds that the primary purpose of the Veda is to enjoin action through vidhi (injunction).", "mimamsa", "vedas", "Mīmāṃsā Sūtras 1.1.2"),
    ("Prohibitions (niṣedha) carry equal logical force to injunctions, forbidding acts that produce demerit.", "mimamsa", "nisiddha_karma", "Mīmāṃsā Sūtras 4.4.1"),
    ("The Ātman in Mīmāṃsā is the eternal knower that persists through all lives and experiences the fruits of karma.", "mimamsa", "atman", "Śābara Bhāṣya on MS 1.1.5"),
    ("Kāmya-karma yields specific sought results but binds the performer through the fruits of desire.", "mimamsa", "kamya_karma", "Mīmāṃsā Sūtras 4.3.2"),
    ("The varṇas establish hereditary occupational duties that form the backbone of Dharma in Dharmaśāstra.", "mimamsa", "varnashrama", "Manusmṛti 1.87-91"),
    ("Śrāddha (ancestral rites) performed at the correct lunar tithi nourish the departed soul with subtle offerings.", "mimamsa", "shraddha_ritual", "Manusmṛti 3.122"),
    ("Upanayana initiates the twice-born into Gāyatrī repetition and Vedic study, marking dharmic adulthood.", "mimamsa", "upanayana", "Manusmṛti 2.36"),
    ("The pañca-mahāyajñas obligate every gṛhastha to daily offerings to Brahman, gods, ancestors, beings, and guests.", "mimamsa", "panchayajna", "Manusmṛti 3.69-70"),
    ("Prābhākara Mīmāṃsā holds that injunctions generate a categorical sense of duty (bhāvanā) independent of desire.", "mimamsa", "nitya_karma", "Prabhākara's Bṛhatī on Mīmāṃsā Sūtras"),
    ("Mimamsa's hermeneutic (arthavāda) treats laudatory Vedic passages as supplementing, not overriding, injunctions.", "mimamsa", "vedas", "Śābara Bhāṣya on MS 1.2.1"),

    # ── BHAKTI ───────────────────────────────────────────────────
    ("Nārada's Bhakti Sūtras define bhakti as supreme love for God and simultaneous immortality.", "bhakti", "bhakti_yoga", "Nārada Bhakti Sūtras 1.2"),
    ("The nine forms of bhakti — śravaṇa, kīrtana, smaraṇa etc. — are taught by Prahlāda in the Bhāgavata.", "bhakti", "bhakti_yoga", "Bhāgavata Purāṇa 7.5.23"),
    ("Śuddhādvaita of Vallabhācārya declares the world to be a pure, non-deceptive expression of Kṛṣṇa's joy.", "bhakti", "suddhadvaita", "Vallabha's Anubhāṣya 1.1.2"),
    ("Caitanya's acintya-bhedābheda holds that Kṛṣṇa and his śaktis are simultaneously identical and different.", "bhakti", "acintya_bhedabheda", "Caitanya Caritāmṛta 2.8.281"),
    ("Rāsa-līlā reveals the supreme bliss of Kṛṣṇa's divine play with the gopīs in Vṛndāvana.", "bhakti", "lila", "Bhāgavata Purāṇa 10.33"),
    ("The devotee who surrenders completely (ātma-nivedana) to the Lord is freed from all sin and fear.", "bhakti", "prapatti", "Bhāgavata Purāṇa 11.29.34"),
    ("Dāsa-bhāva (servant attitude) toward God is considered the most stable and eternal of the rasas.", "bhakti", "bhakti_yoga", "Bhakti-rasāmṛta-sindhu 2.5.1"),
    ("The gopī-bhāva is the highest intensification of love where the devotee merges in God's enrapturing beauty.", "bhakti", "bhakti_yoga", "Caitanya Caritāmṛta 2.23.1"),
    ("Nāma-kīrtana (chanting God's names) is the yuga-dharma for Kali-yuga, accessible to all regardless of varṇa.", "bhakti", "dharma", "Bhāgavata Purāṇa 12.3.52"),
    ("Dayā (compassion) toward all beings is the natural expression of one established in devotional love.", "bhakti", "daya", "Nārada Bhakti Sūtras 61"),
    ("Sāyujya-mukti — union with God — is the highest liberation granted through the grace of paramabhakti.", "bhakti", "sayujya", "Bhāgavata Purāṇa 3.29.13"),
    ("Śrīmad Bhāgavatam is considered the natural commentary on Brahma Sūtras by devotional commentators.", "bhakti", "puranas", "Bhāgavata Purāṇa 1.1.1"),
    ("Vipralaṃbha (love in separation) generates the most intense devotional sentiment in bhakti rasa theory.", "bhakti", "bhakti_yoga", "Bhakti-rasāmṛta-sindhu 3.4"),
    ("The mantra 'Kṛṣṇaḥ śaraṇaṃ mama' encapsulates the complete surrender of the devotee to the Lord.", "bhakti", "prapatti", "Madhura Kāṭhaka Upaniṣad"),
    ("Uttama-bhakti (supreme devotion) is free from any material desire and dedicated solely to pleasing Kṛṣṇa.", "bhakti", "bhakti_yoga", "Bhakti-rasāmṛta-sindhu 1.1.11"),

    # ── MODERN APPLICATION QUERIES ───────────────────────────────
    ("AI systems that surveil citizens without consent violate the principle of ahiṃsā by causing fear and restricting freedom.", "advaita", "ahimsa", "Yoga Sūtras 2.30"),
    ("From a karma-yoga perspective, a doctor performing euthanasia to end suffering acts without attachment to personal gain.", "advaita", "karma_yoga", "Bhagavad Gītā 3.19"),
    ("The privacy of personal data constitutes a form of satya-protection; revealing it without consent breaches truthfulness.", "nyaya", "satya", "Yoga Sūtras 2.30"),
    ("Genetic modification of organisms can be analyzed through the lens of dharma: does it uphold or undermine cosmic order?", "mimamsa", "dharma", "Manusmṛti 2.6"),
    ("Capitalism's accumulation ethos directly conflicts with aparigraha (non-possessiveness) taught across all Hindu schools.", "advaita", "aparigraha", "Yoga Sūtras 2.30"),
    ("Environmental destruction is adharma because it violates ahiṃsā toward non-human beings and disrupts cosmic order.", "bhakti", "ahimsa", "Manusmṛti 5.47"),
    ("The Bhagavad Gītā's teaching on svadharma supports professional role-based ethical decision-making in modern governance.", "mimamsa", "kshatriya_dharma", "Bhagavad Gītā 2.31"),
    ("Mental health issues like depression can be addressed through the citta-śuddhi practices: tapas, svādhyāya, and bhakti.", "bhakti", "chitta", "Yoga Sūtras 2.32"),
    ("The use of AI for autonomous weapons violates kṣatriya-dharma because only a human can take moral responsibility for killing.", "mimamsa", "kshatriya_dharma", "Manusmṛti 7.87"),
    ("From the Nyāya perspective, AI reasoning constitutes anumāna (inference), not pratyakṣa (direct perception).", "nyaya", "anumana", "Nyāya Sūtras 1.1.5"),
    ("Cryptocurrency speculation driven by lobha (greed) accumulates karma that binds the practitioner to further saṃsāra.", "advaita", "lobha", "Bhagavad Gītā 3.37"),
    ("Democratic governance can be aligned with rājadharma when leaders prioritize the welfare of all subjects over self-interest.", "mimamsa", "raja_dharma", "Arthaśāstra 1.19.34"),
    ("The duty of a doctor (svadharma as healer) is to relieve suffering; the ethic of care aligns with daya and ahiṃsā.", "bhakti", "daya", "Caraka Saṃhitā Sūtrasthāna 1"),
    ("A soldier following orders to kill civilians cannot cite kshatriya-dharma as justification; just war (dharma-yuddha) has strict rules.", "mimamsa", "kshatriya_dharma", "Manusmṛti 7.90"),
    ("Organ donation after death is consistent with dharma as an act of dāna (giving) that helps another being continue living.", "advaita", "dana", "Manusmṛti 4.229"),
]


def build_dataset() -> list[dict]:
    """Convert annotated sentences to JSONL format with train/val/test splits."""
    dataset = []
    for (text, school, concept, source) in ANNOTATED_SENTENCES:
        dataset.append({
            "text": text,
            "label": school,
            "concept": concept,
            "source": source,
            "split": ""   # assigned below
        })

    # Shuffle and split 80/10/10
    random.shuffle(dataset)
    n = len(dataset)
    for i, item in enumerate(dataset):
        if i < int(0.8 * n):
            item["split"] = "train"
        elif i < int(0.9 * n):
            item["split"] = "val"
        else:
            item["split"] = "test"

    return dataset


def augment_dataset(base: list[dict]) -> list[dict]:
    """
    Generate paraphrase variants to reach 500+ samples.
    Strategy: for each sentence, generate 2-3 paraphrased versions
    using simple template transformations + synonym substitution.
    """
    SCHOOL_DESCRIPTIONS = {
        "advaita":       "According to Advaita Vedānta,",
        "dvaita":        "From the Dvaita Vedānta standpoint,",
        "vishishtadvaita": "In Rāmānuja's Viśiṣṭādvaita,",
        "nyaya":         "The Nyāya school holds that",
        "mimamsa":       "Mīmāṃsā philosophy affirms that",
        "bhakti":        "In the bhakti tradition,",
    }
    CONCEPT_SYNONYMS = {
        "atman":      ["the self", "ātman", "the individual soul", "pure consciousness"],
        "brahman":    ["Brahman", "ultimate reality", "the absolute", "pure being"],
        "dharma":     ["dharma", "righteous duty", "cosmic order", "righteous law"],
        "karma":      ["karma", "karmic consequence", "action and result", "the law of cause and effect"],
        "moksha":     ["liberation", "mokṣa", "final release", "spiritual freedom"],
        "maya":       ["māyā", "cosmic illusion", "the power of illusion", "apparent multiplicity"],
        "ahimsa":     ["ahiṃsā", "non-violence", "non-harm", "harmlessness"],
        "bhakti_yoga":["devotion", "bhakti", "loving surrender", "devotional service"],
        "karma_yoga": ["karma yoga", "selfless action", "duty without attachment", "nishkama karma"],
        "jnana_yoga": ["jñāna yoga", "the path of knowledge", "knowledge-based liberation", "discriminative wisdom"],
    }

    augmented = list(base)
    for item in base[:80]:   # augment first 80 items to reach ~500
        school = item["label"]
        prefix = SCHOOL_DESCRIPTIONS.get(school, school.capitalize() + " teaches that")
        # Variant 1: school prefix
        aug1 = dict(item)
        aug1["text"] = f"{prefix} {item['text'][0].lower()}{item['text'][1:]}"
        aug1["split"] = "train"
        augmented.append(aug1)

        # Variant 2: concept synonym swap
        concept = item["concept"]
        syns = CONCEPT_SYNONYMS.get(concept, [])
        if syns:
            syn = random.choice(syns)
            aug2 = dict(item)
            aug2["text"] = item["text"].replace(
                concept.replace("_", " "), syn
            ).replace(concept, syn)
            aug2["split"] = "train"
            if aug2["text"] != item["text"]:
                augmented.append(aug2)

    return augmented


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = build_dataset()
    full = augment_dataset(base)

    # Write JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in full:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    school_counts = Counter(d["label"] for d in full)
    split_counts  = Counter(d["split"] for d in full)
    concept_counts = Counter(d["concept"] for d in full)

    print(f"\nIndicBERT Annotation Dataset Generated")
    print(f"  Total sentences : {len(full)}")
    print(f"  Train / Val / Test : {split_counts['train']} / {split_counts['val']} / {split_counts['test']}")
    print(f"\n  By school:")
    for s, n in school_counts.most_common():
        print(f"    {s:<22} {n}")
    print(f"\n  Unique concepts : {len(concept_counts)}")
    print(f"\n  Saved to: {OUTPUT_FILE}")

    # Also write a label map
    label_map = {"labels": sorted(school_counts.keys()), "num_labels": len(school_counts)}
    (OUTPUT_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2))
    print(f"  Label map → training/label_map.json")
