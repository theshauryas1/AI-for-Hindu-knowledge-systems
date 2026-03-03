# Hindu Philosophy Ontology (HPO) — Node & Edge Taxonomy

## Overview

The HPO defines a formal vocabulary for representing entities in Hindu philosophical discourse. It spans 6 major schools (darśanas), 13 entity classes, 15 object properties (edge types), and 9 datatype properties.

---

## Node Types (Classes)

### 1. `Concept`
Generic philosophical concept. Divided into three sub-types:

| Sub-type | Description | Examples |
|----------|-------------|---------|
| `MetaphysicalConcept` | Nature of reality, consciousness, existence | ātman, brahman, māyā, mokṣa, jagat |
| `EthicalConcept` | Moral duty, right action | dharma, karma, ahiṃsā, satya, ṛṇa |
| `EpistemicConcept` | Knowledge sources, inference | pramāṇa, anumāna, pratyakṣa, śabda, upamāna |

### 2. `Text`
Canonical Hindu philosophical writings. Three sub-types:

| Sub-type | Description | Examples |
|----------|-------------|---------|
| `Shruti` | Revealed scripture | Ṛgveda, Chāndogya Upaniṣad, Bṛhadāraṇyaka Upaniṣad |
| `Smriti` | Transmitted texts | Bhagavad Gītā, Manusmṛti, Mahābhārata, Rāmāyaṇa |
| `Sutra` | Aphoristic texts | Brahma Sūtras, Nyāya Sūtras, Mīmāṃsā Sūtras |

### 3. `School`
A distinct philosophical school (darśana):

| School | Sanskrit | Key Founding Text |
|--------|----------|------------------|
| Advaita Vedānta | अद्वैत | Māṇḍūkya Kārikā, Vivekacūḍāmaṇi |
| Dvaita Vedānta | द्वैत | Anuvyākhyāna, Tattva Viveka |
| Viśiṣṭādvaita | विशिष्टाद्वैत | Śrī Bhāṣya, Gīta Bhāṣya |
| Nyāya-Vaiśeṣika | न्याय-वैशेषिक | Nyāya Sūtras, Vaiśeṣika Sūtras |
| Mīmāṃsā | मीमांसा | Mīmāṃsā Sūtras, Śābara Bhāṣya |
| Bhakti | भक्ति | Bhāgavata Purāṇa, Nārada Bhakti Sūtras |

### 4. `Commentator`
A historical philosopher affiliated with a school:

| Name | School | Dates (approx.) |
|------|--------|----------------|
| Śaṅkarācārya | Advaita | 788–820 CE |
| Rāmānujācārya | Viśiṣṭādvaita | 1017–1137 CE |
| Madhvācārya | Dvaita | 1238–1317 CE |
| Gautama | Nyāya | ~150 BCE |
| Jaiminī | Mīmāṃsā | ~400 BCE |
| Patañjali | Yoga/Sāṃkhya | ~200 BCE–200 CE |
| Nimbārka | Dvaitādvaita | ~12th CE |
| Vallabhācārya | Śuddhādvaita | 1479–1531 CE |

### 5. `Principle`
A foundational doctrinal stance:

| Principle | Associated School |
|-----------|-----------------|
| Non-dualism (advaita) | Advaita Vedānta |
| Strict dualism (bheda) | Dvaita Vedānta |
| Qualified non-dualism | Viśiṣṭādvaita |
| Logical inference (anumāna) | Nyāya |
| Vedic eternalism (nityatva) | Mīmāṃsā |
| Divine grace (prasāda) | Bhakti |
| Atomic theory (paramāṇu) | Vaiśeṣika |

### 6. `EthicalRule`
A prescriptive rule from Dharmaśāstra with deontic formalization:

Structure:
- `IF [condition]`
- `THEN [obligation/permission/prohibition]`
- `SOURCE: [text + verse]`
- `SCHOOL_WEIGHT: {school: HIGH/MEDIUM/LOW}`
- `VARNA: [applicable varna]`
- `ASHRAMA: [applicable āśrama]`

### 7. `EthicalScenario`
A real or hypothetical situation requiring dharmic analysis:

Examples:
- "A soldier is ordered to kill prisoners"
- "A brahmin is asked to lie to save a life"
- "A householder faces a conflict between family duty and renunciation"

### 8. `DeonticJudgment`
A formal deontic conclusion produced by the Dharma Engine:

| Operator | Symbol | Meaning |
|----------|--------|---------|
| Obligatory | O(p) | Action p must be performed |
| Permitted | P(p) | Action p may be performed |
| Forbidden | F(p) | Action p must not be performed |

---

## Edge Types (Object Properties)

| Edge | Domain | Range | Description |
|------|--------|-------|-------------|
| `contradicts` | Any | Any | School A contradicts School B on a doctrine |
| `supports` | Any | Any | Text/school supports a doctrine |
| `interprets` | Commentator | Text | Commentator interprets a text |
| `authored` | Commentator | Text | Commentator is the author |
| `belongsTo` | Commentator | School | Commentator belongs to a school |
| `evolves_from` | School | School | School B evolved from School A |
| `is_defined_in` | Concept | Text | Concept is defined in a text |
| `is_endorsed_by` | Concept | School | Concept is central to a school |
| `is_rejected_by` | Concept | School | Concept is rejected by a school |
| `applies_to` | EthicalRule | EthicalScenario | Rule applies to a scenario |
| `grounds` | Text | EthicalRule | Text grounds an ethical rule |
| `embodies` | School | Principle | School embodies a principle |
| `related_to` | Concept | Concept | Symmetric conceptual relation |
| `is_prerequisite_for` | Concept | Concept | Concept A prerequisite for B |
| `results_in` | Concept | Concept | Concept A leads to Concept B |

---

## Cardinality Constraints

| Class | Min Nodes | Target |
|-------|-----------|--------|
| Concept (all) | 150 | 300+ |
| Text | 30 | 60+ |
| School | 6 | 6 |
| Commentator | 8 | 20+ |
| Principle | 10 | 20+ |
| EthicalRule | 80 | 150+ |
| EthicalScenario | 20 | 50+ |

| Edge Type | Min Triples | Target |
|-----------|-------------|--------|
| All combined | 500 | 1000+ |

---

## Design Principles

1. **Open World Assumption**: RDF/OWL semantics; absence of a triple ≠ negation.
2. **School-neutral core**: Core concept definitions are school-neutral; school-specific stances modelled as `is_endorsed_by` / `is_rejected_by` edges.
3. **Source-anchored**: Every concept or rule node should have at least one `is_defined_in` or `grounds` edge to a canonical text.
4. **Deontic separability**: EthicalRules are kept separate from Concepts to allow the Dharma Engine to reason over them independently.
5. **IAST transliteration**: All Sanskrit names use IAST standard for academic interoperability.
