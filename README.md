# HinduMind 🕉️

**A Knowledge Graph-Driven Multi-Agent System for Hindu Philosophical Reasoning and Dharmic Ethical Analysis**

---

## Overview

HinduMind is a research prototype that models multi-school Hindu philosophical reasoning using:

- **Hindu Philosophy Ontology (HPO)** — formal RDF/OWL knowledge graph across 6 darśanas
- **Multi-Agent Reasoning System** — parallel agents for Advaita, Dvaita, Nyāya + a Meta-Agent synthesizer
- **Computational Dharma Engine** — deontic logic formalization of Dharmaśāstra ethical rules

---

## System Architecture

```
User Query
    ↓
Intent + School Detector (IndicBERT classifier)
    ↓
Hindu Knowledge Graph (RDFLib + NetworkX)
    ↓
┌─────────────────────────────────────┐
│         Multi-Agent Reasoner        │
│  Agent 1: Advaita Vedānta           │
│  Agent 2: Dvaita Vedānta            │
│  Agent 3: Nyāya-Vaiśeṣika           │
│  Meta-Agent: Synthesis              │
└─────────────────────────────────────┘
    ↓
Dharma Engine (Deontic Logic + Context Parser)
    ↓
Structured JSON Output
```

---

## Project Structure

```
hinduмind/
├── ontology/           # HPO schema (Turtle/RDF) + taxonomy docs
├── kg/                 # KG construction, population, evaluation
│   └── seed_data/      # Curated JSON: concepts, texts, schools, relations
├── agents/             # School classifer + per-school agents + meta-agent
├── dharma_engine/      # Rule extractor, context parser, deontic reasoner
├── evaluation/         # KG metrics, expert surveys, case studies
├── data/               # Annotated training data
├── notebooks/          # Demo Jupyter notebooks
├── main.py             # Unified CLI entry point
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Populate the knowledge graph
python kg/populate_kg.py

# Query the multi-agent system
python main.py query --text "What is the nature of atman in Advaita Vedanta?"

# Run Dharma Engine on an ethical scenario
python main.py dharma --scenario "Is it dharmic for a kshatriya to refuse battle?"

# Run case studies
python evaluation/case_studies.py

# Evaluate KG quality
python main.py eval --module kg
```

---

## Configuration

Copy `.env.example` to `.env` and set your LLM backend:

```env
LLM_BACKEND=ollama          # or openai / huggingface
OLLAMA_MODEL=mistral        # or llama3
OPENAI_API_KEY=sk-...       # if using OpenAI
```

---

## Philosophical Schools Covered

| School | Sanskrit | Key Thinker | Core Doctrine |
|--------|----------|-------------|---------------|
| Advaita Vedānta | अद्वैत वेदान्त | Śaṅkarācārya | Non-dualism; Brahman = Ātman |
| Dvaita Vedānta | द्वैत वेदान्त | Madhvācārya | Strict dualism; Brahman ≠ Ātman |
| Viśiṣṭādvaita | विशिष्टाद्वैत | Rāmānujācārya | Qualified non-dualism |
| Nyāya-Vaiśeṣika | न्याय-वैशेषिक | Gautama / Kaṇāda | Logic, inference, atomism |
| Mīmāṃsā | मीमांसा | Jaiminī | Vedic ritualism, dharma literalism |
| Bhakti | भक्ति | Multiple | Devotional theism, divine grace |

---

## Research Context

- **Target Journal**: ACM Journal on Computing and Cultural Heritage (JOCCH)
- **Novel Contributions**:
  1. First formal OWL ontology for multi-school Hindu philosophy (HPO)
  2. First multi-agent doctrinal reasoning system across darśanas
  3. First formalization of Dharmaśāstra via deontic logic
  4. End-to-end KG + multi-agent + ethical reasoning for Hindu knowledge

---

## Data Sources

All sources are publicly available:
- [GRETIL](http://gretil.sub.uni-goettingen.de/) — Sanskrit texts corpus
- [Digital Corpus of Sanskrit (DCS)](http://www.sanskrit-linguistics.org/dcs/)
- [Wisdom Library](https://www.wisdomlib.org/) — English summaries
- [IIT-Bombay Sanskrit Resources](https://www.cfilt.iitb.ac.in/)
