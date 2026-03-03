"""
populate_kg.py — Seed the HinduMind HPO Knowledge Graph

Loads all JSON seed data and populates the HPOGraph.
Exports: RDF (Turtle), GraphML, JSON.
"""

import json
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.hpo_graph import HPOGraph

SEED_DIR = Path(__file__).parent / "seed_data"
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_json(filename: str) -> list:
    path = SEED_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Strip JS-style comments (// ...) so we can reuse the annotated JSON
    import re
    text = re.sub(r"//[^\n]*", "", text)
    return json.loads(text)


def populate(graph: HPOGraph) -> HPOGraph:
    # ── 1. Concepts ──────────────────────────────────────────────────────
    concepts = load_json("concepts.json")
    for c in concepts:
        graph.add_concept(
            concept_id=c["id"],
            label=c["label"],
            concept_type=c.get("type", "Concept"),
            definition=c.get("definition", ""),
            sanskrit=c.get("sanskrit", ""),
            related_texts=c.get("related_texts", [])
        )
    print(f"[populate] Loaded {len(concepts)} concepts")

    # ── 2. Texts ─────────────────────────────────────────────────────────
    texts = load_json("texts.json")
    for t in texts:
        graph.add_text(
            text_id=t["id"],
            label=t["label"],
            text_type=t.get("type", "Text"),
            date_approx=t.get("date_approx", ""),
            description=t.get("description", ""),
            sanskrit=t.get("sanskrit", ""),
            school_relevance=t.get("school_relevance", [])
        )
    print(f"[populate] Loaded {len(texts)} texts")

    # ── 3. Schools ───────────────────────────────────────────────────────
    schools = load_json("schools.json")
    for s in schools:
        graph.add_school(
            school_id=s["id"],
            label=s["label"],
            core_doctrine=s.get("core_doctrine", ""),
            founder=s.get("founder", ""),
            period=s.get("period", ""),
            sanskrit=s.get("sanskrit", "")
        )
    print(f"[populate] Loaded {len(schools)} schools")

    # ── 4. Commentators ──────────────────────────────────────────────────
    commentators = load_json("commentators.json")
    for c in commentators:
        graph.add_commentator(
            commentator_id=c["id"],
            label=c["label"],
            school=c.get("school", ""),
            period=c.get("period", ""),
            key_contribution=c.get("key_contribution", ""),
            sanskrit=c.get("sanskrit", "")
        )
    print(f"[populate] Loaded {len(commentators)} commentators")

    # ── 5. Relations ─────────────────────────────────────────────────────
    relations = load_json("relations.json")
    loaded_rels = 0
    skipped = 0
    for r in relations:
        subj = r.get("subject")
        pred = r.get("predicate")
        obj  = r.get("object")
        if not (subj and pred and obj):
            skipped += 1
            continue
        # Skip if nodes don't exist yet (graceful)
        if subj not in graph._nodes or obj not in graph._nodes:
            # Still add — NX allows dangling edges which we can fix later
            pass
        graph.add_relation(
            subject_id=subj,
            predicate=pred,
            object_id=obj,
            note=r.get("note", ""),
            source=r.get("source", ""),
            confidence=float(r.get("confidence", 1.0))
        )
        loaded_rels += 1
    print(f"[populate] Loaded {loaded_rels} relations ({skipped} skipped)")

    return graph


def main():
    print("=" * 60)
    print("  HinduMind — Knowledge Graph Population")
    print("=" * 60)

    graph = HPOGraph()
    graph = populate(graph)

    # ── Print Stats ──────────────────────────────────────────────────────
    stats = graph.stats()
    print("\n── KG Statistics ─────────────────────────────────────────")
    print(f"  Total Nodes : {stats['total_nodes']}")
    print(f"  Total Edges : {stats['total_edges']}")
    print(f"  RDF Triples : {stats['rdf_triples']}")
    print("\n  Node Types:")
    for t, cnt in sorted(stats["node_types"].items()):
        print(f"    {t:<30} {cnt}")
    print("\n  Edge Predicates:")
    for p, cnt in sorted(stats["edge_predicates"].items()):
        print(f"    {p:<30} {cnt}")

    # ── Export ───────────────────────────────────────────────────────────
    graph.export_rdf(str(OUT_DIR / "hpo_graph.ttl"), format="turtle")
    graph.export_graphml(str(OUT_DIR / "hpo_graph.graphml"))
    graph.export_json(str(OUT_DIR / "hpo_graph.json"))

    print("\n[Done] KG exported to kg/output/")
    return graph


if __name__ == "__main__":
    main()
