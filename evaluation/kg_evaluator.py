"""
kg_evaluator.py — Knowledge Graph Quality Evaluation

Produces:
  1. Random triple spot-check CSV (for expert annotation)
  2. Coverage statistics by node type and edge predicate
  3. Inter-annotator agreement template (for Cohen's κ)
"""

import csv
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.populate_kg import populate
from kg.hpo_graph import HPOGraph


OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def generate_spot_check_csv(kg: HPOGraph, n: int = 100) -> str:
    """
    Export n random triples to CSV for expert spot-check.
    Columns: id, subject, predicate, object, source, correctness (blank for expert)
    """
    triples = kg.sample_triples(n)
    path = str(OUT_DIR / "kg_spot_check.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id","subject","predicate","object","source","note",
                           "expert_1_correct","expert_2_correct","agreed"]
        )
        writer.writeheader()
        for i, triple in enumerate(triples, 1):
            writer.writerow({
                "id": i,
                "subject": triple["subject"],
                "predicate": triple["predicate"],
                "object": triple["object"],
                "source": triple.get("source", ""),
                "note": triple.get("note", ""),
                "expert_1_correct": "",   # To be filled by Expert 1
                "expert_2_correct": "",   # To be filled by Expert 2
                "agreed": ""              # Computed post-annotation
            })

    print(f"[KGEval] Spot-check CSV exported → {path} ({len(triples)} triples)")
    return path


def compute_coverage_stats(kg: HPOGraph) -> dict:
    """Compute and display coverage statistics."""
    stats = kg.stats()

    print(f"\n── KG Coverage Statistics ──────────────────────────────")
    print(f"  Total Nodes : {stats['total_nodes']}")
    print(f"  Total Edges : {stats['total_edges']}")
    print(f"  RDF Triples : {stats['rdf_triples']}")
    print(f"\n  Node Types:")
    for t, cnt in sorted(stats["node_types"].items()):
        bar = "█" * min(cnt, 30)
        print(f"    {t:<28} {cnt:>4}  {bar}")
    print(f"\n  Edge Predicates:")
    for p, cnt in sorted(stats["edge_predicates"].items(), key=lambda x: -x[1]):
        bar = "█" * min(cnt, 30)
        print(f"    {p:<28} {cnt:>4}  {bar}")

    # Save to JSON
    path = str(OUT_DIR / "kg_coverage_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[KGEval] Coverage stats saved → {path}")
    return stats


def generate_iaa_template() -> str:
    """
    Generate an Inter-Annotator Agreement (IAA) instruction sheet.
    Used with kg_spot_check.csv for Cohen's κ computation.
    """
    template = """
# HPO Knowledge Graph — Expert Annotation Guide
# Spot-Check for Relation Accuracy (Cohen's κ Evaluation)

## Instructions
For each triple in kg_spot_check.csv, annotate the 'correctness' column as:
  1 = CORRECT   (relation is accurate and well-sourced)
  0 = INCORRECT (relation is wrong or misleading)
  ? = UNCERTAIN (annotator unsure; discuss with other expert)

## Annotation Protocol
- Please annotate independently before comparing with the other expert.
- A triple is CORRECT if:
    (a) The relation (predicate) accurately describes how subject relates to object.
    (b) The source citation (if present) is accurate.
    (c) The claim is consistent with mainstream scholarship on the school/text.
- Do NOT correct the data — only mark correctness.

## Cohen's κ Computation
After both experts annotate, κ is computed with:
  python evaluation/kg_evaluator.py --compute-kappa kg_spot_check.csv

## Relation Guide
| Predicate         | Meaning                                          |
|-------------------|--------------------------------------------------|
| is_defined_in     | Concept is defined/elaborated in this text       |
| is_endorsed_by    | Concept is central to / affirmed by this school  |
| is_rejected_by    | Concept is explicitly denied by this school      |
| contradicts       | School A contradicts School B on a topic         |
| interprets        | Commentator wrote a commentary on this text      |
| belongsTo         | Commentator is affiliated with this school       |
| evolves_from      | School B developed in response to School A       |
| supports          | Text/school explicitly supports this doctrine    |
| results_in        | Concept A causally leads to Concept B            |
"""
    path = str(OUT_DIR / "iaa_annotation_guide.md")
    Path(path).write_text(template.strip(), encoding="utf-8")
    print(f"[KGEval] IAA guide saved → {path}")
    return path


def compute_cohens_kappa(csv_path: str = None) -> float:
    """
    Compute Cohen's κ from annotated spot-check CSV.
    Requires expert_1_correct and expert_2_correct columns filled in.
    """
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    csv_path = csv_path or str(OUT_DIR / "kg_spot_check.csv")
    df = pd.read_csv(csv_path)

    # Filter rows where both experts annotated (non-empty, non-'?')
    df = df[df["expert_1_correct"].astype(str).isin(["0", "1"]) &
            df["expert_2_correct"].astype(str).isin(["0", "1"])]

    if len(df) < 10:
        print("[KGEval] Not enough annotations to compute κ (need ≥10 rows).")
        return float("nan")

    k = cohen_kappa_score(
        df["expert_1_correct"].astype(int),
        df["expert_2_correct"].astype(int)
    )
    print(f"[KGEval] Cohen's κ = {k:.4f}  (n={len(df)} annotated pairs)")
    return k


def run_full_evaluation(kg: HPOGraph = None) -> dict:
    if kg is None:
        kg = HPOGraph()
        populate(kg)

    stats  = compute_coverage_stats(kg)
    csv_p  = generate_spot_check_csv(kg, n=100)
    iaa_p  = generate_iaa_template()

    return {
        "coverage_stats": stats,
        "spot_check_csv": csv_p,
        "iaa_guide": iaa_p
    }


if __name__ == "__main__":
    import sys
    if "--compute-kappa" in sys.argv:
        idx = sys.argv.index("--compute-kappa")
        path = sys.argv[idx+1] if idx+1 < len(sys.argv) else None
        compute_cohens_kappa(path)
    else:
        run_full_evaluation()
