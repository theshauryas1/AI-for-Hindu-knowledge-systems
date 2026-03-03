"""
hpo_graph.py — Hindu Philosophy Ontology Knowledge Graph Engine

Core class wrapping RDFLib + NetworkX for the HPO knowledge graph.
Provides methods to add/query nodes, edges, and run graph algorithms.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import networkx as nx
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

# ─────────────────────────────────────────
# HPO Namespace
# ─────────────────────────────────────────

HPO = Namespace("http://hindumind.org/ontology/hpo#")
BASE = Namespace("http://hindumind.org/entity/")

# ─────────────────────────────────────────
# Entity Classes
# ─────────────────────────────────────────

ENTITY_CLASSES = {
    "MetaphysicalConcept": HPO.MetaphysicalConcept,
    "EthicalConcept":      HPO.EthicalConcept,
    "EpistemicConcept":    HPO.EpistemicConcept,
    "Concept":             HPO.Concept,
    "Text":                HPO.Text,
    "Shruti":              HPO.Shruti,
    "Smriti":              HPO.Smriti,
    "Sutra":               HPO.Sutra,
    "School":              HPO.School,
    "Commentator":         HPO.Commentator,
    "Principle":           HPO.Principle,
    "EthicalRule":         HPO.EthicalRule,
    "EthicalScenario":     HPO.EthicalScenario,
    "DeonticJudgment":     HPO.DeonticJudgment,
}

# ─────────────────────────────────────────
# HPOGraph
# ─────────────────────────────────────────

class HPOGraph:
    """
    Dual-layer knowledge graph:
      - rdf_graph: RDFLib graph for semantic queries (SPARQL)
      - nx_graph:  NetworkX DiGraph for path/centrality algorithms
    """

    def __init__(self):
        self.rdf_graph = Graph()
        self.rdf_graph.bind("hpo", HPO)
        self.rdf_graph.bind("base", BASE)
        self.rdf_graph.bind("owl", OWL)
        self.rdf_graph.bind("rdfs", RDFS)

        self.nx_graph = nx.MultiDiGraph()

        # In-memory node registry {id: {label, type, ...}}
        self._nodes: dict[str, dict] = {}

    # ─────────────────────────────────────
    # Node Addition Methods
    # ─────────────────────────────────────

    def _add_node(self, node_id: str, node_type: str, label: str, properties: dict):
        """Internal: add a node to both RDF and NX graphs."""
        uri = BASE[node_id]
        rdf_class = ENTITY_CLASSES.get(node_type, HPO.PhilosophicalEntity)

        self.rdf_graph.add((uri, RDF.type, rdf_class))
        self.rdf_graph.add((uri, HPO.hasLabel, Literal(label, lang="en")))

        for key, val in properties.items():
            if val:
                self.rdf_graph.add((uri, HPO[key], Literal(str(val), datatype=XSD.string)))

        self.nx_graph.add_node(node_id, label=label, type=node_type, **properties)
        self._nodes[node_id] = {"label": label, "type": node_type, **properties}

    def add_concept(self, concept_id: str, label: str, concept_type: str = "Concept",
                    definition: str = "", sanskrit: str = "", related_texts: list = None):
        self._add_node(concept_id, concept_type, label, {
            "hasSanskritName": sanskrit,
            "hasDefinition": definition,
            "related_texts": json.dumps(related_texts or [])
        })

    def add_text(self, text_id: str, label: str, text_type: str = "Text",
                 date_approx: str = "", description: str = "", sanskrit: str = "",
                 school_relevance: list = None):
        self._add_node(text_id, text_type, label, {
            "hasSanskritName": sanskrit,
            "hasDefinition": description,
            "date_approx": date_approx,
            "school_relevance": json.dumps(school_relevance or [])
        })

    def add_school(self, school_id: str, label: str, core_doctrine: str = "",
                   founder: str = "", period: str = "", sanskrit: str = ""):
        self._add_node(school_id, "School", label, {
            "hasSanskritName": sanskrit,
            "hasDefinition": core_doctrine,
            "founder": founder,
            "period": period
        })

    def add_commentator(self, commentator_id: str, label: str, school: str = "",
                        period: str = "", key_contribution: str = "", sanskrit: str = ""):
        self._add_node(commentator_id, "Commentator", label, {
            "hasSanskritName": sanskrit,
            "hasDefinition": key_contribution,
            "period": period,
            "school": school
        })

    def add_ethical_rule(self, rule_id: str, condition: str, obligation: str,
                         source: str = "", school_weights: dict = None,
                         varna: str = "all", ashrama: str = "all",
                         deontic_operator: str = "O"):
        label = f"Rule: {rule_id}"
        props = {
            "hasCondition": condition,
            "hasObligation": obligation,
            "hasSourceVerse": source,
            "hasSchoolWeight": json.dumps(school_weights or {}),
            "hasVarna": varna,
            "hasAshrama": ashrama,
            "hasDeonticOperator": deontic_operator
        }
        self._add_node(rule_id, "EthicalRule", label, props)

    # ─────────────────────────────────────
    # Relation (Edge) Addition
    # ─────────────────────────────────────

    def add_relation(self, subject_id: str, predicate: str, object_id: str,
                     note: str = "", source: str = "", confidence: float = 1.0):
        """Add a directed edge between two nodes."""
        subj_uri = BASE[subject_id]
        obj_uri  = BASE[object_id]
        pred_uri = HPO[predicate]

        self.rdf_graph.add((subj_uri, pred_uri, obj_uri))
        if note:
            reif = BASE[f"{subject_id}_{predicate}_{object_id}"]
            self.rdf_graph.add((reif, RDF.type, RDF.Statement))
            self.rdf_graph.add((reif, RDF.subject, subj_uri))
            self.rdf_graph.add((reif, RDF.predicate, pred_uri))
            self.rdf_graph.add((reif, RDF.object, obj_uri))
            self.rdf_graph.add((reif, RDFS.comment, Literal(note, lang="en")))

        self.nx_graph.add_edge(subject_id, object_id,
                               predicate=predicate, note=note,
                               source=source, confidence=confidence)

    # ─────────────────────────────────────
    # Query Methods
    # ─────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[dict]:
        """Return node metadata dict or None."""
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: str, predicate: str = None) -> list[dict]:
        """Return all neighbor nodes (outgoing edges), optionally filtered by predicate."""
        result = []
        for _, nbr, data in self.nx_graph.out_edges(node_id, data=True):
            if predicate is None or data.get("predicate") == predicate:
                nbr_node = self._nodes.get(nbr, {"id": nbr})
                result.append({"id": nbr, "predicate": data.get("predicate"), **nbr_node})
        return result

    def query_by_school(self, school_id: str, relation: str = "is_endorsed_by") -> list[str]:
        """Return all concept IDs endorsed/rejected/supported by a school."""
        concepts = []
        for node_id, nbrs in self.nx_graph.pred[school_id].items():
            for _, data in nbrs.items():
                if data.get("predicate") == relation:
                    concepts.append(node_id)
        return concepts

    def get_school_contradictions(self) -> list[dict]:
        """Return all school contradiction pairs."""
        contradictions = []
        for u, v, data in self.nx_graph.edges(data=True):
            if data.get("predicate") == "contradicts":
                contradictions.append({
                    "school_a": u,
                    "school_b": v,
                    "concept": data.get("concept", ""),
                    "note": data.get("note", "")
                })
        return contradictions

    def get_concept_cross_school_view(self, concept_id: str) -> dict:
        """
        Return a structured view of how all schools relate to a concept.
        Returns: {school_id: {stance: endorsed/rejected/neutral, note: str}}
        """
        result = {}
        for _, nbr, data in self.nx_graph.out_edges(concept_id, data=True):
            pred = data.get("predicate")
            if pred in ("is_endorsed_by", "is_rejected_by") and self._nodes.get(nbr, {}).get("type") == "School":
                stance = "endorsed" if pred == "is_endorsed_by" else "rejected"
                result[nbr] = {"stance": stance, "note": data.get("note", "")}
        return result

    def get_texts_for_concept(self, concept_id: str) -> list[dict]:
        """Return all texts where a concept is defined."""
        texts = []
        for _, nbr, data in self.nx_graph.out_edges(concept_id, data=True):
            if data.get("predicate") == "is_defined_in":
                text_node = self._nodes.get(nbr, {})
                texts.append({"id": nbr, "source": data.get("source", ""), **text_node})
        return texts

    def get_school_specific_triples(self, school_id: str) -> list[dict]:
        """
        Return all triples where the school appears (endorsed, rejected, supported).
        Useful for KG-grounded agent prompts.
        """
        triples = []
        # Outgoing from school
        for _, nbr, data in self.nx_graph.out_edges(school_id, data=True):
            triples.append({"subject": school_id, "predicate": data["predicate"], "object": nbr})
        # Incoming to school
        for pred, _, data in self.nx_graph.in_edges(school_id, data=True):
            triples.append({"subject": pred, "predicate": data["predicate"], "object": school_id})
        return triples

    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        """Return shortest path between two nodes (ignoring edge direction)."""
        try:
            return nx.shortest_path(self.nx_graph.to_undirected(), source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_nodes_by_type(self, node_type: str) -> list[dict]:
        """Return all nodes of a given type."""
        return [{"id": nid, **data} for nid, data in self._nodes.items()
                if data.get("type") == node_type]

    def sample_triples(self, n: int = 100) -> list[dict]:
        """Return n random edge triples for expert spot-check evaluation."""
        all_edges = [(u, v, d) for u, v, d in self.nx_graph.edges(data=True)]
        sample = random.sample(all_edges, min(n, len(all_edges)))
        return [{"subject": u, "predicate": d.get("predicate",""), "object": v,
                 "note": d.get("note",""), "source": d.get("source","")}
                for u, v, d in sample]

    # ─────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────

    def stats(self) -> dict:
        """Return KG coverage statistics."""
        type_count = {}
        for _, data in self._nodes.items():
            t = data.get("type", "Unknown")
            type_count[t] = type_count.get(t, 0) + 1

        pred_count = {}
        for _, _, data in self.nx_graph.edges(data=True):
            p = data.get("predicate", "unknown")
            pred_count[p] = pred_count.get(p, 0) + 1

        return {
            "total_nodes": self.nx_graph.number_of_nodes(),
            "total_edges": self.nx_graph.number_of_edges(),
            "rdf_triples": len(self.rdf_graph),
            "node_types": type_count,
            "edge_predicates": pred_count
        }

    # ─────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────

    def export_rdf(self, path: str, format: str = "turtle"):
        """Export RDF graph to file (turtle, xml, json-ld)."""
        self.rdf_graph.serialize(destination=path, format=format)
        print(f"[HPOGraph] RDF exported → {path}")

    def export_graphml(self, path: str):
        """Export NetworkX graph as GraphML for Gephi/Cytoscape visualization."""
        nx.write_graphml(self.nx_graph, path)
        print(f"[HPOGraph] GraphML exported → {path}")

    def export_json(self, path: str):
        """Export full graph as JSON (nodes + edges)."""
        data = {
            "nodes": [{"id": nid, **nd} for nid, nd in self._nodes.items()],
            "edges": [{"source": u, "target": v, **d}
                      for u, v, d in self.nx_graph.edges(data=True)]
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[HPOGraph] JSON exported → {path}")
