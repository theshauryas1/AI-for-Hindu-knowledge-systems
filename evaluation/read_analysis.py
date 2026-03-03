"""
read_analysis.py — ASCII-safe reader for dimensional_analysis.json
"""
import json
from pathlib import Path

j = json.load(open("evaluation/results/dimensional_analysis.json", encoding="utf-8"))
breakdown = j["breakdown"]

GROUPS = {
    "direct_textual":       "A. Direct Textual",
    "contextual_extension": "B. Contextual Extension",
    "modern_analog":        "C. Modern Analog",
    "ambiguity_stress":     "D. Ambiguity Stress",
}
MODELS = ["vanilla_llm", "rag", "symbolic", "hybrid"]

def pct(v):
    return f"{v:.3f}" if v is not None else " N/A"

print("\n=== TABLE 1: PARTIAL ACCURACY BY SCENARIO GROUP ===")
print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
print("-"*65)
for grp, label in GROUPS.items():
    gd = breakdown.get(grp, {})
    row = f"{label:<26}"
    for m in MODELS:
        row += f" {pct(gd.get(m, {}).get('partial')):>9}"
    print(row)

print("\n=== TABLE 2: CITATION INTEGRITY ===")
print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
print("-"*65)
for grp, label in GROUPS.items():
    gd = breakdown.get(grp, {})
    row = f"{label:<26}"
    for m in MODELS:
        row += f" {pct(gd.get(m, {}).get('cite_integrity')):>9}"
    print(row)

print("\n=== TABLE 3: PLURALISM (diversity ratio) ===")
print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
print("-"*65)
for grp, label in GROUPS.items():
    gd = breakdown.get(grp, {})
    row = f"{label:<26}"
    for m in MODELS:
        row += f" {pct(gd.get(m, {}).get('plurality')):>9}"
    print(row)

print("\n=== TABLE 4: CONFLICT DETECTION ACCURACY ===")
print(f"{'Group':<26} {'Vanilla':>9} {'RAG':>9} {'Symbolic':>9} {'Hybrid':>9}")
print("-"*65)
for grp, label in GROUPS.items():
    gd = breakdown.get(grp, {})
    row = f"{label:<26}"
    for m in MODELS:
        row += f" {pct(gd.get(m, {}).get('conflict_correct')):>9}"
    print(row)

print("\n=== NOTE: Hybrid/Vanilla/RAG data only covers Group A (5 runs) ===")
print("    Symbolic covers all 34 scenarios (all 4 groups)")
print("\nSymbolic partial acc by group:")
for grp, label in GROUPS.items():
    v = breakdown.get(grp, {}).get("symbolic", {}).get("partial")
    n = breakdown.get(grp, {}).get("symbolic", {}).get("cite_integrity")
    pl = breakdown.get(grp, {}).get("symbolic", {}).get("plurality")
    print(f"  {label}: partial={pct(v)}  cite_integrity={pct(n)}  pluralism={pct(pl)}")
