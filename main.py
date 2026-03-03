"""
main.py — HinduMind CLI Entry Point

Usage:
  python main.py query  --text "What is the nature of atman in Advaita?"
  python main.py dharma --scenario "Is it dharmic for a kshatriya to refuse battle?"
  python main.py eval   --module kg
  python main.py eval   --module cases
  python main.py eval   --module kappa --csv path/to/csv
"""

import json
import os
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from kg.hpo_graph import HPOGraph
from kg.populate_kg import populate
from agents.school_classifier import SchoolClassifier
from agents.school_agents import build_agents
from agents.meta_agent import MetaAgent

console = Console()
app = typer.Typer(
    name="hinduмind",
    help="🕉️  HinduMind — Knowledge Graph-Driven Hindu Philosophical Reasoning",
    add_completion=False
)

# ── Cached KG (loaded once per session) ─────────────────────────
_KG: HPOGraph | None = None

def get_kg() -> HPOGraph:
    global _KG
    if _KG is None:
        console.print("[bold yellow]⏳ Building HPO Knowledge Graph...[/bold yellow]")
        _KG = HPOGraph()
        populate(_KG)
        stats = _KG.stats()
        console.print(
            f"[green]✓ KG ready:[/green] {stats['total_nodes']} nodes, "
            f"{stats['total_edges']} edges, {stats['rdf_triples']} RDF triples"
        )
    return _KG


# ─────────────────────────────────────────────────────────────────
# query command
# ─────────────────────────────────────────────────────────────────

@app.command()
def query(
    text: str = typer.Option(..., "--text", "-t", help="Philosophical query text"),
    schools: str = typer.Option("advaita,dvaita,nyaya", "--schools", "-s",
                                help="Comma-separated school IDs"),
    backend: str = typer.Option(None, "--backend", "-b",
                                help="LLM backend: ollama/openai/huggingface/mock"),
    output: str = typer.Option(None, "--output", "-o", help="Save JSON output to file"),
):
    """
    Run a philosophical query through the multi-agent reasoning system.

    Example:
      python main.py query --text "What is the nature of atman in Advaita?"
    """
    school_list = [s.strip() for s in schools.split(",")]
    kg = get_kg()

    # 1. Classify
    clf = SchoolClassifier(use_transformer=False)
    clf_result = clf.classify(text)
    console.print(
        f"\n[bold cyan]🔍 School Classifier[/bold cyan]  "
        f"→ [bold]{clf_result['school']}[/bold] "
        f"(confidence: {clf_result['confidence']:.2f})"
    )
    if clf_result["concepts"]:
        console.print(f"   Concepts detected: {', '.join(clf_result['concepts'])}")

    # 2. Build agents
    console.print(f"\n[bold cyan]🤖 Running agents:[/bold cyan] {', '.join(school_list)}")
    agents = build_agents(kg, school_list, backend)
    school_responses = {}
    for school, agent in agents.items():
        console.print(f"   → {school}...", end=" ")
        resp = agent.reason(text, concepts=clf_result["concepts"])
        school_responses[school] = resp
        console.print("[green]✓[/green]")

    # 3. Meta-agent synthesis
    console.print("\n[bold cyan]⚡ Meta-Agent synthesizing...[/bold cyan]", end=" ")
    meta = MetaAgent(kg)
    concept = clf_result["concepts"][0] if clf_result["concepts"] else "dharma"
    response = meta.synthesize(text, concept, school_responses)
    console.print("[green]✓[/green]")

    # 4. Display
    _display_response(response, text)

    # 5. Save
    if output:
        Path(output).write_text(response.to_json(), encoding="utf-8")
        console.print(f"\n[green]✓ Saved → {output}[/green]")


def _display_response(response, query_text: str):
    console.print()
    console.print(Panel(
        f"[bold white]{query_text}[/bold white]",
        title="[bold blue]🕉️  HinduMind Query[/bold blue]",
        border_style="blue"
    ))

    # School responses table
    table = Table(title="Per-School Responses", border_style="cyan", show_lines=True)
    table.add_column("School", style="bold yellow", width=16)
    table.add_column("Response", style="white")
    for school, resp_text in response.responses.items():
        preview = resp_text[:280] + ("..." if len(resp_text) > 280 else "")
        table.add_row(school.upper(), preview)
    console.print(table)

    # Agreements
    if response.agreements:
        console.print(Panel(
            "\n".join(f"• {a}" for a in response.agreements),
            title="[green]✅ Agreements[/green]",
            border_style="green"
        ))

    # Contradictions
    if response.contradictions:
        lines = []
        for c in response.contradictions:
            lines.append(f"• {c.get('school_a','')} ↔ {c.get('school_b','')}: "
                         f"{c.get('on', c.get('note',''))}")
        console.print(Panel(
            "\n".join(lines),
            title="[red]⚡ Contradictions[/red]",
            border_style="red"
        ))

    # Synthesis
    console.print(Panel(
        response.synthesis,
        title="[bold magenta]🔮 Synthesis[/bold magenta]",
        border_style="magenta"
    ))

    # KG Sources
    if response.kg_sources:
        console.print(Panel(
            "\n".join(f"• {s}" for s in response.kg_sources),
            title="[dim]📚 KG Sources[/dim]",
            border_style="dim"
        ))


# ─────────────────────────────────────────────────────────────────
# dharma command
# ─────────────────────────────────────────────────────────────────

@app.command()
def dharma(
    scenario: str = typer.Option(..., "--scenario", help="Ethical scenario to analyze"),
    schools:  str = typer.Option("advaita,dvaita,nyaya", "--schools"),
    backend:  str = typer.Option(None, "--backend"),
    output:   str = typer.Option(None, "--output"),
):
    """
    Run an ethical scenario through the Computational Dharma Engine.

    Example:
      python main.py dharma --scenario "Is it dharmic for a kshatriya to refuse battle?"
    """
    from dharma_engine.context_parser   import ContextParser
    from dharma_engine.deontic_reasoner import DeonticReasoner

    school_list = [s.strip() for s in schools.split(",")]
    kg = get_kg()

    console.print(f"\n[bold cyan]📜 Parsing scenario context...[/bold cyan]")
    ctx = ContextParser().parse(scenario)
    console.print(f"   Role: [bold]{ctx.agent_role}[/bold] | "
                  f"Āśrama: [bold]{ctx.ashrama}[/bold] | "
                  f"Action: [bold]{ctx.action}[/bold]")
    console.print(f"   Situation: {ctx.situation} | Override: {ctx.contextual_override}")

    console.print(f"\n[bold cyan]⚖️  Running Deontic Reasoner...[/bold cyan]")
    verdict = DeonticReasoner().evaluate_scenario(ctx, school_list)
    console.print(f"   Matched [bold]{len(verdict.applicable_rules)}[/bold] rule(s)")

    # Display deontic verdict
    table = Table(title="School-Weighted Deontic Verdicts", border_style="yellow")
    table.add_column("School", style="bold yellow", width=20)
    table.add_column("Verdict", style="white")
    for school, v in verdict.school_verdicts.items():
        color = "green" if "Obligatory" in v else ("red" if "Forbidden" in v else "white")
        table.add_row(school.upper(), f"[{color}]{v}[/{color}]")
    console.print(table)

    console.print(Panel(
        verdict.overall_verdict,
        title="[bold yellow]Overall Verdict[/bold yellow]",
        border_style="yellow"
    ))

    # Run school agents for qualitative opinion
    console.print(f"\n[bold cyan]🤖 Running school agents for qualitative analysis...[/bold cyan]")
    agents = build_agents(kg, school_list, backend)
    school_responses = {}
    for school, agent in agents.items():
        resp = agent.reason(
            query=f"Is it dharmic to {ctx.action.replace('_',' ')}? Scenario: {scenario}",
            concepts=[ctx.action, ctx.agent_role]
        )
        school_responses[school] = resp

    meta = MetaAgent(kg)
    kg_sources = [j["source"] for j in verdict.judgments[:3] if j.get("source")]
    response = meta.synthesize(
        query=scenario,
        concept=ctx.action,
        school_responses=school_responses,
        kg_sources=kg_sources
    )

    _display_response(response, scenario)

    if output:
        result = {"deontic_verdict": verdict.to_dict(), "multi_agent": response.to_dict()}
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[green]✓ Saved → {output}[/green]")


# ─────────────────────────────────────────────────────────────────
# eval command
# ─────────────────────────────────────────────────────────────────

@app.command()
def eval(
    module: str = typer.Option("kg", "--module", "-m",
                                help="Evaluation module: kg / cases / kappa"),
    csv:    str = typer.Option(None, "--csv", help="Path to annotated CSV (for kappa)"),
    backend: str = typer.Option(None, "--backend"),
):
    """
    Run evaluation modules.

    Modules:
      kg     — KG quality evaluation (coverage stats + spot-check CSV)
      cases  — Generate 3 worked case studies
      kappa  — Compute Cohen's κ from annotated spot-check CSV
    """
    if module == "kg":
        from evaluation.kg_evaluator import run_full_evaluation
        kg = get_kg()
        run_full_evaluation(kg)

    elif module == "cases":
        from evaluation.case_studies import run_all
        run_all(schools=["advaita", "dvaita", "nyaya"], llm_backend=backend)

    elif module == "kappa":
        from evaluation.kg_evaluator import compute_cohens_kappa
        k = compute_cohens_kappa(csv)
        console.print(f"\n[bold]Cohen's κ = {k:.4f}[/bold]")

    else:
        console.print(f"[red]Unknown module '{module}'. Use: kg / cases / kappa[/red]")


# ─────────────────────────────────────────────────────────────────
# kg command (populate + export)
# ─────────────────────────────────────────────────────────────────

@app.command(name="build-kg")
def build_kg_cmd(
    export_rdf:  bool = typer.Option(True, "--rdf/--no-rdf"),
    export_ml:   bool = typer.Option(True, "--graphml/--no-graphml"),
    export_json: bool = typer.Option(True, "--json/--no-json"),
):
    """Build and export the HPO Knowledge Graph."""
    from kg.populate_kg import main as populate_main
    populate_main()


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel(
        "[bold white]HinduMind[/bold white] 🕉️\n"
        "[dim]Knowledge Graph-Driven Multi-Agent System for Hindu Philosophical Reasoning[/dim]",
        border_style="blue"
    ))
    app()
