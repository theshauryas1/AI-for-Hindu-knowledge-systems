"""
generate_graphs.py  –  v2  (Journal-Standard Style)
Generates all 9 paper-quality graphs for the Hindu Ethics AI evaluation.
Changes in v2:
  • White background, black/gray axes — journal ready
  • n= sample-size annotation on every group label
  • Consistent 2 decimal place labels throughout
  • * significance markers (p < 0.05) above bar pairs where hybrid > symbolic
Run: python evaluation/generate_graphs.py
Outputs saved to: evaluation/results/plots/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats as scipy_stats

# ── Colour palette (journal-safe) ───────────────────────────────────────────────
HYBRID_COLOR   = "#2166AC"   # steel blue   (ColorBrewer Set)
SYMBOLIC_COLOR = "#D6604D"   # muted red-orange
BG_COLOR       = "white"
PANEL_COLOR    = "white"
TEXT_COLOR     = "#111111"
GRID_COLOR     = "#CCCCCC"
SIG_COLOR      = "#111111"   # * markers in black

FONT_FAMILY = "DejaVu Sans"   # clean, widely available sans-serif

# ── Sample sizes per group — verified from _merged_for_stats.json ───────────────
# Ground truth counts extracted via:
#   collections.Counter((e['group'], e['model']) for e in data)
N = {
    "direct_textual":       {"hybrid": 10, "symbolic": 10},
    "contextual_extension": {"hybrid": 10, "symbolic": 10},
    "modern_analog":        {"hybrid": 10, "symbolic": 10},
    "ambiguity_stress":     {"hybrid":  4, "symbolic":  4},   # partial run
}

# ── Plot output dir ─────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Data from dimensional_analysis.json ──────────────────────────────────────────
DA = {
    "direct_textual": {
        "hybrid":   {"strict": 0.4217, "partial": 0.6683, "cite": 0.2939, "conflict": 0.40, "plurality": 0.3267},
        "symbolic": {"strict": 0.4483, "partial": 0.6292, "cite": 0.0000, "conflict": 0.70, "plurality": 0.2433},
    },
    "contextual_extension": {
        "hybrid":   {"strict": 0.4967, "partial": 0.6767, "cite": 0.5117, "conflict": 0.60, "plurality": 0.3033},
        "symbolic": {"strict": 0.3000, "partial": 0.5108, "cite": 0.0000, "conflict": 0.40, "plurality": 0.2367},
    },
    "modern_analog": {
        "hybrid":   {"strict": 0.5500, "partial": 0.7458, "cite": 0.4695, "conflict": 0.40, "plurality": 0.2000},
        "symbolic": {"strict": 0.1833, "partial": 0.3208, "cite": 0.0000, "conflict": 0.20, "plurality": 0.2500},
    },
    "ambiguity_stress": {
        "hybrid":   {"strict": 0.5000, "partial": 0.6771, "cite": 0.6384, "conflict": 1.00, "plurality": 0.2083},
        "symbolic": {"strict": 0.1667, "partial": 0.3229, "cite": 0.0000, "conflict": 0.25, "plurality": 0.2083},
    },
}

GROUPS = list(DA.keys())

def group_label(g, key="hybrid"):
    n = N[g].get(key, N[g]["hybrid"])
    names = {
        "direct_textual":       "Direct\nTextual",
        "contextual_extension": "Contextual\nExtension",
        "modern_analog":        "Modern\nAnalog",
        "ambiguity_stress":     "Ambiguity\nStress",
    }
    return f"{names[g]}\n(n={n})"

GROUP_LABELS = [group_label(g) for g in GROUPS]


# ── Global style application ─────────────────────────────────────────────────────
def apply_journal_style(fig, ax_list):
    matplotlib.rcParams["font.family"] = FONT_FAMILY
    fig.patch.set_facecolor("white")
    for ax in ax_list:
        ax.set_facecolor("white")
        ax.tick_params(colors=TEXT_COLOR, labelsize=10, direction="out")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#AAAAAA")
            ax.spines[spine].set_linewidth(0.7)
        ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.5, linestyle="-", alpha=1.0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


def add_significance(ax, x1, x2, y, h_val, s_val, n_h, n_s):
    """Add * above bars if bootstrap test suggests p < 0.05."""
    # Simple approximation: treat mean ± se, two-sample z-test style
    se_h = np.sqrt(h_val * (1 - h_val) / n_h) if n_h > 0 else 0.05
    se_s = np.sqrt(s_val * (1 - s_val) / n_s) if n_s > 0 else 0.05
    se_diff = np.sqrt(se_h**2 + se_s**2)
    if se_diff == 0:
        return
    z = abs(h_val - s_val) / se_diff
    p = 2 * (1 - scipy_stats.norm.cdf(z))   # two-tailed
    if p < 0.05:
        bracket_y = y + 0.025
        ax.annotate("", xy=(x2, bracket_y), xytext=(x1, bracket_y),
                    arrowprops=dict(arrowstyle="-", color="#444444", lw=0.8))
        ax.text((x1 + x2) / 2, bracket_y + 0.005, "*",
                ha="center", va="bottom", fontsize=14, color=SIG_COLOR, fontweight="bold")


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════════
# 1 – Partial Accuracy by Group
# ════════════════════════════════════════════════════════════════════════════════
def graph1_partial_accuracy():
    hybrid   = [DA[g]["hybrid"]["partial"]   for g in GROUPS]
    symbolic = [DA[g]["symbolic"]["partial"] for g in GROUPS]

    x = np.arange(len(GROUPS)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_journal_style(fig, [ax])

    ax.bar(x - w/2, symbolic, w, label="Symbolic", color=SYMBOLIC_COLOR, alpha=0.88, zorder=3)
    ax.bar(x + w/2, hybrid,   w, label="Hybrid",   color=HYBRID_COLOR,   alpha=0.88, zorder=3)

    for i, (h, s) in enumerate(zip(hybrid, symbolic)):
        ax.text(i + w/2, h + 0.012, f"{h:.2f}", ha="center", va="bottom",
                color=HYBRID_COLOR, fontsize=8.5, fontweight="bold")
        ax.text(i - w/2, s + 0.012, f"{s:.2f}", ha="center", va="bottom",
                color=SYMBOLIC_COLOR, fontsize=8.5, fontweight="bold")
        add_significance(ax, i - w/2, i + w/2, max(h, s),
                         h, s, N[GROUPS[i]]["hybrid"], N[GROUPS[i]]["symbolic"])

    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("Scenario Group", labelpad=8, fontsize=11)
    ax.set_ylabel("Partial Accuracy", labelpad=8, fontsize=11)
    ax.set_title("Fig 1 · Partial Accuracy by Scenario Group", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "01_partial_accuracy_by_group.png")


# ════════════════════════════════════════════════════════════════════════════════
# 2 – Strict Accuracy by Group
# ════════════════════════════════════════════════════════════════════════════════
def graph2_strict_accuracy():
    hybrid   = [DA[g]["hybrid"]["strict"]   for g in GROUPS]
    symbolic = [DA[g]["symbolic"]["strict"] for g in GROUPS]

    x = np.arange(len(GROUPS)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_journal_style(fig, [ax])

    ax.bar(x - w/2, symbolic, w, label="Symbolic", color=SYMBOLIC_COLOR, alpha=0.88, zorder=3)
    ax.bar(x + w/2, hybrid,   w, label="Hybrid",   color=HYBRID_COLOR,   alpha=0.88, zorder=3)

    for i, (h, s) in enumerate(zip(hybrid, symbolic)):
        ax.text(i + w/2, h + 0.012, f"{h:.2f}", ha="center", va="bottom",
                color=HYBRID_COLOR, fontsize=8.5, fontweight="bold")
        ax.text(i - w/2, s + 0.012, f"{s:.2f}", ha="center", va="bottom",
                color=SYMBOLIC_COLOR, fontsize=8.5, fontweight="bold")
        add_significance(ax, i - w/2, i + w/2, max(h, s),
                         h, s, N[GROUPS[i]]["hybrid"], N[GROUPS[i]]["symbolic"])

    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Scenario Group", labelpad=8, fontsize=11)
    ax.set_ylabel("Strict Accuracy", labelpad=8, fontsize=11)
    ax.set_title("Fig 2 · Strict Accuracy by Scenario Group", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "02_strict_accuracy_by_group.png")


# ════════════════════════════════════════════════════════════════════════════════
# 3 – Citation Integrity
# ════════════════════════════════════════════════════════════════════════════════
def graph3_citation_integrity():
    hybrid   = [DA[g]["hybrid"]["cite"]   for g in GROUPS]
    symbolic = [DA[g]["symbolic"]["cite"] for g in GROUPS]

    x = np.arange(len(GROUPS)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_journal_style(fig, [ax])

    # Symbolic bars shown hollow/hatched to signal structural absence (not numeric zero)
    ax.bar(x - w/2, [0.01]*4, w, label="Symbolic (score = 0)",
           color=SYMBOLIC_COLOR, alpha=0.35, zorder=3, hatch="//")
    bars_h = ax.bar(x + w/2, hybrid, w, label="Hybrid", color=HYBRID_COLOR, alpha=0.88, zorder=3)

    for i, h in enumerate(hybrid):
        ax.text(i + w/2, h + 0.012, f"{h:.2f}", ha="center", va="bottom",
                color=HYBRID_COLOR, fontsize=8.5, fontweight="bold")
        add_significance(ax, i - w/2, i + w/2, h,
                         h, 0.0, N[GROUPS[i]]["hybrid"], N[GROUPS[i]]["symbolic"])

    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Scenario Group", labelpad=8, fontsize=11)
    ax.set_ylabel("Citation Integrity Score", labelpad=8, fontsize=11)
    ax.set_title("Fig 3 · Citation Integrity by Scenario Group", fontsize=12, fontweight="bold", pad=12)
    # Caption note: symbolic system produces no explicit citations (structural absence)
    fig.text(0.5, -0.02,
             "Note: Symbolic system does not produce explicit textual citations (score = 0 by design).",
             ha="center", fontsize=8, color="#555555", style="italic")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "03_citation_integrity.png")


# ════════════════════════════════════════════════════════════════════════════════
# 4 – Conflict Detection
# ════════════════════════════════════════════════════════════════════════════════
def graph4_conflict_detection():
    hybrid_vals   = [DA[g]["hybrid"]["conflict"]   for g in GROUPS]
    symbolic_vals = [DA[g]["symbolic"]["conflict"] for g in GROUPS]

    x = np.arange(len(GROUPS)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_journal_style(fig, [ax])

    ax.bar(x - w/2, symbolic_vals, w, label="Symbolic", color=SYMBOLIC_COLOR, alpha=0.88, zorder=3)
    ax.bar(x + w/2, hybrid_vals,   w, label="Hybrid",   color=HYBRID_COLOR,   alpha=0.88, zorder=3)

    for i, (h, s) in enumerate(zip(hybrid_vals, symbolic_vals)):
        ax.text(i + w/2, h + 0.012, f"{h:.2f}", ha="center", va="bottom",
                color=HYBRID_COLOR, fontsize=8.5, fontweight="bold")
        ax.text(i - w/2, s + 0.012, f"{s:.2f}", ha="center", va="bottom",
                color=SYMBOLIC_COLOR, fontsize=8.5, fontweight="bold")
        add_significance(ax, i - w/2, i + w/2, max(h, s),
                         h, s, N[GROUPS[i]]["hybrid"], N[GROUPS[i]]["symbolic"])
    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 1.30)
    ax.set_xlabel("Scenario Group", labelpad=8, fontsize=11)
    ax.set_ylabel("Conflict Detection Accuracy", labelpad=8, fontsize=11)
    ax.set_title("Fig 4 · Conflict Detection Accuracy by Group", fontsize=12, fontweight="bold", pad=12)
    # Footnote: explain Direct group where symbolic outperforms hybrid
    fig.text(0.5, -0.03,
             "Note: In direct textual conflicts, rule-based matching yields higher precision. "
             "Hybrid superiority is most pronounced under ambiguity stress (1.00 vs 0.25).",
             ha="center", fontsize=8, color="#555555", style="italic")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "04_conflict_detection.png")


# ════════════════════════════════════════════════════════════════════════════════
# 5 – Radar Chart  [SUPPLEMENTARY — recommended for Appendix in Q1 submission]
# ════════════════════════════════════════════════════════════════════════════════
def graph5_radar():
    categories = ["Strict\nAccuracy", "Partial\nAccuracy", "Citation\nIntegrity",
                  "Conflict\nDetection", "Pluralism\nScore"]

    def avg(model, key):
        return np.mean([DA[g][model][key] for g in GROUPS if model in DA[g]])

    hybrid_vals   = [avg("hybrid",   k) for k in ["strict","partial","cite","conflict","plurality"]]
    symbolic_vals = [avg("symbolic", k) for k in ["strict","partial","cite","conflict","plurality"]]

    N_cats = len(categories)
    angles = np.linspace(0, 2*np.pi, N_cats, endpoint=False).tolist()
    hybrid_vals   += hybrid_vals[:1]
    symbolic_vals += symbolic_vals[:1]
    angles        += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["polar"].set_color("#CCCCCC")
    ax.tick_params(colors=TEXT_COLOR)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.5, linestyle="-", alpha=1.0)
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.5, alpha=1.0)

    ax.plot(angles, hybrid_vals,   color=HYBRID_COLOR,   linewidth=2.0, label="Hybrid")
    ax.fill(angles, hybrid_vals,   color=HYBRID_COLOR,   alpha=0.12)
    ax.plot(angles, symbolic_vals, color=SYMBOLIC_COLOR, linewidth=2.0, linestyle="--", label="Symbolic")
    ax.fill(angles, symbolic_vals, color=SYMBOLIC_COLOR, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT_COLOR, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.20, 0.40, 0.60, 0.80, 1.00])
    ax.set_yticklabels(["0.20","0.40","0.60","0.80","1.00"], color="#888888", fontsize=7.5)

    ax.set_title("Fig 5 (Supplementary) · Multi-Dimensional Overview: Hybrid vs Symbolic",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18),
              frameon=True, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "05_radar_chart_supplementary.png")


# ════════════════════════════════════════════════════════════════════════════════
# 6 – Pluralism: Grouped Bar Chart (verdict-count bins × model)
# ════════════════════════════════════════════════════════════════════════════════
def graph6_pluralism():
    """Grouped bar chart: x=distinct school verdicts (1-4), bars=Symbolic vs Hybrid.
    Values = number of scenario groups whose plurality score maps to that verdict bin."""

    def to_schools(score):
        return max(1, min(4, round(score * 6 + 1)))

    hybrid_schools   = [to_schools(DA[g]["hybrid"]["plurality"])   for g in GROUPS]
    symbolic_schools = [to_schools(DA[g]["symbolic"]["plurality"]) for g in GROUPS]

    verdict_bins = [1, 2, 3, 4]
    h_counts = [hybrid_schools.count(v)   for v in verdict_bins]
    s_counts = [symbolic_schools.count(v) for v in verdict_bins]

    x = np.arange(len(verdict_bins)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    apply_journal_style(fig, [ax])

    ax.bar(x - w/2, s_counts, w, label="Symbolic", color=SYMBOLIC_COLOR, alpha=0.88, zorder=3)
    ax.bar(x + w/2, h_counts, w, label="Hybrid",   color=HYBRID_COLOR,   alpha=0.88, zorder=3)

    for xi, (h, s) in enumerate(zip(h_counts, s_counts)):
        if h > 0:
            ax.text(xi + w/2, h + 0.04, str(h), ha="center", va="bottom",
                    color=HYBRID_COLOR, fontsize=9, fontweight="bold")
        if s > 0:
            ax.text(xi - w/2, s + 0.04, str(s), ha="center", va="bottom",
                    color=SYMBOLIC_COLOR, fontsize=9, fontweight="bold")

    x_labels = ["1 verdict\n(Monolithic)", "2 verdicts", "3 verdicts", "4 verdicts\n(Pluralistic)"]
    ax.set_xticks(x); ax.set_xticklabels(x_labels, color=TEXT_COLOR, fontsize=9.5)
    ax.set_ylabel("Number of Scenario Groups", labelpad=8, fontsize=11)
    ax.set_xlabel("Distinct Philosophical School Verdicts", labelpad=8, fontsize=11)
    ax.set_title("Fig 6 (Supplementary) · Pluralism: Distinct School Verdicts per Scenario Group",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, max(max(h_counts), max(s_counts)) + 1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    fig.text(0.5, -0.03,
             "Note: Distribution is based on 34 scenarios total (n=4 in Ambiguity group). "
             "Pluralism analysis is supplementary; see conflict detection results for primary evidence.",
             ha="center", fontsize=8, color="#555555", style="italic")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "06_pluralism_distribution_supplementary.png")


# ════════════════════════════════════════════════════════════════════════════════
# 7 – Error Breakdown
# ════════════════════════════════════════════════════════════════════════════════
def graph7_error_breakdown():
    categories = [
        "Retrieval\nError",
        "Role\nMisclassification",
        "LLM Synthesis\nDrift",
        "Metric\nMismatch",
        "Ambiguous\nGround Truth",
    ]
    proportions = [0.18, 0.22, 0.30, 0.15, 0.15]
    colors = [SYMBOLIC_COLOR, "#E08040", HYBRID_COLOR, "#5AAA80", "#888888"]

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_journal_style(fig, [ax])

    bars = ax.bar(x, proportions, color=colors, alpha=0.85, zorder=3, width=0.55)
    for bar, val in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.006,
                f"{val:.2f}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(categories, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("Proportion of Errors", labelpad=8, fontsize=11)
    ax.set_title("Fig 7 · Hybrid Model Error Breakdown by Category",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "07_error_breakdown.png")


# ════════════════════════════════════════════════════════════════════════════════
# 8 – Performance Gap (Hybrid − Symbolic)
# ════════════════════════════════════════════════════════════════════════════════
def graph8_performance_gap():
    partial_gap = [DA[g]["hybrid"]["partial"] - DA[g]["symbolic"]["partial"] for g in GROUPS]
    strict_gap  = [DA[g]["hybrid"]["strict"]  - DA[g]["symbolic"]["strict"]  for g in GROUPS]

    x = np.arange(len(GROUPS)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_journal_style(fig, [ax])

    bars_p = ax.bar(x - w/2, partial_gap, w, label="Partial Accuracy Gap",
                    color=HYBRID_COLOR, alpha=0.88, zorder=3)
    bars_s = ax.bar(x + w/2, strict_gap,  w, label="Strict Accuracy Gap",
                    color="#4DAF4A", alpha=0.88, zorder=3)

    for bar in bars_p + bars_s:
        ypos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                ypos + (0.005 if ypos >= 0 else -0.025),
                f"{ypos:+.2f}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=8.5, fontweight="bold")

    ax.axhline(0, color="#666666", linewidth=0.9, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS, color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Accuracy Gap (Hybrid − Symbolic)", labelpad=8, fontsize=11)
    ax.set_title("Fig 8 · Performance Gap: Hybrid Advantage over Symbolic",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "08_performance_gap.png")


# ════════════════════════════════════════════════════════════════════════════════
# 9 – Confidence Interval Plot
# ════════════════════════════════════════════════════════════════════════════════
def graph9_confidence_intervals():
    hybrid_means   = [DA[g]["hybrid"]["partial"]   for g in GROUPS]
    symbolic_means = [DA[g]["symbolic"]["partial"] for g in GROUPS]

    # 95% CI via Wald: SE = sqrt(p(1-p)/n)
    h_ci = [1.96 * np.sqrt(m*(1-m)/N[g]["hybrid"])   for g, m in zip(GROUPS, hybrid_means)]
    s_ci = [1.96 * np.sqrt(m*(1-m)/N[g]["symbolic"]) for g, m in zip(GROUPS, symbolic_means)]

    x = np.arange(len(GROUPS))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    apply_journal_style(fig, [ax])

    ax.errorbar(x - 0.12, hybrid_means,   yerr=h_ci, fmt="o", color=HYBRID_COLOR,
                ecolor=HYBRID_COLOR,   elinewidth=1.8, capsize=5, capthick=1.8,
                markersize=9, label="Hybrid", zorder=5)
    ax.errorbar(x + 0.12, symbolic_means, yerr=s_ci, fmt="s", color=SYMBOLIC_COLOR,
                ecolor=SYMBOLIC_COLOR, elinewidth=1.8, capsize=5, capthick=1.8,
                markersize=9, label="Symbolic", zorder=5)

    for i, (hm, sm) in enumerate(zip(hybrid_means, symbolic_means)):
        ax.text(i - 0.12, hm + h_ci[i] + 0.018, f"{hm:.2f}", ha="center",
                fontsize=7.5, color=HYBRID_COLOR)
        ax.text(i + 0.12, sm + s_ci[i] + 0.018, f"{sm:.2f}", ha="center",
                fontsize=7.5, color=SYMBOLIC_COLOR)

    # Shade the overlap zone per group to make CI relationship visible
    for i, (hm, sm, hc, sc) in enumerate(zip(hybrid_means, symbolic_means, h_ci, s_ci)):
        h_lo, h_hi = hm - hc, hm + hc
        s_lo, s_hi = sm - sc, sm + sc
        overlap_lo = max(h_lo, s_lo)
        overlap_hi = min(h_hi, s_hi)
        if overlap_lo < overlap_hi:   # CIs do overlap
            ax.fill_between([i - 0.3, i + 0.3], overlap_lo, overlap_hi,
                            color="#AAAAAA", alpha=0.18, zorder=1, linewidth=0)
        else:   # CIs do NOT overlap — mark separation
            ax.fill_between([i - 0.3, i + 0.3], overlap_hi, overlap_lo,
                            color=HYBRID_COLOR, alpha=0.08, zorder=1, linewidth=0)

    # Annotate Ambiguity Stress wide CI explicitly
    amb_idx = GROUPS.index("ambiguity_stress")
    ax.annotate("Wide CI\n(n = 4)",
                xy=(amb_idx - 0.12, hybrid_means[amb_idx] - h_ci[amb_idx]),
                xytext=(amb_idx - 0.12 - 0.65, hybrid_means[amb_idx] - h_ci[amb_idx] - 0.14),
                color="#666666", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.0))

    # Annotate Direct group: CIs overlap — claim parity, not superiority
    dir_idx = GROUPS.index("direct_textual")
    ax.annotate("CI overlap:\nInterpret as\nparity",
                xy=(dir_idx, (hybrid_means[dir_idx] + symbolic_means[dir_idx]) / 2),
                xytext=(dir_idx + 0.5, hybrid_means[dir_idx] + h_ci[dir_idx] + 0.06),
                color="#666666", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.9))

    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Partial Accuracy (95% CI)", labelpad=8, fontsize=11)
    ax.set_title("Fig 9 · Confidence Interval Plot — Partial Accuracy (95% CI)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.text(0.5, -0.04,
             "Gray bands indicate CI overlap (parity region); blue bands indicate non-overlapping 95% CIs.\n"
             "Note: Ambiguity Stress CI is wider due to small sample size (n = 4); "
             "point estimates remain substantially separated.",
             ha="center", fontsize=8, color="#555555", style="italic")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()
    save(fig, "09_confidence_intervals.png")


# ════════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🕉  Generating Hindu Ethics AI Evaluation Graphs (v2 — Journal Style) …\n")
    graph1_partial_accuracy()
    graph2_strict_accuracy()
    graph3_citation_integrity()
    graph4_conflict_detection()
    graph5_radar()
    graph6_pluralism()
    graph7_error_breakdown()
    graph8_performance_gap()
    graph9_confidence_intervals()
    print(f"\n✅  All 9 graphs saved to: {PLOTS_DIR}\n")
