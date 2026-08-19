"""
Module 4 — Dashboard Layer
Implements: DashboardApp (as a Streamlit script, per the SDD's deployment
design — a single-process Streamlit app, no separate application server).

Unlike v1, this version calls Module 1/2/3's pipelines directly rather than
always going through PipelineRunner.run_full_pipeline() as one black box.
That's a deliberate choice, not a step backwards from the Facade pattern:
PipelineRunner still exists (pipeline_runner.py) for programmatic/batch use
where "always run everything" is the right default. The dashboard, though,
needs two things PipelineRunner's single entry point doesn't give it:
  1. Independent on/off toggles for Module 2 (druggability) and Module 3
     (CRISPR safety) — a user may only care about one pathway.
  2. Fine-grained progress narration per stage, via st.status(), so the
     user can actually see what's happening instead of staring at a single
     spinner for up to ~2 minutes (NFR-11).
Module 2 and 3 still run concurrently via ThreadPoolExecutor when both are
selected, same as PipelineRunner does internally.

Run with:
    streamlit run app.py
"""

import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

import sys


import sys
import os
import shutil
import streamlit as st

st.sidebar.write("Python:", sys.executable)
st.sidebar.write("fpocket:", shutil.which("fpocket"))

from data_ingestor import DataIngestor, MockGeneProvider, OpenTargetsProvider, FallbackGeneProvider
from network_analyzer import NetworkAnalyzer
from target_discovery import TargetDiscoveryPipeline

from structural_ml import PocketDetector, FeatureExtractor, DruggabilityEngine, DruggabilityPipeline
from crispr_safety import SequenceFetcher, OffTargetScanner, CrisprSafetyEngine, CrisprSafetyPipeline

from pipeline_runner import fetch_best_structure, generate_verdict
from db import Database


st.set_page_config(page_title="BioMedix-AI", page_icon="🧬", layout="wide")

# A little custom styling — Streamlit's defaults are fine but flat; this
# just gives the verdict banner and metric cards a bit more visual weight.
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .bmx-verdict {
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .bmx-verdict.strong   { background: rgba(34,197,94,0.15);  border-left: 6px solid #22c55e; }
    .bmx-verdict.mixed    { background: rgba(234,179,8,0.15);  border-left: 6px solid #eab308; }
    .bmx-verdict.weak     { background: rgba(239,68,68,0.15);  border-left: 6px solid #ef4444; }
    .bmx-verdict.partial  { background: rgba(59,130,246,0.15); border-left: 6px solid #3b82f6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Wiring — cached so Streamlit's rerun-on-every-interaction model doesn't
# reopen a DB connection or reload the ML models on every button click.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db():
    try:
        return Database()
    except Exception as e:
        st.sidebar.warning(
            f"No PostgreSQL connection ({e}). Running without persistence — "
            f"results won't be saved or downloadable from History.",
            icon="⚠️",
        )
        return None


@st.cache_resource
def get_components(_db):
    """Builds the individual sub-pipelines once and reuses them across
    reruns/button clicks. Returns a plain dict rather than a PipelineRunner
    so app.py can call Module 2/3 independently (see module docstring)."""
    gene_provider = FallbackGeneProvider(primary=OpenTargetsProvider(), fallback=MockGeneProvider())
    ingestor = DataIngestor(gene_provider=gene_provider)
    target_pipeline = TargetDiscoveryPipeline(
        ingestor=ingestor, analyzer=NetworkAnalyzer(), db=_db
    )

    detector = PocketDetector() if shutil.which("fpocket") else None
    st.sidebar.write("fpocket location:", shutil.which("fpocket"))
    model_path = os.environ.get("DRUGGABILITY_MODEL_PATH", "druggability_model.joblib")
    engine = DruggabilityEngine(model_path=model_path if os.path.exists(model_path) else None)
    druggability_pipeline = DruggabilityPipeline(
        detector=detector, extractor=FeatureExtractor(), engine=engine, db=_db
    )

    crispr_pipeline = CrisprSafetyPipeline(
        fetcher=SequenceFetcher(), scanner=OffTargetScanner(), engine=CrisprSafetyEngine(), db=_db
    )

    return {
        "target_pipeline": target_pipeline,
        "druggability_pipeline": druggability_pipeline,
        "crispr_pipeline": crispr_pipeline,
    }


# ---------------------------------------------------------------------------
# Network graph rendering
# ---------------------------------------------------------------------------
def render_network_figure(graph: nx.Graph, hub_gene: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No PPI edges found", ha="center", va="center")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(graph, seed=42, k=0.9)
    weights = [graph[u][v].get("weight", 1) for u, v in graph.edges()]
    max_w = max(weights) if weights else 1
    widths = [0.5 + 3 * (w / max_w) for w in weights]

    node_colors = ["#ef4444" if n == hub_gene else "#3b82f6" for n in graph.nodes()]
    node_sizes = [900 if n == hub_gene else 500 for n in graph.nodes()]

    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.4, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold", ax=ax)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report generation (Markdown, downloadable)
# ---------------------------------------------------------------------------
def build_markdown_report(hub_result, drug_result, crispr_result, verdict) -> str:
    lines = [
        f"# BioMedix-AI Report — {hub_result.disease_name}",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "> Research/academic prototype — not intended for clinical decision-making.",
        "",
        f"## Verdict\n{verdict}",
        "",
        f"## Hub Gene: `{hub_result.hub_gene_symbol}`",
        f"- Network size: {hub_result.graph_summary.get('num_nodes', 0)} nodes, "
        f"{hub_result.graph_summary.get('num_edges', 0)} edges",
        "",
        "### Top genes by centrality",
        "| Gene | Centrality |",
        "|---|---|",
    ]
    top5 = sorted(hub_result.centrality_scores.items(), key=lambda x: -x[1])[:5]
    lines += [f"| {g} | {s:.4f} |" for g, s in top5]

    lines.append("\n## Druggability")
    if drug_result is None:
        lines.append("_Not run for this analysis._")
    else:
        lines.append(f"- Score: **{drug_result.druggability_score:.2f}** (0-1 scale)")
        lines.append(f"- Fallback heuristic used: {drug_result.used_fallback}")
        if drug_result.pdb_id:
            lines.append(f"- Structure: {drug_result.pdb_id}")
        for k, v in (drug_result.pocket_features or {}).items():
            lines.append(f"  - {k}: {v}")

    lines.append("\n## CRISPR Safety")
    if crispr_result is None:
        lines.append("_Not run for this analysis._")
    else:
        lines.append(f"- Safety score: **{crispr_result.safety_score:.2f}** (0-1 scale, higher = safer)")
        lines.append(f"- Fallback/interim heuristic used: {crispr_result.used_fallback}")
        if crispr_result.flagged_sites:
            lines.append(f"\n### Flagged off-target sites ({len(crispr_result.flagged_sites)})")
            lines.append("| Position | Sequence | Mismatches | PAM OK | Risk |")
            lines.append("|---|---|---|---|---|")
            for s in crispr_result.flagged_sites:
                lines.append(
                    f"| {s.position} | {s.sequence} | {s.mismatches} | {s.pam_ok} | {s.risk_score:.3f} |"
                )
        else:
            lines.append("- No high-risk off-target sites flagged.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI — sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("🧬 BioMedix-AI")
    st.sidebar.caption("In silico drug discovery & target validation")
    page = st.sidebar.radio("", ["🚀 Run Analysis", "📜 History"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Pipeline stages**\n"
        "1. 🕸️ Find the disease's hub gene (PPI network)\n"
        "2. 💊 Score its druggability *(optional)*\n"
        "3. ✂️ Score a guide RNA's off-target risk *(optional)*\n"
        "4. 📋 Combine into a verdict"
    )
    return page


# ---------------------------------------------------------------------------
# UI — input form (SRS UC-1, UC-8, UC-12)
# ---------------------------------------------------------------------------
def render_input_form():
    st.title("Run a new analysis")
    st.caption(
        "Enter a disease, then choose which of the two downstream analyses "
        "to run — they're independent, so you can run either, both, or neither."
    )

    with st.form("pipeline_form"):
        disease_name = st.text_input(
            "Disease name", placeholder="e.g. Alzheimer's disease, rice cadmium toxicity"
        )

        st.markdown("**Which analyses should run on the hub gene?**")
        c1, c2 = st.columns(2)
        with c1:
            run_druggability = st.checkbox("💊 Druggability (Module 2)", value=True)
        with c2:
            run_crispr = st.checkbox("✂️ CRISPR safety (Module 3)", value=False)

        guide_rna = None
        if run_crispr:
            guide_rna = st.text_input(
                "Guide RNA sequence (required for CRISPR safety)",
                placeholder="20 nt, e.g. GATCGGATCCGTAGCTAGCT",
            )

        with st.expander("⚙️ Advanced settings (SRS UC-12)"):
            col1, col2 = st.columns(2)
            with col1:
                species = st.selectbox(
                    "Species", ["human", "rice", "mouse", "arabidopsis", "yeast", "zebrafish"]
                )
                centrality_method = st.selectbox(
                    "Centrality method", ["degree", "betweenness", "eigenvector", "closeness"]
                )
            with col2:
                gene_limit = st.slider("Gene limit", min_value=3, max_value=20, value=10)
                max_mismatches = st.slider(
                    "Max off-target mismatches", min_value=0, max_value=10, value=6
                )

        submitted = st.form_submit_button("Run pipeline", type="primary", use_container_width=True)

    if not submitted:
        return None

    if not disease_name.strip():
        st.error("Please enter a disease name.")
        return None
    if run_crispr and not (guide_rna and guide_rna.strip()):
        st.error("CRISPR safety is selected — please enter a guide RNA sequence.")
        return None
    if not run_druggability and not run_crispr:
        st.error("Select at least one analysis to run (druggability and/or CRISPR safety).")
        return None

    return {
        "disease_name": disease_name.strip(),
        "run_druggability": run_druggability,
        "run_crispr": run_crispr,
        "guide_rna": guide_rna.strip() if guide_rna else None,
        "species": species,
        "centrality_method": centrality_method,
        "gene_limit": gene_limit,
        "max_mismatches": max_mismatches,
    }


# ---------------------------------------------------------------------------
# Orchestration with live step-by-step narration
# ---------------------------------------------------------------------------
def run_analysis(components, db, inputs):
    target_pipeline = components["target_pipeline"]
    druggability_pipeline = components["druggability_pipeline"]
    crispr_pipeline = components["crispr_pipeline"]

    with st.status("Running BioMedix-AI pipeline…", expanded=True) as status:
        status.write("🕸️ Fetching disease-associated genes & building the PPI network…")
        hub_result = target_pipeline.run(
            inputs["disease_name"],
            gene_limit=inputs["gene_limit"],
            species=inputs["species"],
            centrality_method=inputs["centrality_method"],
        )
        hub_gene = hub_result.hub_gene_symbol
        status.write(f"✅ Hub gene identified: **{hub_gene}**")

        drug_result = None
        crispr_result = None

        structure_info = None
        if inputs["run_druggability"]:
            status.write("🔎 Looking up a 3D structure for the hub gene…")
            structure_info = fetch_best_structure(hub_gene, organism=(
                "Homo sapiens" if inputs["species"] == "human" else inputs["species"]
            ))
            if structure_info is None:
                status.write("⚠️ No structure found — will use the sequence-based fallback score.")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_drug = None
            future_crispr = None

            if inputs["run_druggability"]:
                status.write("💊 Scoring druggability…")
                future_drug = executor.submit(
                    druggability_pipeline.run,
                    hub_gene,
                    structure_path=structure_info["structure_path"] if structure_info else None,
                    pdb_id=structure_info["pdb_id"] if structure_info else None,
                )
            if inputs["run_crispr"]:
                status.write("✂️ Scoring CRISPR off-target safety…")
                future_crispr = executor.submit(
                    crispr_pipeline.run,
                    hub_gene,
                    inputs["guide_rna"],
                    species=inputs["species"],
                    max_mismatches=inputs["max_mismatches"],
                )

            if future_drug is not None:
                drug_result = future_drug.result()
            if future_crispr is not None:
                crispr_result = future_crispr.result()

        if structure_info and os.path.exists(structure_info["structure_path"]):
            try:
                os.remove(structure_info["structure_path"])
            except OSError:
                pass

        status.write("📋 Generating verdict…")
        verdict = generate_verdict(drug_result, crispr_result)

        report_id = None
        if db is not None and hub_result.run_id is not None:
            report_id = db.insert_integrated_report(
                run_id=hub_result.run_id,
                verdict_text=verdict,
                druggability_result_id=drug_result.result_id if drug_result else None,
                crispr_result_id=crispr_result.result_id if crispr_result else None,
            )
            status.write(f"💾 Saved as report #{report_id}")

        status.update(label="Done!", state="complete", expanded=False)

    return hub_result, drug_result, crispr_result, verdict, report_id


# ---------------------------------------------------------------------------
# UI — results panel (SRS UC-11)
# ---------------------------------------------------------------------------
def render_results(components, hub_result, drug_result, crispr_result, verdict, report_id):
    style = "strong" if "Strong candidate" in verdict else (
        "weak" if "Neither" in verdict or "No druggability" in verdict else (
            "partial" if ("skipped" in verdict or "no gRNA" in verdict) else "mixed"
        )
    )
    st.markdown(f'<div class="bmx-verdict {style}">📋 {verdict}</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Hub gene", hub_result.hub_gene_symbol)
    m2.metric("Druggability", f"{drug_result.druggability_score:.2f}" if drug_result else "—")
    m3.metric("CRISPR safety", f"{crispr_result.safety_score:.2f}" if crispr_result else "—")

    tab1, tab2, tab3 = st.tabs(["🕸️ Network", "💊 Druggability", "✂️ CRISPR Safety"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.caption(
                f"{hub_result.graph_summary.get('num_nodes', 0)} genes, "
                f"{hub_result.graph_summary.get('num_edges', 0)} interactions — "
                f"red node is the identified hub gene."
            )
            fig = render_network_figure(
                components["target_pipeline"].analyzer.graph, hub_result.hub_gene_symbol
            )
            st.pyplot(fig, use_container_width=True)
        with c2:
            df = pd.DataFrame(
                sorted(hub_result.centrality_scores.items(), key=lambda x: -x[1]),
                columns=["Gene", "Centrality score"],
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download centrality table (CSV)",
                df.to_csv(index=False),
                file_name=f"{hub_result.disease_name}_centrality.csv",
                mime="text/csv",
            )

    with tab2:
        if drug_result is None:
            st.info("Druggability analysis wasn't run for this query.")
        else:
            if drug_result.used_fallback:
                st.warning(
                    "No structure/pocket was found — this is the sequence-based "
                    "heuristic fallback, **not** a live model prediction.",
                    icon="⚠️",
                )
            elif drug_result.pdb_id:
                st.caption(f"Structure used: **{drug_result.pdb_id}**")
            if drug_result.pocket_features:
                st.write("**Pocket features:**")
                st.dataframe(
                    pd.DataFrame(
                        drug_result.pocket_features.items(), columns=["Feature", "Value"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with tab3:
        if crispr_result is None:
            st.info("CRISPR safety analysis wasn't run for this query.")
        else:
            if crispr_result.used_fallback:
                st.warning(
                    "No sequence found or no trained CNN loaded yet — this uses "
                    "the documented fallback/interim heuristic, not a model prediction.",
                    icon="⚠️",
                )
            if crispr_result.flagged_sites:
                st.write(f"**{len(crispr_result.flagged_sites)} flagged off-target site(s):**")
                df = pd.DataFrame(
                    [
                        {
                            "Position": s.position,
                            "Sequence": s.sequence,
                            "Mismatches": s.mismatches,
                            "PAM OK": s.pam_ok,
                            "Risk score": round(s.risk_score, 3),
                        }
                        for s in crispr_result.flagged_sites
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download off-target sites (CSV)",
                    df.to_csv(index=False),
                    file_name=f"{hub_result.disease_name}_off_target_sites.csv",
                    mime="text/csv",
                )
            else:
                st.success("No high-risk off-target sites flagged.")

    st.divider()
    report_md = build_markdown_report(hub_result, drug_result, crispr_result, verdict)
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "📄 Download full report (Markdown)",
            report_md,
            file_name=f"biomedix_report_{hub_result.disease_name.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    if report_id is not None:
        st.caption(f"Saved to the database as report #{report_id} — visible in History.")


# ---------------------------------------------------------------------------
# UI — History page
# ---------------------------------------------------------------------------
def render_history(db):
    st.title("Past reports")
    if db is None:
        st.info("No database connection — history isn't available.")
        return

    rows = db.fetch_all("INTEGRATED_REPORT")
    if not rows:
        st.info("No reports yet — run an analysis first.")
        return

    rows = sorted(rows, key=lambda r: r["report_id"], reverse=True)
    options = {f"#{r['report_id']} — {r['generated_at']}": r["report_id"] for r in rows}
    choice = st.selectbox("Select a report", list(options.keys()))
    report_id = options[choice]

    full = db.get_integrated_report(report_id)
    if full is None:
        st.error("Report not found.")
        return

    st.subheader(f"{full['disease_name']} — hub gene {full['hub_gene_symbol']}")
    st.markdown(f'<div class="bmx-verdict mixed">📋 {full["verdict_text"]}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Centrality method", full["centrality_method"])
    c2.metric(
        "Druggability",
        f"{full['druggability_score']:.2f}" if full["druggability_score"] is not None else "—",
    )
    c3.metric(
        "CRISPR safety",
        f"{full['safety_score']:.2f}" if full["safety_score"] is not None else "—",
    )

    lines = [
        f"# BioMedix-AI Report #{report_id} — {full['disease_name']}",
        f"_Generated {full['generated_at']}_",
        "",
        f"## Verdict\n{full['verdict_text']}",
        "",
        f"- Hub gene: `{full['hub_gene_symbol']}`",
        f"- Centrality method: {full['centrality_method']} (score: {full['centrality_score']:.4f})",
        f"- Druggability score: {full['druggability_score']}",
        f"- CRISPR safety score: {full['safety_score']}",
    ]
    st.download_button(
        "📄 Download this report (Markdown)",
        "\n".join(lines),
        file_name=f"biomedix_report_{report_id}.md",
        mime="text/markdown",
    )

    with st.expander("All reports (raw table)"):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    db = get_db()
    components = get_components(db)

    page = render_sidebar()

    if page == "📜 History":
        render_history(db)
        return

    inputs = render_input_form()
    if inputs is None:
        return

    try:
        hub_result, drug_result, crispr_result, verdict, report_id = run_analysis(
            components, db, inputs
        )
        render_results(components, hub_result, drug_result, crispr_result, verdict, report_id)
    except Exception as e:
        st.error(f"Pipeline run failed: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()