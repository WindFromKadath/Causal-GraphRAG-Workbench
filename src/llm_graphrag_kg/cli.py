"""Command-line interface for LLM-GraphRAG-KG."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_graphrag_kg.adapters.failuresensoriq import (
    DEFAULT_STRESS_SUITES,
    build_failuresensoriq_kg,
    evaluate_failuresensoriq,
    plan_failuresensoriq_stress_suite,
    run_failuresensoriq_stress_suite,
)
from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.factory import create_llm_client
from llm_graphrag_kg.viz.audit_report import export_audit_artifacts
from llm_graphrag_kg.viz.graphml_exporter import export_graphml
from llm_graphrag_kg.viz.html_exporter import export_pyvis_html

app = typer.Typer(
    name="llm-graphrag-kg",
    help="Modular LLM + GraphRAG + Knowledge Graph framework with visual verification.",
    no_args_is_help=True,
)
console = Console()


@app.command("check-llm")
def check_llm(
    provider: Annotated[
        str | None, typer.Option(help="Override LLM provider: openai_compat | mock")
    ] = None,
    config_file: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
) -> None:
    """Inspect active config (API keys redacted) and verify LLM endpoint connectivity."""
    cfg = load_config(config_path=config_file, llm_provider_override=provider)
    summary = cfg.redacted_summary()
    table = Table(title="Active Configuration (Redacted)")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)

    client = create_llm_client(cfg)
    ok, msg = client.health_check()
    style = "green" if ok else "red"
    console.print(Panel(msg, title="LLM Connectivity Check", border_style=style))
    if not ok:
        raise typer.Exit(code=1)


@app.command("build")
def build(
    input_dir: Annotated[
        Path | None, typer.Option("--input", help="Input documents directory.")
    ] = None,
    storage_dir: Annotated[
        Path | None, typer.Option("--storage", help="Storage output directory.")
    ] = None,
    provider: Annotated[
        str | None, typer.Option(help="Override LLM provider: openai_compat | mock")
    ] = None,
    embedder: Annotated[
        str | None, typer.Option(help="Override embedding provider: local_hash | openai_compat")
    ] = None,
    summarize_communities: Annotated[
        bool, typer.Option("--summarize-communities", help="Use LLM to summarize communities.")
    ] = False,
    config_file: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
) -> None:
    """Run offline ingestion, KG extraction, community detection, and index persistence."""
    cfg = load_config(
        config_path=config_file,
        llm_provider_override=provider,
        embedding_provider_override=embedder,
    )
    pipeline = GraphRAGPipeline(cfg)
    stats = pipeline.build(
        input_dir=input_dir,
        storage_dir=storage_dir,
        summarize_communities=summarize_communities,
    )

    table = Table(title="Knowledge Graph Build Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Documents", str(stats.doc_count))
    table.add_row("Chunks", str(stats.chunk_count))
    table.add_row("Entities", str(stats.entity_count))
    table.add_row("Relations", str(stats.relation_count))
    table.add_row("Communities", str(stats.community_count))
    table.add_row("Connected Components", str(stats.connected_components))
    table.add_row("Isolated Entities", str(len(stats.isolated_entities)))
    console.print(table)


@app.command("audit")
def audit(
    storage_dir: Annotated[
        Path | None, typer.Option("--storage", help="Persisted storage dir.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option("--output", help="Audit report output dir.")
    ] = None,
    config_file: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
) -> None:
    """Generate Markdown + CSV quality verification reports for the persisted KG."""
    cfg = load_config(config_path=config_file, llm_provider_override="mock")
    pipeline = GraphRAGPipeline(cfg)
    doc_count = pipeline.load(storage_dir)
    out_dir = output_dir or cfg.output_path
    stats, paths = export_audit_artifacts(
        pipeline.graph_store,
        pipeline.vector_store,
        out_dir,
        doc_count=doc_count,
    )

    console.print(
        Panel(
            f"Entities={stats.entity_count} | Relations={stats.relation_count} | "
            f"Communities={stats.community_count} | Isolated={len(stats.isolated_entities)}\n"
            f"Report: {paths['audit_report']}\n"
            f"Entities CSV: {paths['entities_csv']}\n"
            f"Relations CSV: {paths['relations_csv']}",
            title="KG Audit Artifacts Exported",
            border_style="green",
        )
    )


@app.command("export-viz")
def export_viz(
    storage_dir: Annotated[
        Path | None, typer.Option("--storage", help="Persisted storage dir.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option("--output", help="Visualization output dir.")
    ] = None,
    color_by: Annotated[
        str, typer.Option(help="Node color grouping: type | community")
    ] = "type",
    config_file: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
) -> None:
    """Export interactive PyVis HTML graph and Gephi-compatible GraphML file."""
    cfg = load_config(config_path=config_file, llm_provider_override="mock")
    pipeline = GraphRAGPipeline(cfg)
    pipeline.load(storage_dir)
    out_dir = output_dir or cfg.output_path

    html_path = export_pyvis_html(
        pipeline.graph_store,
        out_dir / "kg_graph.html",
        vector_store=pipeline.vector_store,
        color_by=color_by,
        physics_enabled=cfg.visualization.physics_enabled,
    )
    graphml_path = export_graphml(pipeline.graph_store, out_dir / "kg_graph.graphml")

    console.print(
        Panel(
            f"Interactive HTML: {html_path}\nGraphML (Gephi/Cytoscape): {graphml_path}",
            title="Visualization Exported",
            border_style="cyan",
        )
    )


@app.command("query")
def query(
    question: Annotated[str, typer.Argument(help="Question to ask the GraphRAG system.")],
    mode: Annotated[
        str | None, typer.Option(help="Retrieval mode: local | global | hybrid")
    ] = None,
    provider: Annotated[
        str | None, typer.Option(help="Override LLM provider: openai_compat | mock")
    ] = None,
    embedder: Annotated[
        str | None, typer.Option(help="Override embedding provider: local_hash | openai_compat")
    ] = None,
    storage_dir: Annotated[
        Path | None, typer.Option("--storage", help="Persisted storage dir.")
    ] = None,
    export_html: Annotated[
        bool,
        typer.Option("--export-html", help="Export PyVis HTML highlighting the retrieved subgraph."),
    ] = False,
    compact: Annotated[
        bool | None,
        typer.Option("--compact/--no-compact", help="Enable Stage 2.1 L0/L1 context compaction."),
    ] = None,
    config_file: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
) -> None:
    """Execute a GraphRAG query and inspect answer, subgraph trace, and source evidences."""
    cfg = load_config(
        config_path=config_file,
        llm_provider_override=provider,
        embedding_provider_override=embedder,
    )
    pipeline = GraphRAGPipeline(cfg)
    result = pipeline.query(
        question, mode=mode, storage_dir=storage_dir, enable_compact=compact
    )

    console.print(Panel(result.answer, title=f"GraphRAG Answer ({result.mode})", border_style="green"))

    trace_table = Table(title="Retrieval Trace")
    trace_table.add_column("Dimension", style="cyan")
    trace_table.add_column("Details", style="white")
    trace_table.add_row("Seed Entities", ", ".join(result.metadata.get("seed_entity_names", [])) or "-")
    trace_table.add_row("Expanded Subgraph Nodes", str(len(result.trace.expanded_entity_ids)))
    trace_table.add_row("Subgraph Relations", str(len(result.trace.relation_ids)))
    trace_table.add_row("Communities Used", str(result.trace.community_ids))
    trace_table.add_row("Evidence Chunks", str(len(result.evidences)))
    comp = result.metadata.get("compaction")
    if isinstance(comp, dict):
        trace_table.add_row(
            "Context Compaction",
            f"{comp['raw_tokens_est']} -> {comp['compacted_tokens_est']} tok "
            f"(-{comp['compression_ratio'] * 100:.1f}%, dropped_nodes={comp['dropped_nodes']}, "
            f"levels={comp['applied_levels']})",
        )
    console.print(trace_table)

    if export_html:
        trace_html = export_pyvis_html(
            pipeline.graph_store,
            cfg.output_path / "query_trace.html",
            vector_store=pipeline.vector_store,
            color_by="community",
            trace=result.trace,
            title=f"Query Subgraph Trace ({result.mode}): {question[:60]}",
        )
        console.print(f"[cyan]Trace HTML exported to:[/cyan] {trace_html}")


@app.command("privacy-check")
def privacy_check(
    staged: Annotated[bool, typer.Option("--staged", help="Scan only git staged files.")] = False,
    require_changelog: Annotated[
        bool, typer.Option("--require-changelog", help="Require CHANGELOG.md in scanned files.")
    ] = False,
) -> None:
    """Run privacy & commit hygiene checks before committing or pushing to GitHub."""
    script_path = Path.cwd() / "scripts" / "privacy_check.py"
    if not script_path.is_file():
        console.print("[red]scripts/privacy_check.py not found in current repository root.[/red]")
        raise typer.Exit(code=1)

    spec = importlib.util.spec_from_file_location("privacy_check", script_path)
    if spec is None or spec.loader is None:
        raise typer.Exit(code=1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv: list[str] = []
    if staged:
        argv.append("--staged")
    if require_changelog:
        argv.append("--require-changelog")
    exit_code = int(module.main(argv))
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("ui")
def ui(
    host: Annotated[str, typer.Option(help="Host address to bind Streamlit server.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind Streamlit server.")] = 8501,
) -> None:
    """Launch the Phase 2 Streamlit visual verification workbench."""
    if importlib.util.find_spec("streamlit") is None:
        console.print(
            "[red]Streamlit is not installed in the current environment.[/red]\n"
            "Run: [cyan]uv sync --extra viz[/cyan]"
        )
        raise typer.Exit(code=1)

    app_path = Path(__file__).resolve().parent / "viz" / "app.py"
    console.print(
        Panel(
            f"Launching visual verification workbench at http://{host}:{port}\n"
            "Press Ctrl+C to stop.",
            title="LLM-GraphRAG-KG Workbench",
            border_style="cyan",
        )
    )
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("ingest-fsiq")
def ingest_fsiq(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            help="Path to FailureSensorIQ JSONL (failuresensoriq.mapped.jsonl or all.jsonl).",
        ),
    ],
    config_file: Annotated[
        Path, typer.Option("--config-file", help="YAML config path.")
    ] = Path("configs/failuresensoriq.yaml"),
    summarize_communities: Annotated[
        bool, typer.Option("--summarize-communities", help="Use LLM to summarize communities.")
    ] = False,
    provider: Annotated[
        str | None, typer.Option(help="Override LLM provider: openai_compat | mock")
    ] = None,
) -> None:
    """Ingest IBM FailureSensorIQ into an Ontology-First, ER-cleaned Knowledge Graph."""
    cfg = load_config(config_path=config_file, llm_provider_override=provider)
    stats = build_failuresensoriq_kg(
        source_path=source,
        config=cfg,
        summarize_communities=summarize_communities,
    )

    table = Table(title="FailureSensorIQ Ingestion & ER Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Source JSONL Rows", str(stats.source_records))
    table.add_row("Gold Positive Triples", str(stats.gold_triples_loaded))
    table.add_row("Industrial Assets (Equipment)", str(stats.asset_count))
    table.add_row("Asset-Scoped FailureModes", str(stats.failure_mode_count))
    table.add_row("Canonical FailureCategories", str(stats.failure_category_count))
    table.add_row("Normalized Sensors (SignalChannel)", str(stats.sensor_count))
    table.add_row("ER Typo/Alias Fixes Applied", str(stats.er_normalized_sensor_mentions))
    table.add_row("Total Graph Entities", str(stats.audit.entity_count))
    table.add_row("Total Graph Relations", str(stats.audit.relation_count))
    table.add_row("Louvain Communities", str(stats.audit.community_count))
    table.add_row("Connected Components", str(stats.audit.connected_components))
    table.add_row("Isolated Entities", str(len(stats.audit.isolated_entities)))
    console.print(table)


@app.command("eval-fsiq")
def eval_fsiq(
    questions: Annotated[
        Path,
        typer.Option(
            "--questions",
            help="Path to FailureSensorIQ MCQA JSONL (e.g., failuresensoriq_standard__all.jsonl).",
        ),
    ],
    config_file: Annotated[
        Path, typer.Option("--config-file", help="YAML config path.")
    ] = Path("configs/failuresensoriq.yaml"),
    limit: Annotated[int, typer.Option(help="Number of stratified questions to evaluate.")] = 20,
    seed: Annotated[int, typer.Option(help="Random seed for balanced sampling.")] = 42,
    modes: Annotated[
        str,
        typer.Option(help="Comma-separated modes: llm_only,naive_rag,graphrag_kg"),
    ] = "llm_only,naive_rag,graphrag_kg",
    provider: Annotated[
        str | None, typer.Option(help="Override LLM provider: openai_compat | mock")
    ] = None,
) -> None:
    """Evaluate LLM-only vs Naive RAG vs GraphRAG+KG accuracy on FailureSensorIQ MCQA."""
    cfg = load_config(config_path=config_file, llm_provider_override=provider)
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    summary, report_path = evaluate_failuresensoriq(
        questions_path=questions,
        config=cfg,
        limit=limit,
        seed=seed,
        modes=mode_list,
    )

    table = Table(
        title=f"FailureSensorIQ Evaluation ({summary.total_questions} Qs | {summary.llm_model})"
    )
    table.add_column("Mode", style="cyan")
    table.add_column("Correct / Total", style="white")
    table.add_column("Accuracy", style="green")
    for m in summary.modes:
        table.add_row(m.mode, f"{m.correct} / {m.total}", f"{m.accuracy * 100:.1f}%")
    console.print(table)
    console.print(f"[cyan]Detailed report written to:[/cyan] {report_path}")


@app.command("stress-fsiq")
def stress_fsiq(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            help="Directory holding the 4 FailureSensorIQ JSONL benchmark suites.",
        ),
    ] = Path("data/private/failuresensoriq"),
    config_file: Annotated[
        Path, typer.Option("--config-file", help="YAML config path.")
    ] = Path("configs/failuresensoriq.yaml"),
    limit: Annotated[
        int, typer.Option(help="Questions per suite (set 0 for FULL dataset).")
    ] = 100,
    seed: Annotated[int, typer.Option(help="Random seed for stratified sampling.")] = 42,
    suites: Annotated[
        str | None,
        typer.Option(
            "--suites",
            help="Comma-separated suite names (default: all 4 suites).",
        ),
    ] = None,
    modes: Annotated[
        str,
        typer.Option(help="Comma-separated modes: llm_only,naive_rag,graphrag_kg_raw,graphrag_kg"),
    ] = "llm_only,naive_rag,graphrag_kg_raw,graphrag_kg",
    output_tag: Annotated[
        str,
        typer.Option("--output-tag", help="Subdirectory under output_dir to isolate runs/checkpoints."),
    ] = "stress_test",
    provider: Annotated[
        str | None, typer.Option(help="Override LLM provider: ollama_native | openai_compat | mock")
    ] = None,
    think: Annotated[
        bool | None,
        typer.Option("--think/--no-think", help="Control Ollama thinking mode during stress test."),
    ] = False,
    num_ctx: Annotated[
        int | None, typer.Option("--num-ctx", help="Override Ollama context length (e.g., 32768).")
    ] = 32768,
    concise_json: Annotated[
        bool,
        typer.Option("--concise-json/--verbose-text", help="Enforce compact JSON answer output."),
    ] = True,
    compact: Annotated[
        bool,
        typer.Option("--compact/--no-compact", help="Use Stage 2.1 L0 question-anchor compaction."),
    ] = True,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume from per-suite JSONL checkpoints."),
    ] = True,
    progress_every: Annotated[
        int,
        typer.Option("--progress-every", help="Print live progress & ETA every N questions."),
    ] = 50,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print exact question counts, cached progress, and ETA only."),
    ] = False,
) -> None:
    """Run full or sampled FailureSensorIQ multi-suite benchmark with live ETA and gap matrix."""
    cfg = load_config(config_path=config_file, llm_provider_override=provider)
    if think is not None:
        cfg.env.llm_think = think
    if num_ctx is not None:
        cfg.env.llm_num_ctx = num_ctx

    suite_lookup = dict(DEFAULT_STRESS_SUITES)
    selected_suites = DEFAULT_STRESS_SUITES
    if suites:
        wanted = [s.strip() for s in suites.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in suite_lookup]
        if unknown:
            console.print(
                f"[red]Unknown suite(s): {unknown}. Valid: {list(suite_lookup.keys())}[/red]"
            )
            raise typer.Exit(code=1)
        selected_suites = [(s, suite_lookup[s]) for s in wanted]

    mode_list = [m.strip() for m in modes.split(",") if m.strip()]

    if dry_run:
        plan_rows = plan_failuresensoriq_stress_suite(
            data_dir=data_dir,
            config=cfg,
            output_tag=output_tag,
            limit_per_suite=limit,
            modes=mode_list,
            resume=resume,
            suites=selected_suites,
        )
        ptable = Table(
            title=f"FailureSensorIQ Benchmark Plan (limit={'ALL' if limit <= 0 else limit}, tag={output_tag})"
        )
        ptable.add_column("Suite", style="cyan")
        ptable.add_column("Mode", style="white")
        ptable.add_column("Target Qs", style="green")
        ptable.add_column("Cached", style="yellow")
        ptable.add_column("Remaining Calls", style="magenta")
        ptable.add_column("Est. Time", style="blue")
        tot_calls = 0
        tot_min = 0.0
        for row in plan_rows:
            tot_calls += int(row["remaining_calls"])
            tot_min += float(row["est_minutes"])
            ptable.add_row(
                str(row["suite"]),
                str(row["mode"]),
                str(row["target_questions"]),
                str(row["cached_done"]),
                str(row["remaining_calls"]),
                f"{row['est_minutes']:.1f} min",
            )
        console.print(ptable)
        console.print(
            Panel(
                f"Total Remaining LLM Calls: {tot_calls:,}\n"
                f"Estimated Wall Time (single GPU gemma4:12b, think=False): "
                f"~{tot_min:.1f} min ({tot_min / 60.0:.2f} hours)\n"
                f"Checkpoint Dir: {cfg.output_path / output_tag}",
                title="Dry-Run Summary",
                border_style="cyan",
            )
        )
        return

    report, report_path = run_failuresensoriq_stress_suite(
        data_dir=data_dir,
        config=cfg,
        output_tag=output_tag,
        limit_per_suite=limit,
        seed=seed,
        modes=mode_list,
        concise_json=concise_json,
        compact_anchor_only=compact,
        resume=resume,
        progress_every=progress_every,
        suites=selected_suites,
    )

    label_limit = "ALL" if limit <= 0 else f"{limit} Qs/suite"
    table = Table(
        title=f"FailureSensorIQ Multi-Suite Benchmark ({label_limit} | {report.llm_model})"
    )
    table.add_column("Suite", style="cyan")
    table.add_column("Mode", style="white")
    table.add_column("Exact Match", style="green")
    table.add_column("Macro-F1", style="yellow")
    table.add_column("Avg Prompt Tok", style="blue")
    table.add_column("Repairs", style="white")
    table.add_column("Empty", style="red")
    table.add_column("Avg Latency", style="magenta")
    for s in report.suites:
        for m in s.modes:
            table.add_row(
                s.suite_name,
                m.mode,
                f"{m.accuracy * 100:.1f}% ({m.correct}/{m.total})",
                f"{m.macro_f1 * 100:.1f}%",
                f"{m.avg_prompt_tokens:.0f}",
                str(m.total_repairs),
                str(m.empty_responses),
                f"{m.avg_latency_seconds:.2f}s",
            )
    console.print(table)
    console.print(f"[cyan]Full stress report written to:[/cyan] {report_path}")
