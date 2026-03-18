"""CLI for socr — Multi-Engine Document Processing."""

from pathlib import Path

import click
from rich.console import Console

from socr import __version__
from socr.core.config import EngineType, PipelineConfig

console = Console()

ENGINE_CHOICES = [e.value for e in EngineType if e not in (EngineType.DEEPSEEK_VLLM, EngineType.VLLM)]


class PDFShortcutGroup(click.Group):
    """Allows PDF paths as shorthand for the process command."""

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple:
        if args and args[0].lower().endswith(".pdf"):
            return "process", self.get_command(ctx, "process"), args
        return super().resolve_command(ctx, args)


# --- Shared options ---

def common_options(f):
    """Options shared between process and batch."""
    f = click.option("--primary", type=click.Choice(ENGINE_CHOICES), help="Primary OCR engine")(f)
    f = click.option("--fallback", type=click.Choice(ENGINE_CHOICES), help="Fallback OCR engine")(f)
    f = click.option("--no-audit", is_flag=True, help="Skip quality audit stage")(f)
    f = click.option("--timeout", type=int, default=1800, help="Subprocess timeout in seconds")(f)
    f = click.option("--save-figures", is_flag=True, help="Save extracted figure images")(f)
    f = click.option("--reprocess", is_flag=True, help="Reprocess already-processed files")(f)
    f = click.option("--dry-run", is_flag=True, help="List files without processing")(f)
    f = click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")(f)
    f = click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")(f)
    f = click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), help="YAML config file")(f)
    f = click.option("--profile", type=str, help="Load ~/.config/socr/{profile}.yaml")(f)
    return f


def build_config(
    primary: str | None = None,
    fallback: str | None = None,
    no_audit: bool = False,
    timeout: int = 300,
    save_figures: bool = False,
    reprocess: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    config_path: Path | None = None,
    profile: str | None = None,
    output_dir: Path | None = None,
) -> PipelineConfig:
    """Build PipelineConfig from CLI options."""
    if config_path or profile:
        try:
            config = PipelineConfig.load(profile=profile, config_path=config_path)
        except FileNotFoundError as e:
            raise click.ClickException(str(e))
    else:
        config = PipelineConfig()

    if primary:
        config.primary_engine = EngineType(primary)
    if fallback:
        config.fallback_engine = EngineType(fallback)
    if no_audit:
        config.audit_enabled = False

    config.timeout = timeout
    config.save_figures = save_figures
    config.reprocess = reprocess
    config.dry_run = dry_run
    config.quiet = quiet
    config.verbose = verbose

    if output_dir:
        config.output_dir = output_dir

    return config


# --- Commands ---

@click.group(cls=PDFShortcutGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="socr")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """socr — Multi-Engine Document Processing.

    Usage:
        socr paper.pdf                    # Process PDF (shorthand)
        socr process paper.pdf [OPTIONS]  # Full options
        socr batch ./papers/ [OPTIONS]    # Process directory
        socr engines                      # Check engine status
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), help="Output directory")
@click.option("--hpc-sequential", is_flag=True, help="Use HPC sequential pipeline (vLLM)")
@click.option("--unified", is_flag=True, help="Use UnifiedPipeline (5-phase orchestrator)")
@common_options
def process(pdf_path: Path, output_dir: Path | None, hpc_sequential: bool = False, unified: bool = False, **kwargs) -> None:
    """Process a single PDF document.

    Uses cascading fallback: primary engine first, quality audit,
    then fallback engine for failed documents.

    Example:
        socr process paper.pdf -o ./results/
        socr paper.pdf --primary gemini --quiet
        socr paper.pdf --hpc-sequential --save-figures
        socr paper.pdf --unified
    """
    config = build_config(output_dir=output_dir, **kwargs)

    if hpc_sequential:
        from socr.pipeline.hpc_pipeline import HPCPipeline

        config.hpc.enabled = True
        config.hpc.sequential = True
        pipeline = HPCPipeline(config)
    elif unified:
        from socr.pipeline.orchestrator import UnifiedPipeline

        pipeline = UnifiedPipeline(config)
    else:
        from socr.pipeline.processor import StandardPipeline

        pipeline = StandardPipeline(config)

    try:
        result = pipeline.process(pdf_path, output_dir)
        if not result.success:
            raise click.ClickException(f"Processing failed: {result.error}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        raise click.Abort()


@cli.command()
@click.argument("pdf_dir", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), help="Output directory")
@click.option("--limit", type=int, help="Maximum number of PDFs to process")
@common_options
def batch(pdf_dir: Path, output_dir: Path | None, limit: int | None, **kwargs) -> None:
    """Process all PDFs in a directory.

    Supports incremental processing — unchanged files are skipped
    (use --reprocess to force).

    Example:
        socr batch ~/Papers/ -o ./results/
        socr batch ~/Papers/ --dry-run
        socr batch ~/Papers/ --reprocess --quiet
    """
    from socr.pipeline.processor import StandardPipeline

    config = build_config(output_dir=output_dir, **kwargs)

    # Handle --limit by pre-filtering
    if limit:
        pdfs = sorted(pdf_dir.glob("*.pdf"))[:limit]
        if not pdfs:
            console.print("[yellow]No PDF files found[/yellow]")
            return
        # Process individually with limit applied
        pipeline = StandardPipeline(config)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            limited_dir = Path(tmpdir)
            for pdf in pdfs:
                (limited_dir / pdf.name).symlink_to(pdf)
            pipeline.process_batch(limited_dir, output_dir)
    else:
        pipeline = StandardPipeline(config)
        pipeline.process_batch(pdf_dir, output_dir)


@cli.command()
def engines() -> None:
    """Show available OCR engines and their status."""
    from socr.engines.registry import get_engine

    console.print("\n[bold]Engines[/bold]\n")

    engine_info = [
        (EngineType.GLM, "local via Ollama (0.9B, ~10s/page)"),
        (EngineType.NOUGAT, "local, academic papers"),
        (EngineType.DEEPSEEK, "local via Ollama"),
        (EngineType.MARKER, "local, layout-aware (Surya + Texify)"),
        (EngineType.GEMINI, "cloud, ~$0.0002/page"),
        (EngineType.MISTRAL, "cloud, ~$0.001/page"),
    ]

    for engine_type, desc in engine_info:
        engine = get_engine(engine_type)
        available = engine.is_available()
        status = "[green]+[/green]" if available else "[red]x[/red]"
        console.print(f"  [{status}] {engine_type.value:<12} [dim]{desc}[/dim]")


@cli.group()
def benchmark() -> None:
    """Benchmark suite for OCR quality evaluation.

    Commands:
        socr benchmark init        Create benchmark set and extract ground truth
        socr benchmark run         Run engines on benchmark papers
        socr benchmark score       Print results summary table
        socr benchmark calibrate   Calibrate repair routing from data
    """


@benchmark.command("init")
@click.option(
    "--papers-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing benchmark PDFs (default: Papers library)",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("benchmark"),
    help="Output directory for benchmark data (default: ./benchmark)",
)
def benchmark_init(papers_dir: Path | None, output_dir: Path) -> None:
    """Create benchmark set, extract ground truth, and generate scanned PDFs.

    Resolves the 10 benchmark papers from the Papers library, extracts
    native text as ground truth, and creates 2 synthetic scanned PDFs.
    """
    from socr.benchmark.dataset import build_benchmark_set, BenchmarkPaper
    from socr.benchmark.ground_truth import GroundTruthExtractor
    from socr.benchmark.rasterize import PaperRasterizer, RASTERIZE_SPECS

    output_dir = Path(output_dir)

    # 1. Build benchmark set
    console.print("[bold]Building benchmark set...[/bold]")
    try:
        bench = build_benchmark_set(papers_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    console.print(f"  Found {len(bench.papers)} papers")
    for cat, papers in sorted(bench.by_category().items()):
        console.print(f"    {cat}: {len(papers)} papers")

    # 2. Extract ground truth
    console.print("\n[bold]Extracting ground truth...[/bold]")
    extractor = GroundTruthExtractor()
    gt_dir = output_dir / "ground_truth"

    for paper in bench.papers:
        paper_gt_dir = gt_dir / paper.name
        console.print(f"  {paper.name} ({paper.page_count}p)...", end=" ")
        truths = extractor.extract_and_save(paper.pdf_path, paper_gt_dir)
        paper.ground_truth_path = paper_gt_dir
        total_words = sum(t.word_count for t in truths)
        console.print(f"[green]{total_words} words[/green]")

    # 3. Rasterize synthetic scanned PDFs
    console.print("\n[bold]Creating synthetic scanned PDFs...[/bold]")
    rasterizer = PaperRasterizer()
    scanned_dir = output_dir / "scanned"
    paper_by_name = {p.name: p for p in bench.papers}

    for spec in RASTERIZE_SPECS:
        source_paper = paper_by_name.get(spec["source_name"])
        if not source_paper:
            console.print(f"  [yellow]Skipping {spec['source_name']}: not found[/yellow]")
            continue

        out_path = scanned_dir / f"{spec['output_name']}.pdf"
        console.print(f"  {spec['output_name']} @ {spec['dpi']} DPI...", end=" ")
        rasterizer.rasterize(source_paper.pdf_path, out_path, dpi=spec["dpi"])

        # Add scanned version to benchmark set
        scanned_paper = BenchmarkPaper(
            name=spec["output_name"],
            pdf_path=out_path,
            category="scanned",
            page_count=source_paper.page_count,
            ground_truth_path=source_paper.ground_truth_path,
            notes=spec["notes"],
        )
        bench.papers.append(scanned_paper)
        console.print("[green]done[/green]")

    # 4. Save benchmark set manifest
    manifest_path = output_dir / "benchmark.json"
    bench.save(manifest_path)
    console.print(f"\n[bold green]Benchmark set saved:[/bold green] {manifest_path}")
    console.print(f"  {len(bench.papers)} papers ({len(RASTERIZE_SPECS)} scanned)")


@benchmark.command("run")
@click.option(
    "--benchmark-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("benchmark"),
    help="Benchmark directory (default: ./benchmark)",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("benchmark/results"),
    help="Results output directory (default: ./benchmark/results)",
)
@click.option(
    "--engines",
    "engine_names",
    type=str,
    default=None,
    help="Comma-separated list of engines to run (default: all available)",
)
def benchmark_run(benchmark_dir: Path, output_dir: Path, engine_names: str | None) -> None:
    """Run OCR engines on benchmark papers and score results.

    Loads the benchmark set, runs each selected engine on each paper,
    scores against ground truth, and saves results.

    Example:
        socr benchmark run
        socr benchmark run --engines gemini,deepseek
        socr benchmark run --benchmark-dir ./my-bench -o ./my-results
    """
    from socr.benchmark.dataset import BenchmarkSet
    from socr.benchmark.runner import BenchmarkRunner

    manifest = benchmark_dir / "benchmark.json"
    if not manifest.exists():
        raise click.ClickException(
            f"Benchmark manifest not found: {manifest}\n"
            "Run 'socr benchmark init' first."
        )

    bench = BenchmarkSet.load(manifest)
    console.print(f"[bold]Loaded benchmark:[/bold] {len(bench.papers)} papers")

    # Parse engine selection
    engines: list[EngineType] | None = None
    if engine_names:
        try:
            engines = [EngineType(e.strip()) for e in engine_names.split(",")]
        except ValueError as exc:
            raise click.ClickException(f"Unknown engine: {exc}")

    config = PipelineConfig()
    runner = BenchmarkRunner(config)

    console.print("[bold]Running benchmark...[/bold]")
    results = runner.run(bench, output_dir, engines=engines)

    # Save results
    results_path = output_dir / "results.json"
    results.save(results_path)
    console.print(f"\n[bold green]Results saved:[/bold green] {results_path}")

    # Print summary
    _print_results_summary(results)


@benchmark.command("score")
@click.option(
    "--results-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("benchmark/results/results.json"),
    help="Path to results JSON (default: ./benchmark/results/results.json)",
)
def benchmark_score(results_file: Path) -> None:
    """Print a summary table of benchmark results.

    Loads saved benchmark results and displays WER/CER per engine and paper.

    Example:
        socr benchmark score
        socr benchmark score --results-file ./my-results/results.json
    """
    from socr.benchmark.runner import BenchmarkResults

    results = BenchmarkResults.load(results_file)
    console.print(f"[bold]Loaded results:[/bold] {len(results.runs)} runs")
    _print_results_summary(results)


@benchmark.command("calibrate")
@click.option(
    "--results-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("benchmark/results/results.json"),
    help="Path to results JSON (default: ./benchmark/results/results.json)",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Save calibration report to this path",
)
@click.option(
    "--apply",
    "apply_config",
    type=click.Path(path_type=Path),
    default=None,
    help="Write calibrated config to this YAML path",
)
def benchmark_calibrate(
    results_file: Path,
    output_path: Path | None,
    apply_config: Path | None,
) -> None:
    """Calibrate repair routing from benchmark results.

    Analyzes benchmark results to determine optimal engine chains
    per document category and prints recommendations.

    Example:
        socr benchmark calibrate
        socr benchmark calibrate -o calibration.json
        socr benchmark calibrate --apply ~/.config/socr/config.yaml
    """
    from socr.benchmark.calibrate import RepairCalibrator
    from socr.benchmark.runner import BenchmarkResults

    results = BenchmarkResults.load(results_file)
    console.print(f"[bold]Loaded results:[/bold] {len(results.runs)} runs")

    calibrator = RepairCalibrator()
    report = calibrator.calibrate(results)

    # Print engine profiles
    console.print("\n[bold]Engine Profiles[/bold]\n")
    for profile in report.profiles:
        avg_wer = sum(profile.category_wer.values()) / len(profile.category_wer) if profile.category_wer else float("nan")
        console.print(
            f"  {profile.engine:<12} "
            f"avg_wer={avg_wer:.3f}  "
            f"avg_time={profile.avg_processing_time:.1f}s"
        )
        if profile.failure_mode_recovery:
            for fm, rate in sorted(profile.failure_mode_recovery.items()):
                console.print(f"    {fm}: recovery={rate:.0%}")

    # Print recommended chains
    console.print("\n[bold]Recommended Engine Chains[/bold]\n")
    for category, chain in sorted(report.recommended_chain.items()):
        console.print(f"  {category}: {' -> '.join(chain)}")

    # Save report
    if output_path:
        report.save(output_path)
        console.print(f"\n[bold green]Calibration report saved:[/bold green] {output_path}")

    # Apply to config
    if apply_config:
        import yaml

        config = PipelineConfig()
        calibrator.apply_to_config(report, config)

        config_data = {
            "primary_engine": config.primary_engine.value,
            "fallback_chain": [e.value for e in config.fallback_chain],
        }

        apply_config.parent.mkdir(parents=True, exist_ok=True)
        apply_config.write_text(yaml.dump(config_data, default_flow_style=False))
        console.print(f"[bold green]Config written:[/bold green] {apply_config}")


def _print_results_summary(results) -> None:
    """Print a summary table of benchmark results."""
    from rich.table import Table

    by_engine = results.by_engine()

    table = Table(title="Benchmark Results")
    table.add_column("Engine", style="cyan")
    table.add_column("Papers", justify="right")
    table.add_column("Scored", justify="right")
    table.add_column("Avg WER", justify="right")
    table.add_column("Avg CER", justify="right")
    table.add_column("Avg Time", justify="right")

    for engine_name in sorted(by_engine):
        runs = by_engine[engine_name]
        scored = [r for r in runs if r.score is not None]
        avg_wer = sum(r.score.overall_wer for r in scored) / len(scored) if scored else float("nan")
        avg_cer = sum(r.score.overall_cer for r in scored) / len(scored) if scored else float("nan")
        avg_time = sum(r.result.processing_time for r in runs) / len(runs) if runs else 0.0

        table.add_row(
            engine_name,
            str(len(runs)),
            str(len(scored)),
            f"{avg_wer:.3f}" if scored else "N/A",
            f"{avg_cer:.3f}" if scored else "N/A",
            f"{avg_time:.1f}s",
        )

    console.print(table)

    # Per-paper breakdown
    by_paper = results.by_paper()
    if by_paper:
        paper_table = Table(title="Per-Paper Results")
        paper_table.add_column("Paper", style="cyan")
        paper_table.add_column("Engine", style="green")
        paper_table.add_column("WER", justify="right")
        paper_table.add_column("CER", justify="right")
        paper_table.add_column("Status")

        for paper_name in sorted(by_paper):
            runs = by_paper[paper_name]
            for run in runs:
                if run.score:
                    paper_table.add_row(
                        paper_name,
                        run.engine,
                        f"{run.score.overall_wer:.3f}",
                        f"{run.score.overall_cer:.3f}",
                        "[green]OK[/green]",
                    )
                else:
                    paper_table.add_row(
                        paper_name,
                        run.engine,
                        "N/A",
                        "N/A",
                        f"[red]{run.result.failure_mode.value}[/red]",
                    )

        console.print(paper_table)


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
