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
        socr benchmark init    Create benchmark set and extract ground truth
        socr benchmark run     Run all engines on benchmark (not yet implemented)
        socr benchmark score   Score results against ground truth (not yet implemented)
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
def benchmark_run() -> None:
    """Run all engines on benchmark papers.

    Not implemented yet — placeholder for L2C-02.
    """
    console.print("[yellow]Not implemented yet.[/yellow] See ticket L2C-02.")


@benchmark.command("score")
def benchmark_score() -> None:
    """Score OCR results against ground truth.

    Not implemented yet — placeholder for L2C-02.
    """
    console.print("[yellow]Not implemented yet.[/yellow] See ticket L2C-02.")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
