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


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
