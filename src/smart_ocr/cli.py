"""CLI for smart-ocr - Multi-Engine Document Processing."""

from pathlib import Path

import click
from rich.console import Console

from smart_ocr import __version__
from smart_ocr.core.config import AgentConfig, EngineType
from smart_ocr.pipeline.hpc_pipeline import HPCPipeline
from smart_ocr.pipeline.processor import OCRPipeline
from smart_ocr.ui.theme import AGENT_THEME, ENGINE_LABELS


console = Console(theme=AGENT_THEME)


class PDFShortcutGroup(click.Group):
    """Custom group that allows PDF paths as shorthand for the process command."""

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple:
        """Intercept unknown commands that look like PDF paths."""
        if args:
            cmd_name = args[0]
            # Check if it looks like a PDF file (has .pdf extension)
            if cmd_name.lower().endswith(".pdf"):
                # Treat it as a shortcut to 'process' command
                return "process", self.get_command(ctx, "process"), args
        # Fall back to default behavior
        return super().resolve_command(ctx, args)


@click.group(cls=PDFShortcutGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="smart-ocr")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """smart-ocr - Multi-Engine Document Processing.

    A multi-agent OCR system that uses cascading fallback
    between local and cloud engines for optimal quality and cost.

    Usage:
        smart-ocr paper.pdf                    # Process PDF (shorthand)
        smart-ocr process paper.pdf [OPTIONS]  # Full options
        smart-ocr engines                      # Check engine status
        smart-ocr batch ./papers/              # Process directory
    """
    if ctx.invoked_subcommand is None:
        # No subcommand - show help
        click.echo(ctx.get_help())


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    help="Output file path (default: output/<doc_stem>/<doc_stem>.<ext>)",
)
@click.option(
    "-f", "--format",
    type=click.Choice(["markdown", "json", "txt"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--primary",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    help="Override primary engine selection",
)
@click.option(
    "--fallback",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    help="Override fallback engine selection",
)
@click.option(
    "--no-audit",
    is_flag=True,
    help="Skip quality audit stage",
)
@click.option(
    "--no-figures",
    is_flag=True,
    help="Skip figure processing stage",
)
@click.option(
    "--save-figures",
    is_flag=True,
    help="Save extracted figure images to output/<doc>/figures/",
)
@click.option(
    "--figures-engine",
    type=click.Choice(["gemini", "deepseek", "mistral", "vllm"]),
    help="Engine to use for figure description (default: auto-select)",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Timeout per page/figure in seconds (default: 300)",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Parallel workers for page/figure processing (default: 4, use 1 for sequential)",
)
@click.option(
    "--dpi",
    type=str,
    default="auto",
    help="Render DPI: 'auto' (smart detection), or explicit value like 150, 200, 300",
)
@click.option(
    "--vllm-url",
    type=str,
    help="vLLM server URL (default: http://localhost:8000/v1 or VLLM_BASE_URL)",
)
@click.option(
    "--vllm-model",
    type=str,
    help="vLLM model name (default: Qwen/Qwen2-VL-7B-Instruct)",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--hpc",
    is_flag=True,
    help="Enable HPC mode: parallel multi-engine processing via local vLLM",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom config file (YAML)",
)
@click.option(
    "--profile",
    type=str,
    help="Load profile from ~/.config/smart-ocr/{profile}.yaml",
)
def process(
    pdf_path: Path,
    output: Path | None,
    format: str,
    primary: str | None,
    fallback: str | None,
    no_audit: bool,
    no_figures: bool,
    save_figures: bool,
    figures_engine: str | None,
    timeout: int,
    workers: int,
    dpi: str,
    vllm_url: str | None,
    vllm_model: str | None,
    verbose: bool,
    hpc: bool,
    config_path: Path | None,
    profile: str | None,
) -> None:
    """Process a PDF document with multi-agent OCR.

    Uses cascading fallback: free local engines first (Nougat/DeepSeek),
    quality audit with local LLM, then cloud fallback (Mistral/Gemini)
    for failed pages.

    HPC mode (--hpc) runs multiple OCR engines in parallel via local vLLM
    and reconciles outputs for best quality.

    Example:
        smart-ocr process paper.pdf -o extracted.md
        smart-ocr paper.pdf --hpc --vllm-url http://node:8000/v1
    """
    # Parse DPI (can be "auto" or int)
    render_dpi: int | str = dpi if dpi == "auto" else int(dpi)

    # Load config from file/profile if specified
    if config_path or profile:
        try:
            config = AgentConfig.load_config(profile=profile, config_path=config_path)
        except FileNotFoundError as e:
            raise click.ClickException(str(e))
    else:
        config = AgentConfig()

    # Apply CLI options (override config file settings)
    config.output_format = format
    config.include_figures = not no_figures
    config.save_figures = save_figures
    config.parallel_pages = workers
    config.parallel_figures = max(1, workers // 2)
    config.render_dpi = render_dpi
    config.verbose = verbose

    # HPC mode configuration
    if hpc:
        config.hpc.enabled = True
        if vllm_url:
            config.hpc.vllm_url = vllm_url
        # Disable audit in HPC mode (reconciliation handles quality)
        config.audit.enabled = False

    # Apply timeout to all engine configs
    config.nougat.timeout = timeout
    config.deepseek.timeout = timeout
    config.mistral.timeout = timeout
    config.gemini.timeout = timeout
    config.vllm.timeout = timeout
    config.figure_timeout = timeout
    # Also set engine-level figure timeouts (used by describe_figure)
    config.nougat.figure_timeout = timeout
    config.deepseek.figure_timeout = timeout
    config.mistral.figure_timeout = timeout
    config.gemini.figure_timeout = timeout
    config.vllm.figure_timeout = timeout

    if no_audit:
        config.audit.enabled = False

    if primary:
        config.primary_engine = EngineType(primary)
        config.use_primary_override = True
    if fallback:
        config.fallback_engine = EngineType(fallback)
        config.use_fallback_override = True
    if figures_engine:
        config.figures_engine = EngineType(figures_engine)
        config.use_figures_engine_override = True
    if vllm_url:
        config.vllm.base_url = vllm_url
        config.deepseek_vllm.base_url = vllm_url
    if vllm_model:
        config.vllm.model = vllm_model

    # Select pipeline based on mode
    if config.hpc.enabled:
        pipeline = HPCPipeline(config)
    else:
        pipeline = OCRPipeline(config)

    try:
        result = pipeline.process(pdf_path, output_path=output)

        # Save output
        output_path = pipeline.save_output(result, output)
        console.print(f"\n[success]✓[/success] Output saved to: [info]{output_path}[/info]")

    except KeyboardInterrupt:
        console.print("\n[warning]⚠ Processing cancelled[/warning]")
        raise click.Abort()
    except Exception as e:
        console.print(f"\n[error]✗ Error:[/error] {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument("pdf_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory (default: output/)",
)
@click.option(
    "-f", "--format",
    type=click.Choice(["markdown", "json", "txt"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--primary",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    help="Override primary engine selection",
)
@click.option(
    "--no-audit",
    is_flag=True,
    help="Skip quality audit stage",
)
@click.option(
    "--no-figures",
    is_flag=True,
    help="Skip figure processing stage",
)
@click.option(
    "--save-figures",
    is_flag=True,
    help="Save extracted figure images",
)
@click.option(
    "--figures-engine",
    type=click.Choice(["gemini", "deepseek", "mistral", "vllm"]),
    help="Engine to use for figure description (default: auto-select)",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Timeout per page/figure in seconds (default: 300)",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Parallel workers for page/figure processing (default: 4)",
)
@click.option(
    "--dpi",
    type=str,
    default="auto",
    help="Render DPI: 'auto' (smart detection), or explicit value like 150, 200, 300",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of PDFs to process",
)
@click.option(
    "--hpc",
    is_flag=True,
    help="Enable HPC mode: parallel multi-engine processing via local vLLM",
)
@click.option(
    "--vllm-url",
    type=str,
    help="vLLM server URL for HPC mode (default: http://localhost:8000/v1)",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom config file (YAML)",
)
@click.option(
    "--profile",
    type=str,
    help="Load profile from ~/.config/smart-ocr/{profile}.yaml",
)
def batch(
    pdf_dir: Path,
    output_dir: Path | None,
    format: str,
    primary: str | None,
    no_audit: bool,
    no_figures: bool,
    save_figures: bool,
    figures_engine: str | None,
    timeout: int,
    workers: int,
    dpi: str,
    limit: int | None,
    hpc: bool,
    vllm_url: str | None,
    config_path: Path | None,
    profile: str | None,
) -> None:
    """Process all PDFs in a directory.

    Example:
        smart-ocr batch ~/Papers/ -o extracted/
    """
    # Find all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[warning]No PDF files found in {pdf_dir}[/warning]")
        return

    if limit:
        pdf_files = pdf_files[:limit]

    console.print(f"\n[bold]Batch Processing: {len(pdf_files)} PDFs[/bold]\n")
    if hpc:
        console.print("[info]HPC Mode[/info] - Multi-engine parallel processing\n")

    # Parse DPI (can be "auto" or int)
    render_dpi: int | str = dpi if dpi == "auto" else int(dpi)

    # Load config from file/profile if specified
    if config_path or profile:
        try:
            config = AgentConfig.load_config(profile=profile, config_path=config_path)
        except FileNotFoundError as e:
            raise click.ClickException(str(e))
    else:
        config = AgentConfig()

    # Apply CLI options
    config.output_format = format
    config.include_figures = not no_figures
    config.save_figures = save_figures
    config.parallel_pages = workers
    config.parallel_figures = max(1, workers // 2)
    config.render_dpi = render_dpi

    # HPC mode configuration
    if hpc:
        config.hpc.enabled = True
        if vllm_url:
            config.hpc.vllm_url = vllm_url
            config.vllm.base_url = vllm_url
            config.deepseek_vllm.base_url = vllm_url
        config.audit.enabled = False

    # Apply timeout to all engine configs
    config.nougat.timeout = timeout
    config.deepseek.timeout = timeout
    config.mistral.timeout = timeout
    config.gemini.timeout = timeout
    config.vllm.timeout = timeout
    config.figure_timeout = timeout
    # Also set engine-level figure timeouts (used by describe_figure)
    config.nougat.figure_timeout = timeout
    config.deepseek.figure_timeout = timeout
    config.mistral.figure_timeout = timeout
    config.gemini.figure_timeout = timeout
    config.vllm.figure_timeout = timeout

    if output_dir:
        config.output_dir = output_dir
    if no_audit:
        config.audit.enabled = False
    if primary:
        config.primary_engine = EngineType(primary)
        config.use_primary_override = True
    if figures_engine:
        config.figures_engine = EngineType(figures_engine)
        config.use_figures_engine_override = True

    # Select pipeline based on mode
    if config.hpc.enabled:
        pipeline = HPCPipeline(config)
    else:
        pipeline = OCRPipeline(config)

    results: list[tuple[Path, bool, str]] = []

    for i, pdf_path in enumerate(pdf_files, 1):
        console.print(f"\n[dim][{i}/{len(pdf_files)}][/dim] {pdf_path.name}")
        try:
            result = pipeline.process(pdf_path)
            output_path = pipeline.save_output(result)
            results.append((pdf_path, True, str(output_path)))
            console.print(f"  [success]\\[+][/success] {output_path}")
        except KeyboardInterrupt:
            console.print("\n[warning]\\[!] cancelled[/warning]")
            break
        except Exception as e:
            results.append((pdf_path, False, str(e)))
            console.print(f"  [error]\\[x][/error] {e}")

    # Summary
    success = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - success
    console.print(f"\n[bold]Summary:[/bold] {success} succeeded, {failed} failed")


@cli.command()
def engines() -> None:
    """Show available OCR engines and their status."""
    console.print("\n[header]engines[/header]\n")

    from smart_ocr.engines import (
        DeepSeekEngine,
        DeepSeekVLLMEngine,
        GeminiEngine,
        MistralEngine,
        NougatEngine,
        VLLMEngine,
    )

    engines_info = [
        ("nougat", NougatEngine(), "local, academic papers"),
        ("deepseek", DeepSeekEngine(), "local via ollama, general"),
        ("deepseek-vllm", DeepSeekVLLMEngine(), "local via vLLM, HPC mode OCR"),
        ("vllm", VLLMEngine(), "local, figures only (Qwen2-VL/InternVL2)"),
        ("mistral", MistralEngine(), "cloud, $0.001/page"),
        ("gemini", GeminiEngine(), "cloud, $0.0002/page"),
    ]

    for name, engine, desc in engines_info:
        label = ENGINE_LABELS.get(name, name)
        available = engine.is_available()

        status = "+" if available else "x"
        style = "success" if available else "error"

        console.print(f"  [{style}]\\[{status}][/{style}] [{name}]{label}[/{name}] [dim]{desc}[/dim]")


@cli.command()
@click.option(
    "--ollama-host",
    default="http://localhost:11434",
    help="Ollama server URL",
)
def audit_status(ollama_host: str) -> None:
    """Check quality audit system status."""
    console.print("\n[header]audit[/header]\n")

    from smart_ocr.audit.llm_audit import LLMAuditor

    auditor = LLMAuditor(ollama_host=ollama_host)
    ollama_ok = auditor.is_available()

    status = "+" if ollama_ok else "x"
    style = "success" if ollama_ok else "error"
    
    console.print(f"  [{style}]\\[{status}][/{style}] [ollama]ollama[/ollama] [dim]{ollama_host}[/dim]")

    if ollama_ok:
        console.print(f"      [dim]model: {auditor.model}[/dim]")
        console.print("\n  [success]ready[/success]")
    else:
        console.print("\n  [warning]heuristics only (start ollama for llm audit)[/warning]")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--engine",
    type=click.Choice(["nougat", "deepseek", "mistral", "gemini"]),
    default="gemini",
    help="Engine to use for description",
)
def describe_figures(pdf_path: Path, engine: str) -> None:
    """Extract and describe figures from a PDF.

    Uses vision-capable models to generate descriptions
    for charts, tables, and diagrams.
    """
    console.print(f"\n{pdf_path.name}\n")
    console.print("[warning][!] experimental[/warning]")
    console.print("[dim]use `smart-ocr process` with figures enabled[/dim]")


# Shorthand aliases
@cli.command("p")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def process_shorthand(ctx: click.Context, pdf_path: Path) -> None:
    """Shorthand for 'process' command."""
    ctx.invoke(process, pdf_path=pdf_path)


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
