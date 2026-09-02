"""CLI for socr — Multi-Engine Document Processing."""

from pathlib import Path

import click
from rich.console import Console

from socr import __version__
from socr.core.config import EngineType, PipelineConfig
from socr.review import html as review_html

console = Console()

ENGINE_CHOICES = [
    e.value for e in EngineType if e not in (EngineType.DEEPSEEK_VLLM, EngineType.VLLM)
]
# "auto" probes CLI engines in priority order


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
    f = click.option(
        "--no-audit",
        is_flag=True,
        help=(
            "REMOVED — rejected with an error. This flag controlled no path "
            "(GH-139); for the HPC lane use 'hpc.audit_enabled' in a config file."
        ),
    )(f)
    f = click.option(
        "--no-judge-hard-pages", is_flag=True, help="Disable VLM judge on hard pages (tables/math)"
    )(f)
    f = click.option(
        "--no-dual-pass-tables",
        is_flag=True,
        help="Disable dual-pass table extraction (crop + re-read located tables)",
    )(f)
    f = click.option(
        "--auto-patch-tables",
        is_flag=True,
        help="Let dual-pass auto-patch crop readings into the page (default: flag-only,"
        " never edits)",
    )(f)
    f = click.option(
        "--no-native-first", is_flag=True, help="Disable native-first: run VLM on all pages"
    )(f)
    f = click.option(
        "--native-only",
        is_flag=True,
        help=(
            "Trust the native text layer for ALL born-digital pages — never OCR-enhance them, "
            "even when the layer has known deficiencies (e.g. corrupt-math regions). "
            "Genuine scans still route to OCR normally. "
            "Figure extraction can still run without triggering whole-page OCR. "
            "Incompatible with --no-native-first (that flag wins if both are set)."
        ),
    )(f)
    f = click.option(
        "--recover-corrupt-math",
        is_flag=True,
        help=(
            "Keep native prose and re-read only positively detected corrupt-font equation "
            "crops with the --math-model endpoint. Retains each crop as ground truth and "
            "marks syntax-valid LaTeX as non-authoritative; the page remains WARNING."
        ),
    )(f)
    f = click.option(
        "--detect-equations",
        is_flag=True,
        help=(
            "GH-36a: detect display-equation regions on born-digital pages (model-free). "
            "Saves crop PNGs and records provenance; does NOT splice LaTeX (that is GH-36b). "
            "Default off; throughput and quality must be validated before enabling by default."
        ),
    )(f)
    f = click.option(
        "--recover-clean-equations",
        is_flag=True,
        help=(
            "GH-36b: read detected equation crops with the local VLM, validate with pylatexenc "
            "(1A structural gate), and attach 1A-validated LaTeX adjacently to the inlined crop "
            "(1C non-destructive sidecar). Requires --detect-equations. "
            "Bad/hallucinated LaTeX never replaces native text or the crop. "
            "Default off — enable only after throughput is measured on a real corpus "
            "(per consilium 20260615T210537Z-6621)."
        ),
    )(f)
    f = click.option(
        "--clean-equation-model",
        default=None,
        help=(
            "GH-36b: local Ollama vision model for clean-equation crop → LaTeX "
            "(default: qwen3-vl:30b-a3b-instruct — the validated local instruct VLM). "
            "Never use :8b or the non-instruct :30b. "
            "Cloud opt-in: pass e.g. qwen3.5:cloud explicitly."
        ),
    )(f)
    f = click.option(
        "--math-model",
        default=None,
        help=(
            "Ollama-compatible vision model for corrupt-font equation-crop recovery "
            "(default: qwen3.5:cloud; use qwen3-vl:30b-a3b-instruct for offline). "
            "This is a region-only math path, not the whole-page Qwen OCR tier."
        ),
    )(f)
    f = click.option("--timeout", type=int, default=1800, help="Subprocess timeout in seconds")(f)
    f = click.option(
        "--dpi",
        type=int,
        default=None,
        help="Page render DPI for OCR engines (default 200; higher helps local VLMs)",
    )(f)
    f = click.option(
        "--qwen-backend",
        type=click.Choice(["auto", "ollama", "vllm", "api"]),
        default=None,
        help="Backend for the qwen engine",
    )(f)
    f = click.option(
        "--qwen-vllm-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL of the vLLM server for the agentic VLM "
        "path (HPC), e.g. http://node07:8000/v1. Used when --qwen-backend vllm.",
    )(f)
    f = click.option(
        "--qwen-vllm-model",
        type=str,
        default=None,
        help="HF model id served by vLLM for the agentic VLM path "
        "(default Qwen/Qwen3-VL-30B-A3B-Instruct). Used when --qwen-backend vllm.",
    )(f)
    f = click.option(
        "--qwen-model",
        type=str,
        default=None,
        help="Qwen model override (e.g. qwen3.5:27b local, qwen3.5:cloud)",
    )(f)
    f = click.option(
        "--save-figures",
        is_flag=True,
        help=(
            "Extract figures and write PNG files with image references. "
            "Does NOT generate VLM caption prose — use --describe-figures for that."
        ),
    )(f)
    f = click.option(
        "--describe-figures",
        is_flag=True,
        help=(
            "Run VLM captions on extracted figures (opt-in; implies --save-figures). "
            "Caption failures never overwrite already-written OCR text."
        ),
    )(f)
    f = click.option("--reprocess", is_flag=True, help="Reprocess already-processed files")(f)
    f = click.option("--dry-run", is_flag=True, help="List files without processing")(f)
    f = click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")(f)
    f = click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")(f)
    f = click.option(
        "--config",
        "config_path",
        type=click.Path(exists=True, path_type=Path),
        help="YAML config file",
    )(f)
    f = click.option("--profile", type=str, help="Load ~/.config/socr/{profile}.yaml")(f)
    # Agentic cost-aware routing is the sole default product path.
    f = click.option(
        "--agentic",
        is_flag=True,
        help="Use cost-aware routing (default; retained for command compatibility)",
    )(f)
    f = click.option(
        "--strict-local",
        is_flag=True,
        help="Agentic routing: only local/free rungs, no cloud escalation",
    )(f)
    f = click.option(
        "--judge-backend",
        type=click.Choice(["auto", "vlm", "heuristic"]),
        default="auto",
        help="Quality judge for agentic routing",
    )(f)
    f = click.option(
        "--judge-model",
        type=str,
        default="",
        help="VLM model for the judge (e.g. qwen2-vl:7b)",
    )(f)
    f = click.option(
        "--max-cost-per-page",
        type=float,
        default=0.0,
        help="Skip providers above this $/page (0=no cap)",
    )(f)
    f = click.option(
        "--cost-budget",
        type=float,
        default=0.0,
        help="Stop escalating once doc spend hits this $ (0=unlimited)",
    )(f)
    f = click.option(
        "--write-manifest",
        is_flag=True,
        help="Write a replayable manifest + blob cache",
    )(f)
    f = click.option(
        "--table-judge-ladder",
        is_flag=True,
        help=(
            "GH-353: gate emitted table pages behind a two-rung acceptance judge "
            "(ollama-cloud glm-5.3-flash, then the gemini CLI) before shipping. "
            "Default off. --strict-local + this flag makes both rungs unavailable, "
            "so every table page is demoted to UNVERIFIED rather than shipped "
            "unjudged."
        ),
    )(f)
    return f


def _explicitly_given(name: str) -> bool:
    """Whether the user actually typed this option, rather than Click defaulting it.

    GH-168. `build_config` assigned several options unconditionally, so a value
    loaded from `--config`/`--profile` was overwritten by the CLI's own default
    even when the user never mentioned the option. Measured before this fix, a
    config file setting all seven lost every one:

        cost_budget 5.0 -> 0.0, max_cost_per_page 0.25 -> 0.0,
        write_manifest True -> False, timeout 999 -> 1800,
        judge_backend heuristic -> auto, judge_model my-judge -> "",
        save_figures True -> False

    A silently ignored setting is the "flag that lies" failure #142 names: the
    user believes a budget is in force and scripts around it.

    Click records where each parameter's value came from, which is the only
    signal that distinguishes "not supplied" from "supplied with the value that
    happens to be the default". Outside a Click context -- a direct call from a
    test or library code -- the caller passed the argument deliberately, so it
    is treated as explicit and behaviour is unchanged.
    """
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return True
    try:
        from click.core import ParameterSource

        return ctx.get_parameter_source(name) not in (None, ParameterSource.DEFAULT)
    except Exception:
        return True


def build_config(
    primary: str | None = None,
    fallback: str | None = None,
    no_audit: bool = False,
    no_judge_hard_pages: bool = False,
    no_dual_pass_tables: bool = False,
    auto_patch_tables: bool = False,
    no_native_first: bool = False,
    native_only: bool = False,
    recover_corrupt_math: bool = False,
    detect_equations: bool = False,
    recover_clean_equations: bool = False,
    clean_equation_model: str | None = None,
    math_model: str | None = None,
    timeout: int = 1800,
    dpi: int | None = None,
    qwen_backend: str | None = None,
    qwen_vllm_url: str | None = None,
    qwen_vllm_model: str | None = None,
    qwen_model: str | None = None,
    save_figures: bool = False,
    describe_figures: bool = False,
    reprocess: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    config_path: Path | None = None,
    profile: str | None = None,
    output_dir: Path | None = None,
    agentic: bool = False,
    strict_local: bool = False,
    judge_backend: str = "auto",
    judge_model: str = "",
    max_cost_per_page: float = 0.0,
    cost_budget: float = 0.0,
    write_manifest: bool = False,
    table_judge_ladder: bool = False,
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
        # GH-139. `--no-audit` advertised "skip quality audit stage" and set
        # `audit_enabled=False`, but every consumer of that field is gone: the four
        # gates the issue cited (multi-engine scoring, single-engine scoring, the
        # hard-page judge, repair) lived in the legacy branches that #298 deleted.
        # `PipelineConfig.audit_enabled` has since been DELETED outright, so there
        # is no longer even a field to set.
        #
        # So the flag does not merely fail on the agentic path, as the issue
        # originally framed it: it is inert in EVERY mode -- including HPC, which
        # `@common_options` also exposes it on. HPC's audit IS separable, but it is
        # gated by `config.hpc.audit_enabled` (hpc_pipeline.py:199), a different
        # field this flag never wrote to. Silently accepting it
        # would leave a user believing a constraint is in force and scripting
        # around it, which is the exact failure #142 calls "a flag that lies is
        # worse than a missing flag".
        #
        # Resolution 1 of the issue's own preference order: reject it, loudly.
        # The flag is kept (rather than deleted) so existing scripts get this
        # explanation instead of click's bare "no such option".
        raise click.UsageError(
            "--no-audit no longer does anything and has been rejected rather than "
            "silently ignored (GH-139).\n"
            "\n"
            "It set PipelineConfig.audit_enabled, a field with no remaining "
            "consumer: the gates that once read it (multi-engine scoring, "
            "single-engine scoring, the hard-page judge, repair) were removed in "
            "#298, and the field itself is now deleted.\n"
            "\n"
            "In agentic mode -- the default -- there is no audit stage to skip: the "
            "judge IS the routing algorithm. Escalation happens because a judge "
            "rejected a rung, so with no gate there is no accept/escalate signal "
            "for the ladder to act on.\n"
            "\n"
            "The HPC lane does have a separable audit stage, gated by the DIFFERENT "
            "setting 'hpc.audit_enabled' -- which this flag never wrote to. Set it "
            "in a config file if that is what you wanted.\n"
            "\n"
            "To reduce model spend, use --strict-local (no cloud egress), "
            "--max-cost-per-page, or --cost-budget."
        )
    if no_judge_hard_pages:
        config.judge_hard_pages = False
    if no_dual_pass_tables:
        config.dual_pass_tables = False
    if auto_patch_tables:
        config.auto_patch_tables = True
    if no_native_first:
        config.native_first = False
    if native_only and no_native_first:
        # Incoherent combination: --no-native-first forces OCR on all pages,
        # so --native-only has no effect. Warn and let --no-native-first win.
        console.print(
            "[yellow]Warning:[/yellow] --native-only and --no-native-first are incompatible. "
            "--no-native-first takes precedence (all pages sent to OCR)."
        )
    elif native_only:
        config.native_only = True
    if recover_corrupt_math:
        config.recover_corrupt_math = True
    if detect_equations:
        config.detect_equations = True
    if recover_clean_equations:
        config.recover_clean_equations = True
    if clean_equation_model is not None:
        config.clean_equation_model = clean_equation_model
    if math_model is not None:
        config.math_model = math_model

    if _explicitly_given("timeout"):
        config.timeout = timeout
    if dpi is not None:
        config.render_dpi = dpi
    if qwen_backend is not None:
        config.qwen_backend = qwen_backend
    if qwen_vllm_url is not None:
        config.qwen_vllm_url = qwen_vllm_url
    if qwen_vllm_model is not None:
        config.qwen_vllm_model = qwen_vllm_model
    if qwen_model is not None:
        config.qwen_model = qwen_model
        config.qwen_model_pinned = True
    # --describe-figures implies --save-figures: captions require PNGs on disk.
    if describe_figures:
        config.save_figures = True
        config.describe_figures = True
    else:
        # GH-168: only overwrite what the user actually asked for. `--describe-figures`
        # implies `save_figures`, so its absence must not silently clear a
        # `save_figures: true` loaded from a config file -- nor turn
        # `describe_figures` off when the file asked for it and the flag was not
        # typed either way.
        if _explicitly_given("save_figures"):
            config.save_figures = save_figures
        if _explicitly_given("describe_figures"):
            config.describe_figures = False
    config.reprocess = reprocess
    config.dry_run = dry_run
    config.quiet = quiet
    config.verbose = verbose

    # Agentic cost-aware routing is the sole default; --agentic is a backward-compatibility no-op.
    # The config default (True from PipelineConfig or loaded YAML) is always used.
    if strict_local:
        config.strict_local = True
    if _explicitly_given("judge_backend"):
        config.judge_backend = judge_backend
    if _explicitly_given("judge_model"):
        config.judge_model = judge_model
    if _explicitly_given("max_cost_per_page"):
        config.max_cost_per_page = max_cost_per_page
    if _explicitly_given("cost_budget"):
        config.cost_budget = cost_budget
    if _explicitly_given("write_manifest"):
        config.write_manifest = write_manifest
    # is_flag default is False, matching PipelineConfig's own default — only ever
    # flip it on, never clobber a YAML-config True with an unset CLI flag (the
    # cli.py:371-area unconditional-override trap this ticket calls out for
    # judge_backend/judge_model/max_cost_per_page/cost_budget/write_manifest
    # above, which DOES clobber YAML because those options carry non-None
    # defaults of their own).
    if table_judge_ladder:
        config.table_judge_ladder = True

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
@click.option("--unified", is_flag=True, help="Agentic orchestrator (compatibility)")
@common_options
def process(
    pdf_path: Path,
    output_dir: Path | None,
    hpc_sequential: bool = False,
    unified: bool = False,
    **kwargs,
) -> None:
    """Process a single PDF document.

    Uses cost-aware agentic routing: the judge accepts each page or escalates
    through the provider ladder.

    Example:
        socr process paper.pdf -o ./results/
        socr paper.pdf --primary gemini --quiet
        socr paper.pdf --hpc-sequential --save-figures
        socr paper.pdf --unified
    """
    config = build_config(output_dir=output_dir, **kwargs)

    # GH-368: --dry-run was consulted only inside process_batch, so
    # `socr process <pdf> --dry-run` ran the full real pipeline -- a supposedly
    # dry test OCR'd a PDF for ~56s. Silently ignoring a flag the user typed
    # breaks the no-silent-failure rule this repo holds everywhere else.
    #
    # Placed before AUTO-engine resolution deliberately: that probes the
    # installed engines, and a dry run should not touch the machine at all.
    if config.dry_run:
        if not config.quiet:
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            console.print("[blue]Would process 1 file:[/blue]")
            console.print(f"  {pdf_path.name} ({size_mb:.1f} MB)")
            # GH-401 review, second pass. Report the destination the REAL run
            # would use, by calling the pipeline's own resolver rather than
            # re-deriving it here. `-o` is verbatim; a user-set non-sentinel
            # config.output_dir is verbatim; otherwise it is <input-parent>/ocr/,
            # NOT the legacy `output` sentinel.
            #
            # My first pass swapped "None" for that sentinel and was still
            # wrong: a dry run that misdescribes the run it previews is the same
            # failure as the flag being inert, one step smaller.
            from socr.pipeline.orchestrator import UnifiedPipeline as _Pipe

            # pdf_path itself, exactly as process() passes it (orchestrator.py:643).
            # Passing .parent diverges whenever pdf_path is a directory: the
            # preview would resolve <parent>/ocr while the run resolves
            # <directory>/ocr. Same resolver AND same argument, or the two can
            # still disagree.
            resolved_out = _Pipe(config)._resolve_output_root(pdf_path, output_dir)
            # soft_wrap: a long temp path wrapped mid-string otherwise, which
            # breaks anything reading the destination back out of the output.
            console.print(f"[blue]Output:[/blue] {resolved_out}", soft_wrap=True)
        return

    # Resolve AUTO engine early so we can route to the right pipeline
    if config.primary_engine == EngineType.AUTO:
        from socr.engines.registry import resolve_auto_engine

        config.primary_engine = resolve_auto_engine()
        if not config.quiet:
            console.print(f"[dim]Auto-selected engine: {config.primary_engine.value}[/dim]")

    if hpc_sequential:
        from socr.pipeline.hpc_pipeline import HPCPipeline

        config.hpc.enabled = True
        config.hpc.sequential = True
        pipeline = HPCPipeline(config)
    else:
        # UnifiedPipeline is the sole orchestrator: analyze -> agentic -> assemble.
        # --unified is kept as a backwards-compatible no-op.
        from socr.pipeline.orchestrator import UnifiedPipeline

        pipeline = UnifiedPipeline(config)

    try:
        result = pipeline.process(pdf_path, output_dir)
        if not result.success:
            from socr.core.result import DocumentStatus

            # A complete markdown was written but some pages failed audit (e.g.
            # local-only run where cloud escalation is unavailable). This is a
            # partial success, not a hard failure: keep the usable output and exit
            # 0 with a warning rather than aborting. Only a true ERROR (no text
            # produced) or ERASED CONTENT (pages with no usable output at all)
            # is fatal — a batch pipeline must not record a run that lost pages
            # as a success.
            if result.status == DocumentStatus.AUDIT_FAILED:
                from socr.core.result import LOST_CONTENT_NOTE

                if result.error and LOST_CONTENT_NOTE in result.error:
                    raise click.ClickException(
                        f"Completed but lost content: {result.error}. "
                        "Output written; see audit_log.json."
                    )
                console.print(
                    "[yellow]Completed with warnings:[/yellow] some pages failed "
                    "audit; output written. See audit_log.json."
                )
            else:
                raise click.ClickException(f"Processing failed: {result.error}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        raise click.Abort()


@cli.command()
@click.argument("pdf_dir", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), help="Output directory")
@click.option("--limit", type=int, help="Maximum number of PDFs to process")
@click.option("--unified", is_flag=True, help="Agentic orchestrator (compatibility)")
@common_options
def batch(
    pdf_dir: Path,
    output_dir: Path | None,
    limit: int | None,
    unified: bool = False,
    **kwargs,
) -> None:
    """Process all PDFs in a directory.

    Supports incremental processing — unchanged files are skipped
    (use --reprocess to force).

    Example:
        socr batch ~/Papers/ -o ./results/
        socr batch ~/Papers/ --dry-run
        socr batch ~/Papers/ --unified
    """
    config = build_config(output_dir=output_dir, **kwargs)

    # Resolve AUTO engine
    if config.primary_engine == EngineType.AUTO:
        from socr.engines.registry import resolve_auto_engine

        config.primary_engine = resolve_auto_engine()
        if not config.quiet:
            console.print(f"[dim]Auto-selected engine: {config.primary_engine.value}[/dim]")

    # UnifiedPipeline is the sole orchestrator: analyze -> agentic -> assemble.
    # --unified is a no-op kept for backwards compatibility.
    from socr.pipeline.orchestrator import UnifiedPipeline

    pipeline = UnifiedPipeline(config)

    # Handle --limit by pre-filtering
    if limit:
        pdfs = sorted(pdf_dir.glob("*.pdf"))[:limit]
        if not pdfs:
            console.print("[yellow]No PDF files found[/yellow]")
            return
        import tempfile

        # DATA-LOSS FIX (round-3 HIGH): the limited PDFs are symlinked into a
        # TemporaryDirectory that is destroyed on block exit. If we passed
        # output_dir=None, process_batch would resolve the output root relative
        # to that tmpdir (<tmpdir>/ocr) and ALL output would be deleted on exit.
        # Resolve the REAL, persistent output root from the ORIGINAL pdf_dir
        # FIRST (honoring -o and any non-default configured output_dir via the
        # pipeline's own resolver) and pass it as a concrete dir, so output is
        # written next to the real input, never into the ephemeral tmpdir.
        persistent_out = pipeline._resolve_output_root(pdf_dir, output_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            limited_dir = Path(tmpdir)
            for pdf in pdfs:
                (limited_dir / pdf.name).symlink_to(pdf)
            pipeline.process_batch(limited_dir, persistent_out)
    else:
        pipeline.process_batch(pdf_dir, output_dir)

    # Canon uniform exit policy: nonzero if ANY file failed or was partial.
    # The batch path previously always exited 0 even on total failure.
    if pipeline.last_outcome.exit_code != 0:
        raise SystemExit(pipeline.last_outcome.exit_code)


@cli.command()
def engines() -> None:
    """Show available OCR engines and their status."""
    from socr.engines.registry import get_engine, resolve_auto_engine

    console.print("\n[bold]Engines[/bold]\n")

    engine_info = [
        (EngineType.QWEN, "local via Ollama/vLLM or cloud API (Qwen-VL, best open OCR)"),
        (EngineType.GLM, "local via Ollama (0.9B, ~10s/page)"),
        (EngineType.NOUGAT, "local, academic papers"),
        (EngineType.DEEPSEEK, "local via Ollama"),
        (EngineType.MARKER, "local, layout-aware (Surya + Texify)"),
        (EngineType.GEMINI, "cloud, ~$0.0002/page (CLI)"),
        (EngineType.MISTRAL, "cloud, ~$0.001/page"),
    ]

    for engine_type, desc in engine_info:
        engine = get_engine(engine_type)
        available = engine.is_available()
        status = "[green]+[/green]" if available else "[red]x[/red]"
        console.print(f"  [{status}] {engine_type.value:<12} [dim]{desc}[/dim]")

    # Show what auto would select
    auto_choice = resolve_auto_engine()
    console.print(f"\n  [bold]auto[/bold] would select: [cyan]{auto_choice.value}[/cyan]")


@cli.command()
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    help="Blob cache directory (default: <manifest dir>/cache)",
)
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), help="Write markdown here (default: stdout)"
)
def replay(manifest_path: Path, cache_dir: Path | None, output: Path | None) -> None:
    """Rebuild a document from a manifest + cache — NO engine calls.

    Reproducible reconstruction for a citable corpus: given the manifest written
    by an earlier run and the content-addressed blob cache, reassemble the exact
    markdown without invoking any OCR engine or model. Safe to run headless/HPC.
    """
    from socr.core.cache import BlobStore
    from socr.core.manifest import Manifest, stale_pages
    from socr.core.manifest import replay as do_replay

    cache_dir = cache_dir or manifest_path.parent / "cache"
    manifest = Manifest.load(manifest_path)
    store = BlobStore(cache_dir)

    missing = stale_pages(manifest, store)
    if missing:
        raise click.ClickException(
            f"cache at {cache_dir} is missing blobs for pages {missing}; "
            f"re-run `socr agent` to regenerate them."
        )

    markdown = do_replay(manifest, store)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        console.print(
            f"[green]Replayed[/green] {manifest.pdf_filename} "
            f"({manifest.page_count} pages) -> {output} [dim](0 model calls)[/dim]"
        )
    else:
        click.echo(markdown)


@cli.command("judge-benchmark")
@click.argument("dataset_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--model", default="qwen2-vl:7b", help="Ollama vision model for the judge")
@click.option("--host", default="http://localhost:11434", help="Ollama host")
def judge_benchmark(dataset_dir: Path, model: str, host: str) -> None:
    """Measure the OCR-faithfulness judge against a labeled page set.

    DATASET_DIR must contain labels.json + referenced images/ and ocr/ files
    (see socr.judge.benchmark). Reports the two error rates that matter: false
    negatives (mangled pages let through -> corpus poisoning) and false positives
    (good pages flagged -> wasted re-OCR budget).
    """
    from socr.judge.benchmark import load_dataset, run_benchmark
    from socr.judge.ollama_judge import OllamaVisionJudge

    judge = OllamaVisionJudge(model=model, host=host)
    if not judge.is_available():
        raise click.ClickException(
            f"Ollama model {model!r} not available at {host}. "
            f"Pull it (`ollama pull {model}`) or pass --model/--host."
        )
    dataset = load_dataset(dataset_dir)
    if not dataset:
        raise click.ClickException(f"no labeled pages found in {dataset_dir}")

    console.print(f"[dim]Judging {len(dataset)} pages with {model}...[/dim]")
    report = run_benchmark(judge, dataset)
    console.print("\n[bold]Judge benchmark[/bold]")
    console.print(report.summary())


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
    from socr.benchmark.dataset import BenchmarkPaper, build_benchmark_set
    from socr.benchmark.ground_truth import GroundTruthExtractor
    from socr.benchmark.rasterize import RASTERIZE_SPECS, PaperRasterizer

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
            f"Benchmark manifest not found: {manifest}\nRun 'socr benchmark init' first."
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


@benchmark.command("binding-coverage")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Content-free self-bind coverage manifest",
)
@click.option(
    "--pdf-root",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory containing PDFs referenced by the manifest",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "summary"]),
    default="json",
    show_default=True,
    help="Output format",
)
def benchmark_binding_coverage(manifest_path: Path, pdf_root: Path, output_format: str) -> None:
    """Measure native self-binding coverage without emitting document content."""
    from socr.benchmark.binding_coverage import measure_manifest, summary_text

    if not manifest_path.exists() or not manifest_path.is_file():
        raise click.ClickException(f"manifest is not a file: {manifest_path}")
    if not pdf_root.exists() or not pdf_root.is_dir():
        raise click.ClickException(f"PDF root is not a directory: {pdf_root}")

    try:
        report = measure_manifest(manifest_path, pdf_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(report.to_json(), nl=False)
    else:
        click.echo(summary_text(report), nl=False)


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
        avg_wer = (
            sum(profile.category_wer.values()) / len(profile.category_wer)
            if profile.category_wer
            else float("nan")
        )
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
    table.add_column("Avg NES", justify="right")
    table.add_column("Avg WER", justify="right")
    table.add_column("Avg CER", justify="right")
    table.add_column("Avg Time", justify="right")

    for engine_name in sorted(by_engine):
        runs = by_engine[engine_name]
        scored = [r for r in runs if r.score is not None]
        avg_nes = sum(r.score.overall_nes for r in scored) / len(scored) if scored else float("nan")
        avg_wer = sum(r.score.overall_wer for r in scored) / len(scored) if scored else float("nan")
        avg_cer = sum(r.score.overall_cer for r in scored) / len(scored) if scored else float("nan")
        avg_time = sum(r.result.processing_time for r in runs) / len(runs) if runs else 0.0

        table.add_row(
            engine_name,
            str(len(runs)),
            str(len(scored)),
            f"{avg_nes:.3f}" if scored else "N/A",
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


@cli.command()
@click.argument("doc_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--pdf",
    "pdf_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source PDF the document directory was produced from",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output HTML path (default: <doc_dir>/review.html)",
)
@click.option(
    "--scale",
    type=float,
    default=None,
    help=f"Page render scale (default {review_html.RENDER_SCALE}; higher is sharper and larger)",
)
@click.option(
    "--quality",
    type=int,
    default=None,
    help=f"JPEG quality for page images (default {review_html.JPEG_QUALITY})",
)
def review(
    doc_dir: Path,
    pdf_path: Path,
    output: Path | None,
    scale: float | None,
    quality: int | None,
) -> None:
    """Build a side-by-side page-image/markdown page for hand judgement (GH-220)."""
    out_path = output or (doc_dir / "review.html")

    report = review_html.collect_pages(
        doc_dir,
        pdf_path,
        scale=scale if scale is not None else review_html.RENDER_SCALE,
        quality=quality if quality is not None else review_html.JPEG_QUALITY,
    )
    rendered = review_html.build_review_html(report)
    size = len(rendered.encode("utf-8"))

    if size >= review_html.WRITE_REFUSAL_FLOOR:
        # Never silently drop pages to fit -- a short document would read as a complete one.
        console.print(
            f"[red]Refusing to write:[/red] {size / 1e6:.2f} MB exceeds the "
            f"{review_html.WRITE_REFUSAL_FLOOR / 1e6:.0f} MB write floor "
            f"(host cap {review_html.ARTIFACT_BYTE_CAP / 1e6:.0f} MB).\n"
            f"Lower --scale or --quality and retry; pages are never dropped to fit."
        )
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")

    console.print(f"\n[bold]Review page[/bold] {out_path}  [dim]({size / 1e6:.2f} MB)[/dim]")
    console.print(f"  document status : [bold]{report.doc_status}[/bold]")
    console.print(f"  pages           : {len(report.pages)}")
    console.print(
        f"  pages with a recorded signal : [bold]{report.suspect_count}[/bold]"
        f"  [dim](of which {report.contradiction_count} also report success)[/dim]"
    )
    if report.untrusted_pages:
        console.print(f"  pages with untrusted tables  : {len(report.untrusted_pages)}")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
