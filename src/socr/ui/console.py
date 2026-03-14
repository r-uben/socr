"""Minimal console interface for socr."""

from rich.console import Console
from rich.text import Text

from socr import __version__
from socr.ui.theme import AGENT_THEME, ENGINE_LABELS, STATUS_ICONS


class AgentConsole:
    """Minimal terminal interface for socr."""

    def __init__(self, verbose: bool = False):
        self.console = Console(theme=AGENT_THEME)
        self.verbose = verbose

    def print_header(self) -> None:
        """Print minimal header."""
        self.console.print()
        self.console.print(f"[dim]socr v{__version__}[/dim]")
        self.console.print()

    def print_document_info(
        self,
        filename: str,
        pages: int,
        size_mb: float,
        doc_type: str | None = None,
        detected_features: list[str] | None = None,
    ) -> None:
        """Print document information."""
        self.console.print(f"[header]{filename}[/header]")
        self.console.print(f"[dim]{pages} pages, {size_mb:.1f} MB[/dim]")
        if doc_type:
            self.console.print(f"[dim]type: {doc_type}[/dim]")
        self.console.print()

    def print_stage_header(self, stage_num: int, title: str, subtitle: str = "") -> None:
        """Print a minimal stage header."""
        self.console.print()
        self.console.print(f"[dim]({stage_num})[/dim] [header]{title.lower()}[/header]")

    def print_engine_active(self, engine: str, description: str = "") -> None:
        """Print which engine is active."""
        label = ENGINE_LABELS.get(engine, engine)
        line = f"    [{engine}]{label}[/{engine}]"
        if description:
            line += f" [dim]{description}[/dim]"
        self.console.print(line)

    def print_page_result(
        self,
        page: int,
        status: str,
        message: str = "",
        confidence: float | None = None,
    ) -> None:
        """Print result for a single page."""
        icon = STATUS_ICONS.get(status, ".")
        
        line = Text("    ")
        line.append(f"[{icon}] ", style=status)
        line.append(f"page {page}", style="" if status == "success" else "bold")
        
        if confidence is not None and confidence < 0.8:
            line.append(f" ({confidence:.0%})", style="warning")
        
        if message:
            line.append(f" {message}", style="dim")
        
        self.console.print(line)

    def print_audit_result(
        self,
        metric: str,
        value: str,
        status: str = "info",
    ) -> None:
        """Print an audit metric result."""
        icon = STATUS_ICONS.get(status, ".")
        self.console.print(f"    [{icon}] {metric}: [{status}]{value}[/{status}]")

    def print_cost(self, amount: float, description: str = "") -> None:
        """Print cost information."""
        if amount > 0:
            self.console.print(f"    [dim]cost: ${amount:.4f}[/dim]")

    def print_figure_result(
        self,
        figure_num: int,
        page: int,
        fig_type: str,
        description: str,
    ) -> None:
        """Print result for a processed figure."""
        short_desc = description[:40] + "..." if len(description) > 40 else description
        self.console.print(f"    [+] fig {figure_num} (p.{page}): [info]{fig_type}[/info] [dim]{short_desc}[/dim]")

    def print_summary(
        self,
        pages_success: int,
        pages_total: int,
        figures_count: int,
        time_seconds: float,
        cost: float,
        engines_used: dict[str, int],
        output_path: str,
    ) -> None:
        """Print the final summary."""
        self.console.print()
        self.console.print("[dim]---[/dim]")
        self.console.print()
        
        # Stats
        self.console.print(f"[success]done[/success] {pages_success}/{pages_total} pages")
        
        if figures_count > 0:
            self.console.print(f"     {figures_count} figures")
        
        self.console.print(f"     {time_seconds:.1f}s")
        
        if cost > 0:
            self.console.print(f"     ${cost:.4f}")
        
        # Engines
        engine_parts = [f"{ENGINE_LABELS.get(e, e)} ({c})" for e, c in engines_used.items()]
        self.console.print(f"[dim]     {' + '.join(engine_parts)}[/dim]")
        
        self.console.print()
        self.console.print(f"[dim]->[/dim] {output_path}")
        self.console.print()

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[error][x] {message}[/error]")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[warning][!] {message}[/warning]")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        if self.verbose:
            self.console.print(f"[dim]    {message}[/dim]")

    def rule(self, title: str = "") -> None:
        """Print a subtle divider."""
        self.console.print("[dim]---[/dim]")
