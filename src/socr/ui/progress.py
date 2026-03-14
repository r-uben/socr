"""Progress display components for socr."""

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from socr.ui.theme import (
    AGENT_THEME,
    ENGINE_ICONS,
    ENGINE_LABELS,
    STATUS_ICONS,
)


class AgentProgress:
    """Rich progress display for multi-agent OCR processing."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console(theme=AGENT_THEME)
        self._progress: Progress | None = None
        self._live: Live | None = None
        self._current_task: TaskID | None = None

    def create_progress(self, description: str = "Processing") -> Progress:
        """Create a styled progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40, complete_style="bright_blue", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    @contextmanager
    def stage_progress(
        self,
        stage_name: str,
        engine: str,
        total: int,
        description: str = "",
    ) -> Generator["StageProgressContext", None, None]:
        """Context manager for stage progress with live updates."""
        icon = ENGINE_ICONS.get(engine, "⚙")
        label = ENGINE_LABELS.get(engine, engine)

        progress = self.create_progress()

        # Create header
        header = Text()
        header.append(f"{icon} ", style="bold")
        header.append(label, style=engine)
        if description:
            header.append(f" {description}", style="dim")

        task_id = progress.add_task(str(header), total=total)

        ctx = StageProgressContext(progress, task_id, self.console)

        with progress:
            yield ctx


class StageProgressContext:
    """Context for updating stage progress."""

    def __init__(self, progress: Progress, task_id: TaskID, console: Console):
        self.progress = progress
        self.task_id = task_id
        self.console = console
        self._results: list[dict] = []

    def advance(self, amount: int = 1) -> None:
        """Advance the progress bar."""
        self.progress.advance(self.task_id, amount)

    def update(self, description: str | None = None, completed: int | None = None) -> None:
        """Update progress state."""
        kwargs = {}
        if description:
            kwargs["description"] = description
        if completed is not None:
            kwargs["completed"] = completed
        self.progress.update(self.task_id, **kwargs)

    def add_result(
        self,
        item: int | str,
        status: str,
        message: str = "",
        confidence: float | None = None,
    ) -> None:
        """Record a result (displayed after progress completes)."""
        self._results.append({
            "item": item,
            "status": status,
            "message": message,
            "confidence": confidence,
        })

    def print_results(self, show_all: bool = False) -> None:
        """Print recorded results."""
        # Group by status
        successes = [r for r in self._results if r["status"] == "success"]
        warnings = [r for r in self._results if r["status"] == "warning"]
        errors = [r for r in self._results if r["status"] == "error"]

        # Print summary for successes if too many
        if len(successes) > 5 and not show_all:
            items = [str(r["item"]) for r in successes[:3]]
            self.console.print(
                f"   {STATUS_ICONS['success']} Items {', '.join(items)}... "
                f"[success](+{len(successes) - 3} more)[/success]"
            )
        elif successes:
            for r in successes:
                self._print_result(r)

        # Always print warnings and errors
        for r in warnings:
            self._print_result(r)
        for r in errors:
            self._print_result(r)

    def _print_result(self, result: dict) -> None:
        """Print a single result line."""
        icon = STATUS_ICONS.get(result["status"], "?")
        style = result["status"]

        line = Text("   ")
        line.append(f"{icon} ", style=style)
        line.append(f"Item {result['item']}", style="bold" if result["status"] != "success" else "")

        if result["confidence"] is not None:
            conf = result["confidence"]
            conf_style = "success" if conf >= 0.8 else "warning" if conf >= 0.6 else "error"
            line.append(f" ({conf:.0%})", style=conf_style)

        if result["message"]:
            line.append(f" - {result['message']}", style="dim")

        self.console.print(line)


class MultiEngineProgress:
    """Display progress across multiple engines simultaneously."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console(theme=AGENT_THEME)
        self.engines: dict[str, dict] = {}

    def add_engine(
        self,
        engine: str,
        total: int,
        status: str = "pending",
    ) -> None:
        """Add an engine to track."""
        self.engines[engine] = {
            "total": total,
            "completed": 0,
            "status": status,
            "message": "",
        }

    def update_engine(
        self,
        engine: str,
        completed: int | None = None,
        status: str | None = None,
        message: str | None = None,
    ) -> None:
        """Update engine progress."""
        if engine in self.engines:
            if completed is not None:
                self.engines[engine]["completed"] = completed
            if status is not None:
                self.engines[engine]["status"] = status
            if message is not None:
                self.engines[engine]["message"] = message

    def render(self) -> Table:
        """Render the multi-engine progress table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Engine", style="bold")
        table.add_column("Progress", width=30)
        table.add_column("Status")

        for engine, data in self.engines.items():
            icon = ENGINE_ICONS.get(engine, "⚙")
            label = ENGINE_LABELS.get(engine, engine)
            status_icon = STATUS_ICONS.get(data["status"], "○")

            # Build progress bar
            pct = data["completed"] / max(data["total"], 1)
            filled = int(pct * 20)
            bar = "█" * filled + "░" * (20 - filled)

            engine_cell = Text(f"{icon} {label}")
            engine_cell.stylize(engine)

            progress_cell = Text()
            progress_cell.append(bar, style="bright_blue" if data["status"] == "running" else "dim")
            progress_cell.append(f" {data['completed']}/{data['total']}", style="dim")

            status_cell = Text(f"{status_icon} ")
            status_cell.append(data["message"] or data["status"], style=data["status"])

            table.add_row(engine_cell, progress_cell, status_cell)

        return table
