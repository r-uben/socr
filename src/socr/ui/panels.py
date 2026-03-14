"""Minimal panel components for socr."""

from rich.console import Group, RenderableType
from rich.text import Text

from socr.ui.theme import ENGINE_LABELS, STATUS_ICONS


class StagePanel:
    """A minimal stage display."""

    def __init__(
        self,
        stage_num: int,
        title: str,
        subtitle: str = "",
        color: str | None = None,
    ):
        self.stage_num = stage_num
        self.title = title
        self.subtitle = subtitle
        self.content_lines: list[RenderableType] = []

    def add_engine_header(self, engine: str, description: str = "") -> None:
        """Add an engine header line."""
        label = ENGINE_LABELS.get(engine, engine)
        line = Text()
        line.append(f"    {label}", style=engine)
        if description:
            line.append(f" {description}", style="dim")
        self.content_lines.append(line)

    def add_progress_line(
        self,
        current: int,
        total: int,
        label: str = "",
        width: int = 20,
    ) -> None:
        """Add a minimal progress line."""
        self.content_lines.append(Text(f"    {current}/{total}", style="dim"))

    def add_result(
        self,
        item: str,
        status: str,
        message: str = "",
        confidence: float | None = None,
    ) -> None:
        """Add a result line."""
        icon = STATUS_ICONS.get(status, ".")
        line = Text()
        line.append(f"    [{icon}] ", style=status)
        line.append(item)
        if message:
            line.append(f" {message}", style="dim")
        self.content_lines.append(line)

    def add_metric(self, label: str, value: str, status: str = "info") -> None:
        """Add a metric line."""
        self.content_lines.append(Text(f"    {label}: {value}", style="dim"))

    def add_cost(self, amount: float, description: str = "") -> None:
        """Add a cost line."""
        if amount > 0:
            self.content_lines.append(Text(f"    cost: ${amount:.4f}", style="dim"))

    def add_text(self, text: str | Text) -> None:
        """Add arbitrary text."""
        if isinstance(text, str):
            text = Text(f"    {text}", style="dim")
        self.content_lines.append(text)

    def add_spacing(self) -> None:
        """Add a blank line."""
        self.content_lines.append(Text())

    def render(self) -> Group:
        """Render as a group of lines."""
        header = Text()
        header.append(f"({self.stage_num}) ", style="dim")
        header.append(self.title.lower(), style="header")
        return Group(header, *self.content_lines)


class SummaryPanel:
    """Minimal summary display."""

    def __init__(self):
        self.pages_success = 0
        self.pages_total = 0
        self.figures_count = 0
        self.time_seconds = 0.0
        self.cost = 0.0
        self.engines_used: dict[str, int] = {}
        self.output_path = ""
        self.output_files: list[str] = []

    def set_stats(
        self,
        pages_success: int,
        pages_total: int,
        figures_count: int = 0,
        time_seconds: float = 0.0,
        cost: float = 0.0,
    ) -> None:
        """Set processing statistics."""
        self.pages_success = pages_success
        self.pages_total = pages_total
        self.figures_count = figures_count
        self.time_seconds = time_seconds
        self.cost = cost

    def add_engine_usage(self, engine: str, count: int) -> None:
        """Record engine usage."""
        self.engines_used[engine] = self.engines_used.get(engine, 0) + count

    def set_output(self, path: str, files: list[str] | None = None) -> None:
        """Set output location."""
        self.output_path = path
        self.output_files = files or []

    def render(self) -> Group:
        """Render the summary."""
        lines = []
        
        lines.append(Text("---", style="dim"))
        lines.append(Text())
        
        # Stats
        lines.append(Text(f"done {self.pages_success}/{self.pages_total} pages", style="success"))
        
        if self.figures_count > 0:
            lines.append(Text(f"     {self.figures_count} figures", style="dim"))
        
        lines.append(Text(f"     {self.time_seconds:.1f}s", style="dim"))
        
        if self.cost > 0:
            lines.append(Text(f"     ${self.cost:.4f}", style="dim"))
        
        # Engines
        engine_parts = [f"{ENGINE_LABELS.get(e, e)} ({c})" for e, c in self.engines_used.items()]
        lines.append(Text(f"     {' + '.join(engine_parts)}", style="dim"))
        
        lines.append(Text())
        lines.append(Text(f"-> {self.output_path}", style="dim"))
        
        return Group(*lines)


class AuditPanel:
    """Minimal audit display."""

    def __init__(self):
        self.metrics: list[dict] = []
        self.llm_results: list[dict] = []

    def add_metric(
        self,
        name: str,
        value: str,
        threshold: str | None = None,
        passed: bool = True,
    ) -> None:
        """Add an audit metric."""
        self.metrics.append({
            "name": name,
            "value": value,
            "threshold": threshold,
            "passed": passed,
        })

    def add_llm_review(
        self,
        item: str,
        verdict: str,
        reason: str = "",
    ) -> None:
        """Add LLM review result."""
        self.llm_results.append({
            "item": item,
            "verdict": verdict,
            "reason": reason,
        })

    def render(self) -> Group:
        """Render the audit panel."""
        lines = []
        
        for m in self.metrics:
            icon = STATUS_ICONS["success"] if m["passed"] else STATUS_ICONS["warning"]
            style = "success" if m["passed"] else "warning"
            lines.append(Text(f"    [{icon}] {m['name']}: {m['value']}", style=style))
        
        for r in self.llm_results:
            status = "success" if r["verdict"] == "acceptable" else "warning"
            icon = STATUS_ICONS["success"] if r["verdict"] == "acceptable" else STATUS_ICONS["warning"]
            line = f"    [{icon}] {r['item']}: {r['verdict']}"
            if r["reason"]:
                line += f" {r['reason']}"
            lines.append(Text(line, style=status))
        
        return Group(*lines)
