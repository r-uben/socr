"""Multi-engine output reconciliation for HPC mode.

Reconciles outputs from multiple OCR engines (DeepSeek-vLLM, Nougat, etc.)
to produce the best combined result. Uses structural merge for LaTeX equations
with optional LLM for conflict resolution.
"""

import re
from dataclasses import dataclass, field

from smart_ocr.core.result import PageResult, PageStatus


@dataclass
class EngineOutput:
    """Output from a single engine for a page."""

    engine: str
    text: str
    confidence: float = 0.0
    processing_time: float = 0.0


@dataclass
class ReconciliationResult:
    """Result of reconciling multiple engine outputs."""

    text: str
    primary_engine: str
    engines_used: list[str] = field(default_factory=list)
    latex_source: str = ""  # Engine that provided LaTeX
    conflicts_resolved: int = 0
    confidence: float = 0.0


@dataclass
class LaTeXBlock:
    """A LaTeX equation block with position info."""

    content: str
    is_display: bool  # $$...$$ or \begin{equation}
    start_pos: int
    end_pos: int
    normalized: str = ""  # Normalized form for comparison


class OutputReconciler:
    """Reconciles outputs from multiple OCR engines.

    Strategy:
    1. Use DeepSeek-vLLM for primary text structure
    2. Extract LaTeX blocks from Nougat (better at equations)
    3. Merge LaTeX blocks into DeepSeek text at appropriate positions
    4. Optionally use LLM to resolve conflicts
    """

    def __init__(
        self,
        use_llm_reconciler: bool = False,
        reconciler_model: str = "",
        vllm_url: str = "",
    ) -> None:
        self.use_llm_reconciler = use_llm_reconciler
        self.reconciler_model = reconciler_model
        self.vllm_url = vllm_url

    def reconcile(
        self,
        outputs: list[EngineOutput],
        page_num: int,
    ) -> ReconciliationResult:
        """Reconcile outputs from multiple engines.

        Args:
            outputs: List of EngineOutput from different engines
            page_num: Page number for logging

        Returns:
            ReconciliationResult with merged text
        """
        if not outputs:
            return ReconciliationResult(
                text="",
                primary_engine="none",
                confidence=0.0,
            )

        if len(outputs) == 1:
            output = outputs[0]
            return ReconciliationResult(
                text=output.text,
                primary_engine=output.engine,
                engines_used=[output.engine],
                confidence=output.confidence,
            )

        # Identify primary (DeepSeek-vLLM) and LaTeX source (Nougat)
        primary_output = None
        nougat_output = None
        other_outputs = []

        for output in outputs:
            if output.engine == "deepseek-vllm":
                primary_output = output
            elif output.engine == "nougat":
                nougat_output = output
            else:
                other_outputs.append(output)

        # Fallback: use first output as primary if DeepSeek not available
        if primary_output is None:
            primary_output = outputs[0]

        # If we have both DeepSeek and Nougat, merge LaTeX
        if nougat_output and primary_output.engine != "nougat":
            merged_text, latex_count = self._merge_latex_into_text(
                primary_output.text,
                nougat_output.text,
            )
            return ReconciliationResult(
                text=merged_text,
                primary_engine=primary_output.engine,
                engines_used=[o.engine for o in outputs],
                latex_source="nougat" if latex_count > 0 else "",
                conflicts_resolved=latex_count,
                confidence=max(o.confidence for o in outputs),
            )

        # No Nougat available, use primary output
        return ReconciliationResult(
            text=primary_output.text,
            primary_engine=primary_output.engine,
            engines_used=[o.engine for o in outputs],
            confidence=primary_output.confidence,
        )

    def _extract_latex_blocks(self, text: str) -> list[LaTeXBlock]:
        """Extract LaTeX equation blocks from text.

        Handles:
        - Inline math: $...$
        - Display math: $$...$$
        - LaTeX environments: \\begin{equation}...\\end{equation}
        - Other environments: align, gather, etc.
        """
        blocks = []

        # Pattern for display math ($$...$$)
        display_pattern = r'\$\$([^$]+)\$\$'
        for match in re.finditer(display_pattern, text, re.DOTALL):
            blocks.append(LaTeXBlock(
                content=match.group(0),
                is_display=True,
                start_pos=match.start(),
                end_pos=match.end(),
                normalized=self._normalize_latex(match.group(1)),
            ))

        # Pattern for LaTeX environments
        env_pattern = r'\\begin\{(equation|align|gather|multline)\*?\}(.+?)\\end\{\1\*?\}'
        for match in re.finditer(env_pattern, text, re.DOTALL):
            blocks.append(LaTeXBlock(
                content=match.group(0),
                is_display=True,
                start_pos=match.start(),
                end_pos=match.end(),
                normalized=self._normalize_latex(match.group(2)),
            ))

        # Pattern for inline math ($...$) - avoid matching $$
        inline_pattern = r'(?<!\$)\$(?!\$)([^$]+)\$(?!\$)'
        for match in re.finditer(inline_pattern, text):
            blocks.append(LaTeXBlock(
                content=match.group(0),
                is_display=False,
                start_pos=match.start(),
                end_pos=match.end(),
                normalized=self._normalize_latex(match.group(1)),
            ))

        # Sort by position
        blocks.sort(key=lambda b: b.start_pos)
        return blocks

    def _normalize_latex(self, latex: str) -> str:
        """Normalize LaTeX for comparison."""
        # Remove whitespace variations
        normalized = re.sub(r'\s+', ' ', latex.strip())
        # Remove common formatting differences
        normalized = normalized.replace(r'\ ', ' ')
        normalized = normalized.replace(r'\,', ' ')
        return normalized

    def _merge_latex_into_text(
        self,
        base_text: str,
        latex_source_text: str,
    ) -> tuple[str, int]:
        """Merge LaTeX blocks from Nougat into DeepSeek text.

        Args:
            base_text: Primary text (from DeepSeek-vLLM)
            latex_source_text: Text with better LaTeX (from Nougat)

        Returns:
            Tuple of (merged_text, number_of_latex_blocks_merged)
        """
        # Extract LaTeX from both sources
        base_blocks = self._extract_latex_blocks(base_text)
        source_blocks = self._extract_latex_blocks(latex_source_text)

        if not source_blocks:
            return base_text, 0

        # If base has no LaTeX but source does, try to find insertion points
        if not base_blocks and source_blocks:
            # Simple strategy: append LaTeX blocks that look like display equations
            # at paragraph boundaries
            merged = base_text
            merged_count = 0

            for block in source_blocks:
                if block.is_display:
                    # Try to find a good insertion point
                    # Look for text that might reference this equation
                    insertion_point = self._find_insertion_point(merged, block)
                    if insertion_point >= 0:
                        merged = (
                            merged[:insertion_point] +
                            "\n\n" + block.content + "\n\n" +
                            merged[insertion_point:]
                        )
                        merged_count += 1

            return merged, merged_count

        # Both have LaTeX - replace base LaTeX with source LaTeX where they match
        merged = base_text
        replacements = 0

        for source_block in source_blocks:
            # Find matching block in base by normalized content
            for base_block in base_blocks:
                if self._latex_blocks_match(base_block, source_block):
                    # Replace base block with source block (better formatting)
                    merged = merged.replace(base_block.content, source_block.content)
                    replacements += 1
                    break

        return merged, replacements

    def _latex_blocks_match(self, block1: LaTeXBlock, block2: LaTeXBlock) -> bool:
        """Check if two LaTeX blocks represent the same equation."""
        # Same display mode and similar content
        if block1.is_display != block2.is_display:
            return False

        # Check normalized content similarity
        norm1 = block1.normalized.lower()
        norm2 = block2.normalized.lower()

        # Simple containment check
        if norm1 in norm2 or norm2 in norm1:
            return True

        # Check for common equation patterns
        # E.g., both might have "frac" or specific variable names
        common_tokens = set(re.findall(r'\\?\w+', norm1)) & set(re.findall(r'\\?\w+', norm2))
        total_tokens = set(re.findall(r'\\?\w+', norm1)) | set(re.findall(r'\\?\w+', norm2))

        if total_tokens and len(common_tokens) / len(total_tokens) > 0.6:
            return True

        return False

    def _find_insertion_point(self, text: str, latex_block: LaTeXBlock) -> int:
        """Find a good insertion point for a LaTeX block.

        Looks for paragraph breaks or equation references.
        """
        # Look for equation references like "equation (1)" or "formula"
        patterns = [
            r'equation\s*\(\d+\)',
            r'formula\s*\(\d+\)',
            r'as follows:',
            r'given by:',
            r'defined as:',
            r'where:',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Insert after the match
                return match.end()

        # Find the first paragraph break
        para_break = text.find('\n\n')
        if para_break > 0:
            return para_break

        return -1

    def _llm_reconcile(
        self,
        outputs: list[EngineOutput],
        context: str = "",
    ) -> str:
        """Use LLM to reconcile conflicting outputs.

        Not implemented yet - placeholder for future enhancement.
        """
        # TODO: Implement LLM-based reconciliation
        # This would send the outputs to the reconciler model and ask it
        # to pick the best version or merge them intelligently
        raise NotImplementedError("LLM reconciliation not yet implemented")


def create_page_result_from_reconciliation(
    reconciliation: ReconciliationResult,
    page_num: int,
    processing_time: float = 0.0,
) -> PageResult:
    """Convert ReconciliationResult to PageResult."""
    return PageResult(
        page_num=page_num,
        text=reconciliation.text,
        status=PageStatus.SUCCESS if reconciliation.text else PageStatus.ERROR,
        engine=f"reconciled({','.join(reconciliation.engines_used)})",
        confidence=reconciliation.confidence,
        processing_time=processing_time,
        cost=0.0,
    )
