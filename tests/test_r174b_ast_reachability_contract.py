"""R174b acceptance tests: AST reachability and architectural contract.

Verifies:
- The 13 legacy-only methods are absent from UnifiedPipeline (or verified as legacy-only).
- process() control flow calls only analyze -> agentic -> assemble and common helpers.
- No agentic-reachable processing method reads any of the 6 dead config fields.
- consensus.py and repair.py are absent and have zero imports anywhere under src/socr.
- reconciler.py survives and is imported by hpc_pipeline.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_PATH = REPO_ROOT / "src/socr/pipeline/orchestrator.py"

LEGACY_13_METHODS = frozenset(
    {
        "_backbone_native_first",
        "_phase_repair",
        "_phase_dual_pass_tables",
        "_score_per_page",
        "_phase_judge_hard_pages",
        "_phase_backbone",
        "_phase_score_multi",
        "_backbone_multi_engine",
        "_phase_consensus",
        "_score_repair_result",
        "_score_whole_doc",
        "_phase_score",
        "_native_table_structure_gate_applies",
    }
)

DEAD_CONFIG_FIELDS = frozenset(
    {
        "multi_engine",
        "consensus_enabled",
        "consensus_use_llm",
        "consensus_ollama_model",
        "max_retries",
        "truncation_retries",
    }
)


def _branch_calls(node: ast.AST, method_names: set[str]) -> set[str]:
    """Collect self.<method> Attribute references from a branch, excluding nested defs."""
    calls: set[str] = set()

    class BranchVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, child):
            if child is not node:
                return
            self.generic_visit(child)

        def visit_AsyncFunctionDef(self, child):
            if child is not node:
                return
            self.generic_visit(child)

        def visit_ClassDef(self, child):
            return

        def visit_Attribute(self, child):
            if isinstance(child.value, ast.Name) and child.value.id == "self":
                if child.attr in method_names:
                    calls.add(child.attr)
            self.generic_visit(child)

    BranchVisitor().visit(node)
    return calls


def _branch_call_sequence(node: ast.AST, method_names: set[str]) -> list[str]:
    """Collect bound-method references in source order, excluding nested defs."""
    calls: list[str] = []

    class BranchVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, child):
            if child is not node:
                return
            self.generic_visit(child)

        def visit_AsyncFunctionDef(self, child):
            if child is not node:
                return
            self.generic_visit(child)

        def visit_ClassDef(self, child):
            return

        def visit_Attribute(self, child):
            if isinstance(child.value, ast.Name) and child.value.id == "self":
                if child.attr in method_names:
                    calls.append(child.attr)
            self.generic_visit(child)

    BranchVisitor().visit(node)
    return calls


def _mentions_field(node: ast.AST, field_name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == field_name:
            if isinstance(child.value, ast.Attribute) and child.value.attr == "config":
                return True
            if isinstance(child.value, ast.Name) and child.value.id in {"config", "cfg"}:
                return True
        if isinstance(child, ast.Name) and child.id == field_name:
            return True
    return False


def _get_unified_pipeline_ast():
    with open(ORCHESTRATOR_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(ORCHESTRATOR_PATH))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "UnifiedPipeline":
            return node, tree
    raise AssertionError("UnifiedPipeline class not found in orchestrator.py")


def _legacy_module_name(value: str) -> str | None:
    """Return the deleted module name represented by an import target."""
    targets = {
        "consensus": {"consensus", ".consensus", "socr.pipeline.consensus"},
        "repair": {"repair", ".repair", "socr.pipeline.repair"},
    }
    for module_name, spellings in targets.items():
        if value in spellings:
            return module_name
    return None


class TestUnifiedPipelineASTReachability:
    """AST reachability proofs for UnifiedPipeline."""

    def test_legacy_13_methods_reachability_partition(self):
        """Derive reachability from process() AST and verify legacy deletion.

        Proves that:
        - All 13 legacy method definitions are absent from UnifiedPipeline.
        - Direct process() phase sequence is analyze -> agentic -> assemble.
        - No routing conditional reads config.agentic or multi_engine.
        - _sparse_page_ok is reachable through the bound-method-aware graph from process().
        """
        cls_node, _ = _get_unified_pipeline_ast()
        methods = {
            item.name: item
            for item in cls_node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        # 1. Assert all 13 legacy methods are absent
        present_legacy_methods = LEGACY_13_METHODS & set(methods.keys())
        assert not present_legacy_methods, (
            f"Legacy methods still defined in UnifiedPipeline: {sorted(present_legacy_methods)}"
        )

        # 2. Build bound-method-aware call graph
        call_graph = {name: set() for name in methods}
        for name, func_node in methods.items():
            for subnode in ast.walk(func_node):
                if isinstance(subnode, ast.Attribute):
                    if isinstance(subnode.value, ast.Name) and subnode.value.id == "self":
                        if subnode.attr in methods:
                            call_graph[name].add(subnode.attr)

        def get_reachable(roots: set[str]) -> set[str]:
            visited = set(roots)
            queue = list(roots)
            while queue:
                curr = queue.pop(0)
                for neighbor in call_graph.get(curr, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return visited

        # 3. Derive direct roots from process() AST.  The phase sequence is
        # intentionally checked in source order: a set of expected roots would
        # allow a deleted routing fork to remain hidden in process().
        process_node = methods["process"]
        method_names = set(methods)
        process_call_sequence = _branch_call_sequence(process_node, method_names)
        process_roots = set(process_call_sequence)

        phase_roots = [name for name in process_call_sequence if name.startswith("_phase_")]
        assert phase_roots == ["_phase_analyze", "_phase_agentic", "_phase_assemble"], (
            "process() must have exactly the direct phase sequence "
            "_phase_analyze -> _phase_agentic -> _phase_assemble; "
            f"got {phase_roots}"
        )

        assert "_phase_analyze" in process_roots, "process() must call _phase_analyze"
        assert "_phase_agentic" in process_roots, "process() must call _phase_agentic"
        assert "_phase_assemble" in process_roots, "process() must call _phase_assemble"

        # 4. Assert no routing conditional reads config.agentic or multi_engine
        for if_node in [n for n in ast.walk(process_node) if isinstance(n, ast.If)]:
            assert not _mentions_field(if_node.test, "agentic"), (
                "process() still contains routing conditional reading config.agentic"
            )
            assert not _mentions_field(if_node.test, "multi_engine"), (
                "process() still contains routing conditional reading multi_engine"
            )

        # 5. Assert reachability of live methods including callback _sparse_page_ok
        agentic_roots = {phase_roots[1]}
        assert agentic_roots, "process() must derive an _phase_agentic root"
        agentic_reachable = get_reachable(agentic_roots)
        assert "_sparse_page_ok" in agentic_reachable, (
            "_sparse_page_ok must be reachable from derived agentic roots "
            "via the bound-method-aware graph"
        )
        assert "_reread_page_tables" in agentic_reachable, (
            "_reread_page_tables must remain reachable in agentic path"
        )

    def test_agentic_processing_methods_read_no_dead_config_fields(self):
        """Prove that no agentic processing method reads any of the 6 dead config fields."""
        cls_node, _ = _get_unified_pipeline_ast()
        methods = {
            item.name: item
            for item in cls_node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        call_graph = {name: set() for name in methods}
        field_reads = {name: set() for name in methods}

        for name, func_node in methods.items():
            for subnode in ast.walk(func_node):
                if isinstance(subnode, ast.Attribute):
                    if isinstance(subnode.value, ast.Name) and subnode.value.id == "self":
                        if subnode.attr in methods:
                            call_graph[name].add(subnode.attr)
                    elif (
                        isinstance(subnode.value, ast.Attribute) and subnode.value.attr == "config"
                    ):
                        field_reads[name].add(subnode.attr)
                    elif isinstance(subnode.value, ast.Name) and subnode.value.id in {
                        "config",
                        "cfg",
                    }:
                        field_reads[name].add(subnode.attr)

        process_node = methods["process"]
        process_roots = _branch_calls(process_node, set(methods.keys()))

        visited = set(process_roots)
        queue = list(visited)
        while queue:
            curr = queue.pop(0)
            for neighbor in call_graph.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        for m in visited:
            reads = field_reads.get(m, set()) & DEAD_CONFIG_FIELDS
            assert not reads, f"Agentic-reachable method {m} reads dead config fields: {reads}"


class TestModuleImportContracts:
    """AST import scan over all src/socr Python modules."""

    def test_consensus_and_repair_import_boundaries(self):
        """consensus.py and repair.py must be deleted and have zero consumers."""
        consensus_path = REPO_ROOT / "src/socr/pipeline/consensus.py"
        repair_path = REPO_ROOT / "src/socr/pipeline/repair.py"

        assert not consensus_path.exists(), "src/socr/pipeline/consensus.py must be deleted"
        assert not repair_path.exists(), "src/socr/pipeline/repair.py must be deleted"

        import_consumers: dict[str, set[str]] = {"consensus": set(), "repair": set()}

        for py_path in sorted(REPO_ROOT.glob("src/socr/**/*.py")):
            rel_path = py_path.relative_to(REPO_ROOT)
            with open(py_path, encoding="utf-8") as f:
                mod_tree = ast.parse(f.read(), filename=str(py_path))
            for node in ast.walk(mod_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = _legacy_module_name(alias.name)
                        if module_name is not None:
                            import_consumers[module_name].add(str(rel_path))
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    module_name = _legacy_module_name(mod)
                    if module_name is not None:
                        import_consumers[module_name].add(str(rel_path))
                    elif mod in {"", "socr.pipeline"}:
                        for alias in node.names:
                            module_name = _legacy_module_name(alias.name)
                            if module_name is not None:
                                import_consumers[module_name].add(str(rel_path))
                elif isinstance(node, ast.Call) and node.args:
                    func = node.func
                    is_dynamic = (
                        isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}
                    ) or (
                        isinstance(func, ast.Attribute)
                        and func.attr in {"import_module", "__import__"}
                    )
                    if (
                        is_dynamic
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        module_name = _legacy_module_name(node.args[0].value)
                        if module_name is not None:
                            import_consumers[module_name].add(str(rel_path))

        assert not import_consumers["consensus"], (
            f"consensus imported in: {import_consumers['consensus']}"
        )
        assert not import_consumers["repair"], f"repair imported in: {import_consumers['repair']}"

    def test_reconciler_survives_and_is_imported_by_hpc(self):
        """reconciler.py is not part of the legacy stack and must be imported by hpc_pipeline.py."""
        reconciler_file = REPO_ROOT / "src/socr/pipeline/reconciler.py"
        assert reconciler_file.exists(), "src/socr/pipeline/reconciler.py must survive"

        hpc_file = REPO_ROOT / "src/socr/pipeline/hpc_pipeline.py"
        assert hpc_file.exists(), "src/socr/pipeline/hpc_pipeline.py must exist"

        with open(hpc_file, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(hpc_file))

        imports_reconciler = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "reconciler" in node.module:
                    imports_reconciler = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "reconciler" in alias.name:
                        imports_reconciler = True

        assert imports_reconciler, "hpc_pipeline.py must import reconciler"
