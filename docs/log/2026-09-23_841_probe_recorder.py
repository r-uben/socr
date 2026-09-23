"""pytest plugin: record which tests reach the live AUTO-engine probe (#841).

Wraps socr.engines.registry.resolve_auto_engine and resolve_local_engine so each
call records the current test node id, then lets the real call proceed. Loaded
with -p probe_recorder; writes one node id per line to $PROBE_OUT at session end.
"""

import os
import pytest

_hits = {}
_current = {"id": None}


def pytest_runtest_setup(item):
    _current["id"] = item.nodeid


def pytest_configure(config):
    from socr.engines import registry
    from socr.pipeline import orchestrator

    for name in ("resolve_auto_engine", "resolve_local_engine"):
        real = getattr(registry, name)

        def wrap(*a, _real=real, _name=name, **k):
            _hits.setdefault(_current["id"], set()).add(_name)
            return _real(*a, **k)

        setattr(registry, name, wrap)
        # orchestrator imports resolve_auto_engine by name at module load, so the
        # registry attribute alone would not see its calls.
        if hasattr(orchestrator, name):
            setattr(orchestrator, name, wrap)


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["PROBE_OUT"], "w") as fh:
        for node, names in sorted(_hits.items(), key=lambda kv: str(kv[0])):
            fh.write(f"{node}\t{','.join(sorted(names))}\n")
