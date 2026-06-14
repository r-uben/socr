"""Lightweight Ollama helpers — no engine-framework dependencies."""

from __future__ import annotations

import subprocess


def check_ollama_model(model_name: str) -> str | None:
    """Return an error string if the model is not available, or None if it is."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return "Ollama is not running or not installed"
        model_names = [
            line.split()[0] for line in result.stdout.strip().splitlines()[1:] if line.split()
        ]
        if model_name not in model_names:
            return f"Ollama model '{model_name}' not found. Pull it with: ollama pull {model_name}"
    except FileNotFoundError:
        return "Ollama is not installed (ollama command not found)"
    except subprocess.TimeoutExpired:
        return "Ollama did not respond (timeout)"
    return None
