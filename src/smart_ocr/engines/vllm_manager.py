"""vLLM server lifecycle manager for HPC sequential mode.

Manages starting, stopping, and monitoring local vLLM servers.
Used when smart-ocr needs to swap models on a single GPU.
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class ServerConfig:
    """Configuration for vLLM server."""

    model: str
    port: int = 8000
    host: str = "0.0.0.0"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    trust_remote_code: bool = True
    dtype: str = "auto"
    api_key: str = "EMPTY"


class VLLMServerManager:
    """Manage local vLLM server lifecycle for sequential HPC mode.

    This class handles:
    - Starting vLLM with a specific model
    - Waiting for the server to be ready
    - Stopping the server and freeing GPU memory
    - Graceful cleanup on exit

    Example:
        manager = VLLMServerManager()
        manager.start(ServerConfig(model="deepseek-ai/DeepSeek-OCR"))
        # ... use the server ...
        manager.stop()
    """

    def __init__(self, verbose: bool = False) -> None:
        self.process: subprocess.Popen | None = None
        self.current_model: str | None = None
        self.current_port: int | None = None
        self.verbose = verbose
        self._log_file: Path | None = None

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def start(
        self,
        config: ServerConfig,
        timeout: int = 180,
    ) -> bool:
        """Start vLLM server with specified model.

        Args:
            config: Server configuration
            timeout: Seconds to wait for server to be ready

        Returns:
            True if server started successfully

        Raises:
            RuntimeError: If server fails to start
        """
        # Stop any existing server first
        if self.process is not None:
            self.stop()

        # Build command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--host", config.host,
            "--port", str(config.port),
            "--max-model-len", str(config.max_model_len),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--dtype", config.dtype,
            "--api-key", config.api_key,
        ]

        if config.trust_remote_code:
            cmd.append("--trust-remote-code")

        # Set up logging
        self._log_file = Path(f"/tmp/vllm_server_{config.port}.log")

        env = os.environ.copy()
        # Ensure CUDA is visible
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = "0"

        if self.verbose:
            print(f"[vllm-manager] Starting: {config.model} on port {config.port}")
            print(f"[vllm-manager] Command: {' '.join(cmd)}")

        try:
            with open(self._log_file, "w") as log_f:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid,  # Create new process group for clean kill
                )

            self.current_model = config.model
            self.current_port = config.port

            # Wait for server to be ready
            if not self._wait_for_ready(config.port, timeout):
                self._print_logs()
                self.stop()
                raise RuntimeError(
                    f"vLLM server failed to start within {timeout}s. "
                    f"Check logs at {self._log_file}"
                )

            if self.verbose:
                print(f"[vllm-manager] Server ready: {config.model}")

            return True

        except FileNotFoundError:
            raise RuntimeError(
                "vLLM not found. Install with: pip install vllm"
            )
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to start vLLM server: {e}")

    def stop(self, force: bool = False) -> None:
        """Stop vLLM server and free GPU memory.

        Args:
            force: If True, use SIGKILL instead of SIGTERM
        """
        if self.process is None:
            return

        if self.verbose:
            print(f"[vllm-manager] Stopping: {self.current_model}")

        try:
            # Kill the entire process group
            pgid = os.getpgid(self.process.pid)
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            pass

        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait(timeout=5)
            except Exception:
                pass

        self.process = None
        self.current_model = None
        self.current_port = None

        # Force GPU memory release
        self._clear_gpu_memory()

        if self.verbose:
            print("[vllm-manager] Server stopped")

    def is_running(self) -> bool:
        """Check if server process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_base_url(self) -> str:
        """Get the base URL for the running server."""
        if self.current_port is None:
            raise RuntimeError("No server running")
        return f"http://localhost:{self.current_port}/v1"

    def _wait_for_ready(self, port: int, timeout: int) -> bool:
        """Wait for server to be ready.

        Args:
            port: Server port
            timeout: Maximum seconds to wait

        Returns:
            True if server is ready
        """
        url = f"http://localhost:{port}/v1/models"
        start_time = time.time()
        check_interval = 2.0

        while time.time() - start_time < timeout:
            # Check if process died
            if self.process and self.process.poll() is not None:
                return False

            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(url)
                    if response.status_code == 200:
                        return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            time.sleep(check_interval)

        return False

    def _clear_gpu_memory(self) -> None:
        """Force GPU memory release."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Additional cleanup: run gc
        import gc
        gc.collect()

    def _print_logs(self, lines: int = 50) -> None:
        """Print last N lines of server logs."""
        if self._log_file and self._log_file.exists():
            print(f"\n[vllm-manager] Last {lines} lines of server log:")
            with open(self._log_file) as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    print(f"  {line.rstrip()}")

    def _cleanup(self) -> None:
        """Cleanup handler for atexit."""
        if self.process is not None:
            self.stop(force=True)

    def __enter__(self) -> "VLLMServerManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __del__(self) -> None:
        self._cleanup()


def detect_gpu_setup() -> str:
    """Detect GPU configuration.

    Returns:
        "cpu" - No GPU available
        "single-gpu" - Single GPU
        "multi-gpu" - Multiple GPUs
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"

        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            return "single-gpu"
        return "multi-gpu"
    except ImportError:
        return "cpu"


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
    except ImportError:
        pass
    return 0.0
