"""Subprocess wrapper for the Soup CLI (hybrid integration).

Soup (https://github.com/MakazhanAlpamys/Soup) is used as an external tool
for export/eval/data tasks — it is never imported, only invoked via
subprocess. This keeps ``soup-cli`` an optional dependency.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


class SoupNotAvailableError(RuntimeError):
    """Raised when the `soup` binary is not found on PATH."""


class SoupClient:
    """Thin subprocess client for ``soup`` CLI commands.

    Example:
        client = SoupClient()
        if client.is_available():
            gguf_path = client.export_gguf("./output", quant="q4_k_m")
    """

    def __init__(self, soup_bin: str = "soup"):
        self.soup_bin = soup_bin

    def is_available(self) -> bool:
        """Return True when the `soup` binary is found on PATH."""
        return shutil.which(self.soup_bin) is not None

    def _require_available(self) -> None:
        if not self.is_available():
            raise SoupNotAvailableError(
                f"'{self.soup_bin}' not found on PATH. "
                "Install with: pip install \"soup-cli[train]>=0.74.0,<0.75.0\" "
                "or: pip install -e \".[soup]\""
            )

    def _run(self, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Run a soup command, logging it first. Raises on non-zero exit."""
        self._require_available()
        cmd = [self.soup_bin, *args]
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            logger.error(f"'{' '.join(cmd)}' failed: {e.stderr or e.stdout or e}")
            raise RuntimeError(f"soup command failed: {' '.join(cmd)}") from e
        return result

    def export_gguf(
        self,
        model_path: str,
        quant: str = "q4_k_m",
        output_path: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Export a model to GGUF format via ``soup export``.

        Args:
            model_path:  HF model directory or adapter output directory.
            quant:       GGUF quantization (default ``q4_k_m``).
            output_path: Optional destination path for the .gguf file.
            timeout:     Optional subprocess timeout in seconds.

        Returns:
            The output path of the exported .gguf file.
        """
        args = ["export", "--model", model_path, "--format", "gguf", "--quant", quant]
        if output_path:
            args += ["--output", output_path]
        self._run(args, timeout=timeout)
        if output_path:
            return output_path
        # Soup derives the filename when --output is omitted; return stdout for discovery.
        return model_path
