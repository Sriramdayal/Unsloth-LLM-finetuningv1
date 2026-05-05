"""
Platform-specific llama-cpp-python bootstrap.

llama-cpp-python uses different native GPU backends per OS:

  Windows NVIDIA  → CUBLAS  (ggml-cuda.dll — DLL path bootstrap needed)
  Linux   NVIDIA  → CUDA    (libllama.so links cudart directly — no fix needed)
  macOS   Silicon → Metal   (libllama.dylib links Metal — no fix needed)
  Any     CPU     → CPU     (no GPU library needed)

The only platform requiring a path fix is native Windows, where cudart64_12.dll
is not in the system PATH but IS bundled inside PyTorch's install.

Usage:
    Call bootstrap_platform_dlls() once before `import llama_cpp`.
    It is safe to call multiple times (no-op after first call).
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windows helpers
# ---------------------------------------------------------------------------

def _find_torch_cuda_lib() -> str | None:
    """Return the path to PyTorch's bundled CUDA lib directory (Windows)."""
    try:
        import torch
        candidate = os.path.join(os.path.dirname(torch.__file__), "lib")
        cudart = os.path.join(candidate, "cudart64_12.dll")
        if os.path.exists(cudart):
            return candidate
    except ImportError:
        pass
    return None


def _find_cuda_toolkit() -> str | None:
    """Return NVIDIA CUDA Toolkit bin directory if the toolkit is installed."""
    for ver in ("12.4", "12.3", "12.2", "12.1", "12.0", "11.8"):
        p = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{ver}\bin"
        if os.path.exists(p):
            return p
    return None


def _find_llama_lib_dir() -> str | None:
    """Return the llama_cpp lib directory (contains llama.dll / ggml-cuda.dll)."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.origin:
            lib_dir = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isdir(lib_dir):
                return lib_dir
    except Exception:
        pass
    return None


def _bootstrap_windows() -> None:
    """
    Add CUDA and llama_cpp DLL directories to the Windows DLL search path.

    Required because llama.dll → ggml-cuda.dll → cudart64_12.dll, and
    cudart64_12.dll is not in %PATH% unless the CUDA Toolkit is installed.
    PyTorch bundles its own copy at <torch>/lib/cudart64_12.dll.
    """
    added: list[str] = []

    # 1. llama_cpp/lib  (llama.dll, ggml-cuda.dll, ggml-base.dll, etc.)
    llama_lib = _find_llama_lib_dir()
    if llama_lib:
        os.add_dll_directory(llama_lib)
        added.append(llama_lib)

    # 2. PyTorch bundled CUDA runtime (preferred — no toolkit needed)
    torch_lib = _find_torch_cuda_lib()
    if torch_lib:
        os.add_dll_directory(torch_lib)
        added.append(torch_lib)

    # 3. CUDA Toolkit (optional — only if installed separately)
    toolkit = _find_cuda_toolkit()
    if toolkit:
        os.add_dll_directory(toolkit)
        added.append(toolkit)

    if added:
        logger.debug(f"[llama_loader] Windows DLL directories registered: {added}")
    else:
        logger.warning(
            "[llama_loader] Could not locate CUDA runtime DLLs on Windows. "
            "llama-cpp-python GPU acceleration may fail to load."
        )


# ---------------------------------------------------------------------------
# macOS helpers
# ---------------------------------------------------------------------------

def _check_metal_available() -> None:
    """
    Log Metal GPU availability on macOS.
    llama.cpp Metal support is built into the wheel — no path fix needed.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        if "Metal" in result.stdout or "Apple" in result.stdout:
            logger.debug("[llama_loader] macOS Metal GPU detected — llama.cpp will use Metal.")
        else:
            logger.debug("[llama_loader] macOS: Metal GPU not detected — using CPU.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Linux helpers
# ---------------------------------------------------------------------------

def _check_linux_cuda() -> None:
    """
    Log CUDA availability on Linux.
    libcudart.so is in LD_LIBRARY_PATH automatically — no fix needed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.debug(
                f"[llama_loader] Linux CUDA detected: {torch.cuda.get_device_name(0)}. "
                "llama.cpp will use CUDA kernels."
            )
        else:
            logger.debug("[llama_loader] Linux: No CUDA detected — llama.cpp will use CPU.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def bootstrap_platform_dlls() -> None:
    """
    Prepare the environment for llama-cpp-python import.

    Must be called BEFORE `import llama_cpp` so native libraries load correctly.
    Safe to call multiple times — subsequent calls are no-ops.

    Platform actions:
      Windows → adds DLL search directories (cudart64_12.dll fix)
      macOS   → logs Metal GPU availability (no path fix required)
      Linux   → logs CUDA availability (no path fix required)
    """
    if getattr(bootstrap_platform_dlls, "_done", False):
        return
    bootstrap_platform_dlls._done = True  # type: ignore[attr-defined]

    if sys.platform == "win32":
        _bootstrap_windows()
    elif sys.platform == "darwin":
        _check_metal_available()
    else:
        _check_linux_cuda()


# Backwards-compatible alias (used by older model_runner.py imports)
bootstrap_windows_cuda_dlls = bootstrap_platform_dlls
