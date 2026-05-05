"""
Windows CUDA DLL bootstrap for llama-cpp-python.

llama-cpp-python's prebuilt CUBLAS wheel ships ggml-cuda.dll which depends on
cudart64_12.dll. On Windows without a CUDA Toolkit installation, this DLL is
only available inside PyTorch's bundled libs.

This module must be imported BEFORE `import llama_cpp` so that os.add_dll_directory
adds the required paths before ctypes tries to load llama.dll.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _find_torch_cuda_lib() -> str | None:
    """Return the path to PyTorch's bundled CUDA lib directory, if available."""
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
    """Return the NVIDIA CUDA Toolkit bin directory if installed."""
    for ver in ("12.4", "12.3", "12.2", "12.1", "12.0", "11.8"):
        p = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{ver}\bin"
        if os.path.exists(p):
            return p
    return None


def bootstrap_windows_cuda_dlls() -> None:
    """
    Add CUDA and llama_cpp lib directories to the DLL search path.

    Must be called before `import llama_cpp` on Windows so that Windows can
    resolve llama.dll's dependencies (cudart64_12.dll, cublas64_12.dll, etc.).

    Safe to call multiple times — subsequent calls are no-ops.
    """
    if sys.platform != "win32":
        return  # Only needed on Windows

    if getattr(bootstrap_windows_cuda_dlls, "_done", False):
        return
    bootstrap_windows_cuda_dlls._done = True  # type: ignore[attr-defined]

    added: list[str] = []

    # 1. llama_cpp lib dir (contains llama.dll, ggml-cuda.dll, etc.)
    try:
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.origin:
            lib_dir = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isdir(lib_dir):
                os.add_dll_directory(lib_dir)
                added.append(lib_dir)
    except Exception:
        pass

    # 2. PyTorch bundled cudart (preferred — always present when torch is installed)
    torch_lib = _find_torch_cuda_lib()
    if torch_lib:
        os.add_dll_directory(torch_lib)
        added.append(torch_lib)

    # 3. CUDA Toolkit bin (optional — only if user has toolkit installed)
    toolkit = _find_cuda_toolkit()
    if toolkit:
        os.add_dll_directory(toolkit)
        added.append(toolkit)

    if added:
        logger.debug(f"[llama_loader] Added DLL directories: {added}")
    else:
        logger.warning(
            "[llama_loader] Could not locate CUDA runtime DLLs. "
            "llama-cpp-python may fail to load on Windows."
        )
