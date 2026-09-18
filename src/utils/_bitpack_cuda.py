"""Lazy, optional CUDA extension. CPU use never imports the build toolchain."""

import os
from pathlib import Path
import threading
import warnings

_extension = None
_attempted = False
_error = None
_lock = threading.Lock()


def backend_status():
    return {
        "requested": os.environ.get("MEMFLORA_BITPACK_BACKEND", "auto"),
        "loaded": _extension is not None,
        "build_attempted": _attempted,
        "error": _error,
    }


def cuda_extension():
    """auto falls back once with a warning; cuda makes build failures fatal."""
    global _extension, _attempted, _error
    mode = os.environ.get("MEMFLORA_BITPACK_BACKEND", "auto")
    if mode not in ("auto", "cuda", "torch"):
        raise ValueError("MEMFLORA_BITPACK_BACKEND must be auto, cuda, or torch")
    if mode == "torch":
        return None
    with _lock:
        if not _attempted:
            _attempted = True
            try:
                from torch.utils.cpp_extension import CUDA_HOME, load

                if CUDA_HOME is None:
                    raise RuntimeError("CUDA toolkit/nvcc not found; set CUDA_HOME")
                source = Path(__file__).with_name("csrc")
                previous_jobs = os.environ.get("MAX_JOBS")
                try:
                    # Concurrent C++/nvcc compilation can exhaust small Jetsons.
                    os.environ.setdefault("MAX_JOBS", "1")
                    _extension = load(
                        name="memflora_bitpack_cuda",
                        sources=[str(source / "bitpack.cpp"), str(source / "bitpack.cu")],
                        extra_cflags=["-O2"],
                        extra_cuda_cflags=["-O2"],
                        verbose=False,
                    )
                finally:
                    if previous_jobs is None:
                        os.environ.pop("MAX_JOBS", None)
            except (ImportError, OSError, RuntimeError) as exc:
                _error = str(exc)
                if mode == "auto":
                    warnings.warn(
                        "MemFLoRA CUDA bit packing unavailable; using the torch "
                        f"fallback. {_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        if _extension is None and mode == "cuda":
            raise RuntimeError(f"MemFLoRA CUDA bit packing unavailable: {_error}")
        return _extension
