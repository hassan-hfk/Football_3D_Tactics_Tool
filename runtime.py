"""

  12 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  13 pip install colorlog                                                                                                                                                      
  14 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  15 pip install einops                                                                                                                                                        
  16 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  17 pip install imageio                                                                                                                                                       
  18 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  19 pip install hydra_zen                                                                                                                                                     
  20 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  21 pip install pytorch_lightning                                                                                                                                             
  22 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  23 pip install timm                                                                                                                                                          
  24 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  25 pip install wis3d                                                                                                                                                         
  26 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  27 pip install ffmpeg                                                                                                                                                        
  28 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  29 python .\lift_gvhmr.py .\outputs\49fe8625.json --video '.\samples\sample_4 - Trim.mp4'                                                                                    
  30 pip install ffmpeg   

  
runtime.py — single source of truth for device selection, autocast,
threading config, and ONNX runtime providers. Imported by every stage
that touches a model.

Lossless guarantee:
  Weights are always stored and loaded in FP32. Lower precision applies
  only inside the autocast forward pass.
    - CUDA  : FP16 autocast (numerically safe for these inference models)
    - MPS   : FP16 autocast (supported since PyTorch 2.0)
    - CPU   : BF16 autocast IF the CPU supports it (Apple Silicon, Intel
              with AVX-512 BF16, AMD Zen 4), otherwise FP32.
  Set FOOTBALL_DISABLE_LOW_PRECISION=1 to force FP32 everywhere — useful
  while running the equivalence harness to establish a baseline.

Public surface:
  pick_device(pref="auto") -> "cuda" | "mps" | "cpu"
  autocast_dtype(device)   -> torch dtype used in the forward pass
  inference_ctx(device)    -> context manager (inference_mode + autocast)
  to_eval_device(model, d) -> model.eval().to(d), grads disabled
  onnx_providers(device)   -> ordered provider list for onnxruntime
  make_onnx_session(p, d)  -> InferenceSession with full graph optim
  print_banner(device=None)-> one-line runtime summary

CPU threading is applied process-wide at import time. Importing runtime
multiple times is a no-op (idempotent).
"""

from __future__ import annotations

import os
import platform
from contextlib import contextmanager, nullcontext


# ─────────────────────────────────────────────────────────────────────────────
# Module flags driven by env vars (cheap to read, no torch import yet)
# ─────────────────────────────────────────────────────────────────────────────

_FORCE_FP32 = os.environ.get("FOOTBALL_DISABLE_LOW_PRECISION", "0") == "1"
_DISABLE_CPU_BF16 = os.environ.get("FOOTBALL_DISABLE_CPU_BF16", "0") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# CPU threading config (applied once at first import)
# ─────────────────────────────────────────────────────────────────────────────

_CPU_THREADS_CONFIGURED = False


def _configure_cpu_threads() -> bool:
    global _CPU_THREADS_CONFIGURED
    if _CPU_THREADS_CONFIGURED:
        return True
    try:
        import torch
        n_logical = os.cpu_count() or 1
        # Leave one core for the OS / FastAPI event loop. On 4-core boxes
        # we still keep 3 threads which is fine.
        n_threads = max(1, n_logical - 1)
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(n_threads)
        if "MKL_NUM_THREADS" not in os.environ:
            os.environ["MKL_NUM_THREADS"] = str(n_threads)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(min(2, n_threads))
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
        _CPU_THREADS_CONFIGURED = True
        return True
    except Exception as e:
        print(f"[runtime] cpu thread config skipped: {e}")
        return False


_configure_cpu_threads()


def cpu_threads_configured() -> bool:
    return _CPU_THREADS_CONFIGURED


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def pick_device(pref: str = "auto") -> str:
    """Resolve a device preference. 'auto' = cuda > mps > cpu."""
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# CPU BF16 detection (cached)
# ─────────────────────────────────────────────────────────────────────────────

_BF16_CPU_CACHE: bool | None = None


def _cpu_supports_bf16() -> bool:
    """Hardware BF16 inference support on this CPU.

    Covers:
      - Intel Xeon w/ AVX-512 BF16 (Cooper Lake +)
      - Intel/AMD with AMX
      - Apple Silicon (M1+)
      - AWS Graviton 3+
    """
    global _BF16_CPU_CACHE
    if _BF16_CPU_CACHE is not None:
        return _BF16_CPU_CACHE
    if _DISABLE_CPU_BF16 or _FORCE_FP32:
        _BF16_CPU_CACHE = False
        return False

    ok = False
    try:
        import torch
        # PyTorch private helper if available (Intel)
        cpu_mod = getattr(torch, "cpu", None)
        if cpu_mod is not None:
            for fn_name in ("_is_cpu_support_avx512_bf16",
                            "_is_cpu_support_amx_bf16"):
                fn = getattr(cpu_mod, fn_name, None)
                if callable(fn):
                    try:
                        if fn():
                            ok = True
                            break
                    except Exception:
                        pass

        # Apple Silicon: arm64 Darwin always supports BF16 inference
        if not ok:
            mach = platform.machine().lower()
            sysname = platform.system()
            if sysname == "Darwin" and mach in ("arm64", "aarch64"):
                ok = True

        # Linux ARM64 (Graviton 3+) usually has BF16 — be conservative,
        # require an explicit opt-in via env var.
        if not ok and platform.machine().lower() in ("aarch64",):
            if os.environ.get("FOOTBALL_ENABLE_ARM_BF16", "0") == "1":
                ok = True
    except Exception:
        ok = False

    _BF16_CPU_CACHE = ok
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Autocast dtype per device
# ─────────────────────────────────────────────────────────────────────────────

def autocast_dtype(device: str):
    """Torch dtype the forward pass should run in. FP32 if forced off."""
    import torch
    if _FORCE_FP32:
        return torch.float32
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    if device == "cpu":
        return torch.bfloat16 if _cpu_supports_bf16() else torch.float32
    return torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# Combined inference context
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def inference_ctx(device: str):
    """inference_mode + device-appropriate autocast.

        with inference_ctx(device):
            y = model(x)

    Always safe to nest with explicit no_grad() — inference_mode is stricter
    than no_grad() and overrides it.
    """
    import torch
    dt = autocast_dtype(device)

    if device == "cuda" and dt == torch.float16:
        ac = torch.autocast(device_type="cuda", dtype=dt, enabled=True)
    elif device == "mps" and dt == torch.float16:
        # MPS autocast landed in PyTorch 2.0
        try:
            ac = torch.autocast(device_type="mps", dtype=dt, enabled=True)
        except (RuntimeError, ValueError):
            # Older torch builds: fall back to no autocast (still correct,
            # just slower). Not an error.
            ac = nullcontext()
    elif device == "cpu" and dt == torch.bfloat16:
        ac = torch.autocast(device_type="cpu", dtype=dt, enabled=True)
    else:
        ac = nullcontext()

    with torch.inference_mode(), ac:
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Model placement
# ─────────────────────────────────────────────────────────────────────────────

def to_eval_device(model, device: str):
    """Move a model to a device, set eval(), and disable grads on params."""
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime providers per device
# ─────────────────────────────────────────────────────────────────────────────

def onnx_providers(device: str):
    """Best provider stack for onnxruntime on this device, priority order.
    Falls back to providers actually present in the installed ORT build.
    Returns None if onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    if device == "cuda":
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "mps":
        wanted = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        wanted = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    providers = [p for p in wanted if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def make_onnx_session(onnx_path: str, device: str):
    """Create an onnxruntime InferenceSession with sensible defaults."""
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if device == "cpu":
        n = max(1, (os.cpu_count() or 2) - 1)
        sess_opts.intra_op_num_threads = n
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = onnx_providers(device)
    return ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                providers=providers)


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(device: str | None = None) -> None:
    """One-line runtime summary. Call once at app startup."""
    import torch
    if device is None:
        device = pick_device()
    dt = autocast_dtype(device)
    dt_name = str(dt).split(".")[-1]

    onnx_ok = False
    try:
        import onnxruntime as ort  # noqa: F401
        onnx_ok = True
    except ImportError:
        pass

    force = " FORCED_FP32" if _FORCE_FP32 else ""
    print(f"[runtime] device={device}  autocast={dt_name}  "
          f"torch_threads={torch.get_num_threads()}  "
          f"cpu_bf16={'yes' if _cpu_supports_bf16() else 'no'}  "
          f"onnx={'yes' if onnx_ok else 'no'}{force}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner()
    d = pick_device()
    print(f"  pick_device('auto') = {d}")
    print(f"  autocast_dtype     = {autocast_dtype(d)}")
    print(f"  onnx_providers     = {onnx_providers(d)}")
    # Smoke test the context manager
    try:
        import torch
        x = torch.randn(4, 4)
        with inference_ctx(d):
            y = x @ x
        print(f"  inference_ctx OK  output_dtype={y.dtype}")
    except Exception as e:
        print(f"  inference_ctx FAILED: {e}")
