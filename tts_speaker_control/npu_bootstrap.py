"""torch_npu runtime initialization and CosyVoice device binding."""
from __future__ import annotations

from contextlib import nullcontext


def init_torch_npu(device_id: int = 0) -> str:
    """Initialize torch_npu and return device string (e.g. npu:0)."""
    import torch

    try:
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
        if hasattr(torch_npu.npu, "config"):
            torch_npu.npu.config.allow_internal_format = False
    except ImportError as exc:
        raise RuntimeError("torch_npu not found") from exc

    if not torch.npu.is_available():
        raise RuntimeError("NPU not available")

    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def patch_cosyvoice_for_npu(cosyvoice, device_id: int = 0):
    """Bind CosyVoiceModel to npu device and llm_context."""
    import torch

    device = torch.device(f"npu:{device_id}")
    model = cosyvoice.model
    model.device = device

    for module in (model.llm, model.flow, model.hift):
        module.to(device)

    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            model.llm_context = torch.npu.stream(torch.npu.Stream(device))
        except Exception:
            model.llm_context = nullcontext()
    else:
        model.llm_context = nullcontext()
