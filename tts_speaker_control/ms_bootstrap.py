"""MindSpore runtime initialization for Ascend."""
from __future__ import annotations


def init_mindspore_context(device_id: int = 0):
    import mindspore as ms

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=device_id)
    try:
        ms.set_device("Ascend", device_id)
    except Exception:
        pass
