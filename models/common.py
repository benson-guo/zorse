# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributed.utils import _p_assert

class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()
        # Adding a dummy parameter, initialized to 0
        self.dummy_param = nn.Parameter(
            nn.init.constant_(nn.Parameter(torch.empty(1)), 0)
        )

    def forward(self, x):
        return x


def _cast_buffers_to_dtype_and_device_patch(
    buffers,
    buffer_dtypes,
    device: torch.device,
) -> None:
    """
    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them
    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the
    corresponding buffer is only moved to ``device``.
    """
    # Patch issue in llama where len(buffers) != len(buffer_dtypes)
    if buffers is not None and len(buffers) > len(buffer_dtypes):
        while len(buffers) != len(buffer_dtypes):
            buffer_dtypes.append(buffer_dtypes[-1])

    _p_assert(
        buffer_dtypes is None or len(buffers) == len(buffer_dtypes),
        f"Expects `buffers` and `buffer_dtypes` to have the same length if "
        f"`buffer_dtypes` is specified but got {len(buffers)} and "
        f"{len(buffer_dtypes)}",
    )
    for buffer, buffer_dtype in zip(buffers, buffer_dtypes):
        if not torch.is_floating_point(buffer) or buffer_dtype is None:
            buffer.data = buffer.to(device=device)
        else:
            buffer.data = buffer.to(device=device, dtype=buffer_dtype)