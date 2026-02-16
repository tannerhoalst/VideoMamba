from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .videomamba.videomamba import Block, create_block

LayerState = Tuple[Tensor, Tensor]
PackedShape = Optional[Tuple[int, int, int]]


class BiMambaRefinerBlock(nn.Module):
    """Bidirectional wrapper around two independent unidirectional VideoMamba blocks."""

    def __init__(
        self,
        dim: int,
        ssm_cfg: Optional[Dict[str, object]] = None,
        **block_kwargs,
    ):
        super().__init__()
        layer_idx = block_kwargs.pop("layer_idx", None)

        self.block_fwd = create_block(
            d_model=dim,
            ssm_cfg=ssm_cfg,
            layer_idx=layer_idx,
            bimamba=False,
            **block_kwargs,
        )

        bwd_layer_idx = None if layer_idx is None else int(layer_idx) + 1_000_000
        self.block_bwd = create_block(
            d_model=dim,
            ssm_cfg=ssm_cfg,
            layer_idx=bwd_layer_idx,
            bimamba=False,
            **block_kwargs,
        )

        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _pack_tokens(x: Tensor) -> Tuple[Tensor, PackedShape]:
        if x.ndim == 3:
            return x, None
        if x.ndim == 4:
            b, t, n, c = x.shape
            return x.reshape(b, t * n, c), (b, t, n)
        raise ValueError("Expected x to be [B, L, C] or [B, T, N, C].")

    @staticmethod
    def _unpack_tokens(x: Tensor, packed_shape: PackedShape) -> Tensor:
        if packed_shape is None:
            return x
        b, t, n = packed_shape
        return x.reshape(b, t, n, x.shape[-1])

    @staticmethod
    def _flip_time(x: Tensor, packed_shape: PackedShape) -> Tensor:
        if packed_shape is None:
            return torch.flip(x, dims=[1])
        b, t, n = packed_shape
        return torch.flip(x.reshape(b, t, n, x.shape[-1]), dims=[1]).reshape(
            b, t * n, x.shape[-1]
        )

    @staticmethod
    def _ensure_state(
        block: Block,
        state: Optional[LayerState],
        batch_size: int,
        device: torch.device,
    ) -> LayerState:
        if state is not None:
            return state
        return block.mixer.allocate_state(batch_size=batch_size, device=device)

    def allocate_state(
        self, batch_size: int, dtype=None, device=None
    ) -> Tuple[LayerState, LayerState]:
        fwd_state = self.block_fwd.mixer.allocate_state(
            batch_size=batch_size, dtype=dtype, device=device
        )
        bwd_state = self.block_bwd.mixer.allocate_state(
            batch_size=batch_size, dtype=dtype, device=device
        )
        return fwd_state, bwd_state

    def forward(
        self,
        x: Tensor,
        state_fwd: Optional[LayerState] = None,
        state_bwd_init: Optional[LayerState] = None,
        use_checkpoint: bool = False,
    ) -> Tuple[Tensor, LayerState]:
        x_seq, packed_shape = self._pack_tokens(x)
        batch_size = x_seq.shape[0]

        fwd_state = self._ensure_state(
            block=self.block_fwd,
            state=state_fwd,
            batch_size=batch_size,
            device=x_seq.device,
        )
        out_fwd, _, new_state_fwd = self.block_fwd(
            x_seq,
            state=fwd_state,
            return_state=True,
            use_checkpoint=use_checkpoint,
        )

        bwd_state = self._ensure_state(
            block=self.block_bwd,
            state=state_bwd_init,
            batch_size=batch_size,
            device=x_seq.device,
        )
        x_rev = self._flip_time(x_seq, packed_shape)
        out_bwd_rev, _, _ = self.block_bwd(
            x_rev,
            state=bwd_state,
            return_state=True,
            use_checkpoint=use_checkpoint,
        )
        out_bwd = self._flip_time(out_bwd_rev, packed_shape)

        gate_input = torch.cat([out_fwd, out_bwd], dim=-1)
        gate = self.fusion_gate(gate_input)
        out = gate * out_fwd + (1.0 - gate) * out_bwd
        out = self.out_proj(out)

        return self._unpack_tokens(out, packed_shape), new_state_fwd
