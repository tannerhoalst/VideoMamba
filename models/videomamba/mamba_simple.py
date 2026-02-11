# Copyright (c) 2023, Tri Dao, Albert Gu.

import inspect
import math
import os
from typing import Any, MutableMapping, Optional, Protocol, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from torch import Tensor

try:
    _SELECTIVE_SCAN_HAS_INITIAL_STATE = (
        "initial_state" in inspect.signature(selective_scan_fn).parameters
    )
except (TypeError, ValueError):
    _SELECTIVE_SCAN_HAS_INITIAL_STATE = False


class InferenceParamsLike(Protocol):
    seqlen_offset: int
    key_value_memory_dict: MutableMapping[int, Tuple[Tensor, Tensor]]


def _selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    initial_state=None,
    return_last_state=False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(
                rearrange(B.float(), "... (L two) -> ... L two", two=2)
            )
        if is_variable_C:
            C = torch.view_as_complex(
                rearrange(C.float(), "... (L two) -> ... L two", two=2)
            )
    else:
        B = B.float()
        C = C.float()
    if initial_state is None:
        x = A.new_zeros((batch, dim, dstate))
    else:
        if initial_state.stride(-1) != 1:
            initial_state = initial_state.contiguous()
        x = initial_state.float()
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    if return_last_state:
        assert last_state is not None
        return out, last_state
    return out


def _selective_scan_with_state(
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    initial_state=None,
    return_last_state=False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if initial_state is None:
        return cast(
            Union[Tensor, Tuple[Tensor, Tensor]],
            selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                D,
                z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=return_last_state,
            ),
        )
    if _SELECTIVE_SCAN_HAS_INITIAL_STATE:
        selective_scan_fn_any = cast(Any, selective_scan_fn)
        return selective_scan_fn_any(
            x,
            dt,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            initial_state=initial_state,
            return_last_state=return_last_state,
        )
    if initial_state.stride(-1) != 1:
        initial_state = initial_state.contiguous()
    batch, dim, seqlen = x.shape
    out = torch.empty((batch, dim, seqlen), device=x.device, dtype=x.dtype)
    state = initial_state
    for i in range(seqlen):
        z_i = z[:, :, i] if z is not None else None
        out[:, :, i] = selective_state_update(
            state,
            x[:, :, i],
            dt[:, :, i],
            A,
            B[:, :, i],
            C[:, :, i],
            D,
            z=z_i,
            dt_bias=delta_bias,
            dt_softplus=delta_softplus,
        )
    return out if not return_last_state else (out, state)


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,  # Fused kernel options
        layer_idx: Optional[int] = None,
        bimamba: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **_: Any,
    ):
        factory_kwargs: dict[str, Any] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        if not bimamba:
            raise NotImplementedError(
                "This minimal VideoMamba package only supports bimamba=True."
            )
        disable_fused = os.getenv("VIDEOMAMBA_DISABLE_FUSED", "").lower()
        if disable_fused in {"1", "true", "yes", "y", "on"}:
            use_fast_path = False
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        assert self.dt_proj.bias is not None
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        setattr(self.dt_proj.bias, "_no_reinit", True)

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        setattr(self.A_log, "_no_weight_decay", True)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        setattr(self.D, "_no_weight_decay", True)

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(
        self,
        hidden_states: Tensor,
        inference_params: Optional[InferenceParamsLike] = None,
        ssm_state: Optional[Tensor] = None,
        state: Optional[Tuple[Tensor, Tensor]] = None,
        return_state: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states unless return_state=True.

        Args:
            state: Optional tuple (conv_state, ssm_state) for streaming training when
                inference_params is None.
            return_state: Whether to return the updated (conv_state, ssm_state).
        """
        if state is not None and ssm_state is not None:
            raise ValueError("Pass either state or ssm_state, not both.")
        if inference_params is not None and state is not None:
            raise ValueError("state is not supported with inference_params.")

        batch, seqlen, dim = hidden_states.shape

        conv_state = None
        if state is not None:
            conv_state, ssm_state = state

        use_inference_params = inference_params is not None
        if use_inference_params:
            conv_state, cache_state = self._get_states_from_cache(
                inference_params, batch
            )
            if ssm_state is None:
                ssm_state = cache_state
            if inference_params.seqlen_offset > 0:
                assert conv_state is not None
                assert ssm_state is not None
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
            # Preserve inference cache behavior; ignore return_state in this path.
            return_state = False

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path
            and not use_inference_params
            and ssm_state is None
            and conv_state is None
            and not return_state
        ):
            out = cast(
                Tensor,
                mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                ),
            )
        else:
            x, z = xz.chunk(2, dim=1)
            x_in = x
            # Compute short convolution
            if use_inference_params:
                assert conv_state is not None
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(x_in, (self.d_conv - x_in.shape[-1], 0))
                )  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            new_conv_state = None
            if conv_state is not None and not use_inference_params:
                x_cat = torch.cat([conv_state, x_in], dim=-1)
                x = causal_conv1d_fn(
                    x=x_cat,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                assert x is not None
                x = x[..., -seqlen:]
                if return_state:
                    new_conv_state = x_cat[..., -self.d_conv :]
            else:
                x = causal_conv1d_fn(
                    x=x_in,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                assert x is not None
                if return_state:
                    new_conv_state = F.pad(
                        x_in, (self.d_conv - x_in.shape[-1], 0)
                    )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            new_ssm_state = None
            use_inplace_ssm = use_inference_params or (
                ssm_state is not None and state is None and not return_state
            )
            return_last_state = return_state or use_inplace_ssm
            scan_out = _selective_scan_with_state(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                initial_state=ssm_state,
                return_last_state=return_last_state,
            )
            if return_last_state:
                y, last_state = cast(Tuple[Tensor, Tensor], scan_out)
                if use_inplace_ssm:
                    assert ssm_state is not None
                    ssm_state.copy_(last_state)
                else:
                    new_ssm_state = last_state
            else:
                y = cast(Tensor, scan_out)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        if return_state:
            assert new_conv_state is not None
            assert new_ssm_state is not None
            return out, (new_conv_state, new_ssm_state)
        return out

    def step(
        self, hidden_states: Tensor, conv_state: Tensor, ssm_state: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        x = causal_conv1d_update(
            x,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        y = selective_state_update(
            ssm_state,
            x,
            dt,
            A,
            B,
            C,
            self.D,
            z=z,
            dt_bias=self.dt_proj.bias,
            dt_softplus=True,
        )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def allocate_state(
        self, batch_size: int, dtype=None, device=None
    ) -> Tuple[Tensor, Tensor]:
        """Allocate zero (conv_state, ssm_state) for streaming training."""
        if device is None:
            device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self,
        inference_params: InferenceParamsLike,
        batch_size: int,
        initialize_states: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        assert self.layer_idx is not None
        def _allocate_states() -> Tuple[Tensor, Tensor]:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            return conv_state, ssm_state

        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state, ssm_state = _allocate_states()
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            if conv_state.shape[0] != batch_size or ssm_state.shape[0] != batch_size:
                conv_state, ssm_state = _allocate_states()
                inference_params.key_value_memory_dict[self.layer_idx] = (
                    conv_state,
                    ssm_state,
                )
            elif initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
