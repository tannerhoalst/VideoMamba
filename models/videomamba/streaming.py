from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple, Union

import torch
from torch import Tensor

STREAMING_CONTRACT_VERSION = "1.0.0"

LayerState = Tuple[Tensor, Tensor]
StreamingState = Union[List[LayerState], Tuple[LayerState, ...], Dict[int, LayerState]]


@dataclass(frozen=True)
class StateShape:
    conv_state: Tuple[int, int, int]
    ssm_state: Tuple[int, int, int]


@dataclass(frozen=True)
class ForwardReturnSemantics:
    without_state: str
    with_state: str


_FORWARD_RETURN_SEMANTICS_BY_POOL_NORM = {
    True: ForwardReturnSemantics(
        without_state="(x_vis, x_pool)",
        with_state="(x_vis, x_pool, next_state)",
    ),
    False: ForwardReturnSemantics(
        without_state="x_vis",
        with_state="(x_vis, next_state)",
    ),
}


class _LayerLike(Protocol):
    mixer: object


class _ModelLike(Protocol):
    layers: Sequence[_LayerLike]
    add_pool_norm: bool


def forward_return_semantics(add_pool_norm: bool) -> ForwardReturnSemantics:
    return _FORWARD_RETURN_SEMANTICS_BY_POOL_NORM[bool(add_pool_norm)]


def model_forward_return_semantics(model: _ModelLike) -> ForwardReturnSemantics:
    return forward_return_semantics(bool(getattr(model, "add_pool_norm", True)))


def expected_state_shapes(model: _ModelLike, batch_size: int) -> Dict[int, StateShape]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    shapes: Dict[int, StateShape] = {}
    for idx, layer in enumerate(model.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            raise TypeError(f"Layer {idx} does not expose a mixer attribute.")
        try:
            d_inner = int(getattr(mixer, "d_inner"))
            d_conv = int(getattr(mixer, "d_conv"))
            d_state = int(getattr(mixer, "d_state"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                f"Layer {idx} mixer does not expose integer d_inner/d_conv/d_state."
            ) from exc
        shapes[idx] = StateShape(
            conv_state=(batch_size, d_inner, d_conv),
            ssm_state=(batch_size, d_inner, d_state),
        )
    return shapes


def allocate_state(
    model: object,
    batch_size: int,
    dtype=None,
    device=None,
    as_dict: bool = False,
) -> StreamingState:
    allocate_fn = getattr(model, "allocate_state", None)
    if callable(allocate_fn):
        return allocate_fn(batch_size, dtype=dtype, device=device, as_dict=as_dict)

    init_fn = getattr(model, "init_state", None)
    if callable(init_fn):
        return init_fn(batch_size, dtype=dtype, device=device, as_dict=as_dict)

    raise TypeError("Model does not expose allocate_state(...) or init_state(...).")


def validate_state(model: _ModelLike, state: StreamingState, batch_size: int) -> None:
    shapes = expected_state_shapes(model, batch_size)
    depth = len(shapes)

    if isinstance(state, dict):
        keys = set(state.keys())
        expected_keys = set(range(depth))
        if keys != expected_keys:
            raise ValueError(
                f"State dict keys mismatch: expected {sorted(expected_keys)}, got {sorted(keys)}."
            )
        items = [state[idx] for idx in range(depth)]
    elif isinstance(state, (list, tuple)):
        if len(state) != depth:
            raise ValueError(f"State length mismatch: expected {depth}, got {len(state)}.")
        items = list(state)
    else:
        raise TypeError("State must be a list, tuple, or dict indexed by layer id.")

    for idx, layer_state in enumerate(items):
        if not isinstance(layer_state, (list, tuple)) or len(layer_state) != 2:
            raise TypeError(
                "Each layer state must be a 2-tuple: (conv_state, ssm_state)."
            )
        conv_state, ssm_state = layer_state
        if not torch.is_tensor(conv_state) or not torch.is_tensor(ssm_state):
            raise TypeError("conv_state and ssm_state must both be tensors.")

        expected = shapes[idx]
        conv_shape = tuple(conv_state.shape)
        ssm_shape = tuple(ssm_state.shape)
        if conv_shape != expected.conv_state:
            raise ValueError(
                f"Layer {idx} conv_state shape mismatch: expected {expected.conv_state}, got {conv_shape}."
            )
        if ssm_shape != expected.ssm_state:
            raise ValueError(
                f"Layer {idx} ssm_state shape mismatch: expected {expected.ssm_state}, got {ssm_shape}."
            )
