# VideoMamba Backbone Utility

## Setup

```shell
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -U torch  # install a CUDA wheel that matches your system
pip install -e .
```

This repository provides a minimal VideoMamba backbone implementation for video applications.
It focuses on the encoder, state handling, checkpoint loading, and chunked processing.

CUDA requirement: this package uses `causal-conv1d` CUDA kernels and does not provide a CPU fallback.

## Quick Usage

```python
from types import SimpleNamespace
from models.videomamba import build_videomamba

config = SimpleNamespace(
    vision_encoder=SimpleNamespace(
        img_size=224,
        patch_size=16,
        depth=24,
        embed_dim=192,
        channels=3,
        drop_path_rate=0.0,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        fused_add_norm=True,
        rms_norm=True,
        residual_in_fp32=True,
        bimamba=True,
        pool_type="cls+avg",
        kernel_size=1,
        num_frames=8,
        use_checkpoint=False,
        checkpoint_num=0,
        pretrained=None,
    )
)

model = build_videomamba(config)
```

## Output API

With default `add_pool_norm=True`:
- `forward(...) -> (x_vis, x_pool)`
- `forward(..., ssm_state=state) -> (x_vis, x_pool, next_state)`

With `add_pool_norm=False`:
- `forward(...) -> x_vis`
- `forward(..., ssm_state=state) -> (x_vis, next_state)`

For feature-only output:
- `forward_features(...) -> x_vis`
- `forward_features(..., ssm_state=state) -> (x_vis, next_state)`

## Checkpoint Contract

- `vision_encoder.channels` is required (no `in_chans` fallback).
- Pretrained checkpoints must be plain `state_dict` files (no `{"model": ...}` / `{"module": ...}` wrapper).
- If `pretrained` is set, `vision_encoder.ckpt_num_frame` must be provided.

## Streaming / Chunked Inference

Mamba state can be carried across chunks using `(conv_state, ssm_state)` per layer.

```python
state = model.init_state(batch_size=2, dtype=x.dtype, device=x.device)
offset_tokens = chunk_start // model.patch_embed.tubelet_size

# returns (x_vis, x_pool, next_state) with default add_pool_norm=True
x_vis, x_pool, state = model(
    x_chunk,
    ssm_state=state,
    temporal_pos_offset=offset_tokens,
)
```

State shapes (per layer):
- `conv_state`: `(B, D_inner, d_conv)`
- `ssm_state`: `(B, D_inner, d_state)`

Note: when using streaming with `keep_temporal=True` on non-initial chunks (`temporal_pos_offset > 0`),
`pool_type='avg'` is supported. CLS-based temporal pooling is chunk-boundary dependent.

## Performance Notes

- Disable fused kernels via `VIDEOMAMBA_DISABLE_FUSED=1` or `ssm_cfg={"use_fast_path": False}`.
- Chunk size trades throughput for memory: larger chunks reduce launch overhead, increase activation memory.
