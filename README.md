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

## Quick Usage

```python
from types import SimpleNamespace
from video_mamba import build_videomamba

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

Legacy imports continue to work:

```python
from models.videomamba import build_videomamba
```

## Stable Public Imports

- Stable alias: `video_mamba`
- Legacy path (kept for compatibility): `models.videomamba`

The following are part of the stable top-level API:
- `video_mamba.build_videomamba`
- `video_mamba.PretrainVideoMamba`
- `video_mamba.allocate_state`
- `video_mamba.expected_state_shapes`
- `video_mamba.validate_state`
- `video_mamba.STREAMING_CONTRACT_VERSION`

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

Streaming note:
- For continuation chunks (`ssm_state` from `init_state` and `temporal_pos_offset > 0`), CLS is not re-inserted.
- This keeps chunked execution behavior aligned with full-sequence execution.

## Checkpoint Contract

- `vision_encoder.channels` is required (no `in_chans` fallback).
- Pretrained checkpoints must be plain `state_dict` files (no `{"model": ...}` / `{"module": ...}` wrapper).
- If `pretrained` is set, `vision_encoder.ckpt_num_frame` must be provided.

## Streaming / Chunked Inference

Mamba state can be carried across chunks using `(conv_state, ssm_state)` per layer.

```python
from video_mamba import STREAMING_CONTRACT_VERSION, allocate_state

state = allocate_state(model, batch_size=2, dtype=x.dtype, device=x.device)
offset_tokens = chunk_start // model.patch_embed.tubelet_size

# returns (x_vis, x_pool, next_state) with default add_pool_norm=True
x_vis, x_pool, state = model(
    x_chunk,
    ssm_state=state,
    temporal_pos_offset=offset_tokens,
)
```

Contract API version:
- `video_mamba.STREAMING_CONTRACT_VERSION` (current: `1.0.0`)
- `model.streaming_contract_version`

State shapes (per layer):
- `conv_state`: `(B, D_inner, d_conv)`
- `ssm_state`: `(B, D_inner, d_state)`

Forward return semantics contract:
- `add_pool_norm=True`:
  - no state input: `(x_vis, x_pool)`
  - with state input: `(x_vis, x_pool, next_state)`
- `add_pool_norm=False`:
  - no state input: `x_vis`
  - with state input: `(x_vis, next_state)`

Note: when using streaming with `keep_temporal=True` on non-initial chunks (`temporal_pos_offset > 0`),
`pool_type='avg'` is supported. CLS-based pooling requires a CLS token and is not available for continuation chunks.

## Determinism Knobs (Training/Inference)

Use `video_mamba.configure_determinism(...)` (or CLI helpers) in your training and inference entrypoints:

```python
from video_mamba import configure_determinism

configure_determinism(
    seed=7,
    deterministic=True,
    warn_only=True,
    cudnn_benchmark=False,
    allow_tf32=False,
)
```

The streaming check script exposes these flags directly:

```shell
python scripts/check_streaming_state.py \
  --seed 7 \
  --deterministic \
  --deterministic-warn-only \
  --cudnn-benchmark off \
  --allow-tf32 off
```

## CI Compatibility Matrix

GitHub Actions workflow: `.github/workflows/torch-cuda-matrix.yml`

- `torch-2.4.1-cu121`
- `torch-2.5.1-cu124`
- `torch-dev-nightly-cu128` (or override with `TORCH_DEV_INSTALL_CMD`)

Each lane runs a minimal CUDA streaming forward contract test.

## Performance Notes

- Disable fused kernels via `VIDEOMAMBA_DISABLE_FUSED=1` or `ssm_cfg={"use_fast_path": False}`.
- Chunk size trades throughput for memory: larger chunks reduce launch overhead, increase activation memory.

### 5090-class GPU Presets

Preset A (max throughput):
- `chunk_size` (temporal tokens): `64`
- state mode: full streaming state (`allocate_state` contract, carry `(conv_state, ssm_state)`)
- `fused_add_norm=True`, `rms_norm=True`, `ssm_cfg={"use_fast_path": True}`
- determinism: `False` (fastest)

Preset B (balanced):
- `chunk_size` (temporal tokens): `32`
- state mode: full streaming state
- `fused_add_norm=True`, `rms_norm=True`, `ssm_cfg={"use_fast_path": True}`
- determinism: `True` for reproducible runs

Preset C (lowest latency / tight memory):
- `chunk_size` (temporal tokens): `8-16`
- state mode: full streaming state
- `fused_add_norm=False` only if debugging/compatibility requires it (otherwise keep fused on)
- determinism: choose per workload; expect lower throughput when enabled
