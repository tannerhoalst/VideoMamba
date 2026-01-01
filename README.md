# VideoMamba Backbone

## Setup and installation

```shell
python3 -m venv .venv
source .venv/bin/activate

pip install ninja setuptools wheel packaging
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 -U
pip install --no-build-isolation -U --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu130 -e .
```

This repository contains a minimal VideoMamba video encoder backbone for use in
other projects. Training scripts, datasets, and alternative backbones have been
removed to keep the package lightweight.

## Usage

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
        clip_decoder_embed_dim=192,
        clip_output_dim=512,
        clip_return_layer=1,
        clip_student_return_interval=1,
        pretrained=None,
    )
)

model = build_videomamba(config)
```

For more details on arguments and outputs, see `models/videomamba/videomamba.py`.

## Streaming training

Chunked training can carry differentiable Mamba state across windows. State is a
tuple of `(conv_state, ssm_state)` per layer.

```python
from models.videomamba.mamba_simple import Mamba

model = Mamba(d_model=192, d_state=16, d_conv=4, expand=2)
x = torch.randn(2, 12, 192)  # (B, L, D)

out1, state = model(x[:, :6], return_state=True)
out2, state = model(x[:, 6:], state=state, return_state=True)
out = torch.cat([out1, out2], dim=1)
```

State shapes (per layer):
- `conv_state`: `(B, D_inner, d_conv)`
- `ssm_state`: `(B, D_inner, d_state)`

For VideoMamba, pass a per-layer list/tuple/dict of states and use
`temporal_pos_offset` to align chunked temporal positions (post-tubelet).

```python
state = model.init_state(batch_size=2, dtype=x.dtype, device=x.device)
offset_tokens = chunk_start // model.patch_embed.tubelet_size
out = model(x_chunk, ssm_state=state, temporal_pos_offset=offset_tokens)
```

## Training Utilities

Optional training utilities (configs, logging, distributed helpers, and loss
functions) are available under `utils/` and `models/criterions.py`. Install them
with:

```shell
pip install -e .[training]
```

Determinism: use `utils.basic_utils.setup_seed(seed, deterministic=True)` to
enable deterministic CUDA kernels (may reduce throughput).

Performance: fused kernels can be disabled by setting
`VIDEOMAMBA_DISABLE_FUSED=1` or by passing `ssm_cfg={"use_fast_path": False}` in
the model config. Chunk size is a throughput vs memory tradeoff: larger chunks
reduce kernel launch overhead but increase activation memory.
