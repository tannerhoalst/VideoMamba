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

## Training Utilities

Optional training utilities (configs, logging, distributed helpers, and loss
functions) are available under `utils/` and `models/criterions.py`. Install them
with:

```shell
pip install -e .[training]
```
