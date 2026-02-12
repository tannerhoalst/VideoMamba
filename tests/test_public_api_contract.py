from typing import Any

import pytest
import torch

import video_mamba
from models.videomamba import build_videomamba as legacy_build_videomamba
from models.videomamba.videomamba import PretrainVideoMamba


def _small_model(**overrides: Any) -> PretrainVideoMamba:
    kwargs: dict[str, Any] = dict(
        img_size=8,
        patch_size=4,
        depth=2,
        embed_dim=16,
        channels=3,
        ssm_cfg={"use_fast_path": False},
        fused_add_norm=False,
        rms_norm=False,
        residual_in_fp32=False,
        kernel_size=1,
        num_frames=4,
    )
    kwargs.update(overrides)
    return PretrainVideoMamba(**kwargs)


def test_top_level_alias_points_to_legacy_builder():
    assert video_mamba.build_videomamba is legacy_build_videomamba


def test_streaming_contract_allocate_and_validate_cpu():
    model = _small_model()
    batch_size = 2

    state = video_mamba.allocate_state(model, batch_size=batch_size, dtype=torch.float32)
    video_mamba.validate_state(model, state, batch_size=batch_size)

    shapes = video_mamba.expected_state_shapes(model, batch_size=batch_size)
    assert len(shapes) == model.depth
    assert shapes[0].conv_state == (batch_size, model.layers[0].mixer.d_inner, 4)
    assert shapes[0].ssm_state == (batch_size, model.layers[0].mixer.d_inner, 16)


def test_model_contract_metadata_and_forward_semantics():
    model = _small_model(add_pool_norm=True)
    assert model.streaming_contract_version == video_mamba.STREAMING_CONTRACT_VERSION

    semantics = model.forward_return_semantics()
    assert semantics.without_state == "(x_vis, x_pool)"
    assert semantics.with_state == "(x_vis, x_pool, next_state)"

    no_pool_model = _small_model(add_pool_norm=False)
    no_pool_semantics = no_pool_model.forward_return_semantics()
    assert no_pool_semantics.without_state == "x_vis"
    assert no_pool_semantics.with_state == "(x_vis, next_state)"


def test_configure_determinism_reseeds_torch_rng():
    video_mamba.configure_determinism(seed=1234, deterministic=True)
    x1 = torch.randn(8)
    video_mamba.configure_determinism(seed=1234, deterministic=True)
    x2 = torch.randn(8)
    torch.testing.assert_close(x1, x2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_minimal_cuda_streaming_forward_contract():
    model = _small_model(add_pool_norm=False).cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    state = video_mamba.allocate_state(model, batch_size=1, dtype=x.dtype, device=x.device)

    with torch.no_grad():
        first_chunk, state = model(
            x[:, :, :2],
            mask=None,
            use_image=False,
            ssm_state=state,
            temporal_pos_offset=0,
        )
        second_chunk, next_state = model(
            x[:, :, 2:],
            mask=None,
            use_image=False,
            ssm_state=state,
            temporal_pos_offset=2,
        )

    video_mamba.validate_state(model, next_state, batch_size=1)
    assert first_chunk.shape == (1, 1 + 2 * 2 * 2, model.embed_dim)
    assert second_chunk.shape == (1, 2 * 2 * 2, model.embed_dim)
