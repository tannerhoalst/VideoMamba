from types import SimpleNamespace
from typing import Any

import pytest
import torch

from models.mask import RandomMaskingGenerator, TubeMaskingGenerator
from models.videomamba import build_videomamba
from models.videomamba.videomamba import PretrainVideoMamba


def _small_model(**overrides):
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
        clip_decoder_embed_dim=16,
        clip_output_dim=8,
        clip_return_layer=1,
    )
    kwargs.update(overrides)
    return PretrainVideoMamba(**kwargs)


def test_mask_generators_include_cls_by_default():
    mask_tube = TubeMaskingGenerator((4, 2, 2), 0.5, batch=3, device="cpu")
    mask_rand = RandomMaskingGenerator((4, 2, 2), 0.5, batch=3, device="cpu")
    assert mask_tube.dtype == torch.bool
    assert mask_rand.dtype == torch.bool
    assert mask_tube.shape == (3, 1 + 4 * 2 * 2)
    assert mask_rand.shape == (3, 1 + 4 * 2 * 2)
    assert torch.equal(mask_tube[:, 0], torch.zeros(3, dtype=torch.bool))
    assert torch.equal(mask_rand[:, 0], torch.zeros(3, dtype=torch.bool))


def test_bimamba_false_is_rejected():
    with pytest.raises(NotImplementedError, match="bimamba=True"):
        _small_model(bimamba=False)


def test_build_videomamba_namespace_with_pretrained(tmp_path):
    reference = _small_model()
    ckpt_path = tmp_path / "mini_ckpt.pt"
    torch.save(reference.state_dict(), ckpt_path)

    cfg = SimpleNamespace(
        vision_encoder=SimpleNamespace(
            img_size=8,
            patch_size=4,
            depth=2,
            embed_dim=16,
            channels=3,
            drop_path_rate=0.0,
            ssm_cfg={"use_fast_path": False},
            norm_epsilon=1e-5,
            fused_add_norm=False,
            rms_norm=False,
            residual_in_fp32=False,
            bimamba=True,
            pool_type="cls+avg",
            kernel_size=1,
            num_frames=4,
            use_checkpoint=False,
            checkpoint_num=0,
            clip_decoder_embed_dim=16,
            clip_output_dim=8,
            clip_return_layer=1,
            clip_student_return_interval=1,
            pretrained=str(ckpt_path),
            ckpt_num_frame=4,
        )
    )

    model = build_videomamba(cfg)
    assert isinstance(model, PretrainVideoMamba)


def test_load_state_dict_rejects_wrapped_checkpoint(tmp_path):
    wrapped_path = tmp_path / "wrapped.pt"
    torch.save({"model": _small_model().state_dict()}, wrapped_path)
    cfg = SimpleNamespace(
        vision_encoder=SimpleNamespace(
            img_size=8,
            patch_size=4,
            depth=2,
            embed_dim=16,
            channels=3,
            drop_path_rate=0.0,
            ssm_cfg={"use_fast_path": False},
            norm_epsilon=1e-5,
            fused_add_norm=False,
            rms_norm=False,
            residual_in_fp32=False,
            bimamba=True,
            pool_type="cls+avg",
            kernel_size=1,
            num_frames=4,
            use_checkpoint=False,
            checkpoint_num=0,
            clip_decoder_embed_dim=16,
            clip_output_dim=8,
            clip_return_layer=1,
            clip_student_return_interval=1,
            pretrained=str(wrapped_path),
            ckpt_num_frame=4,
        )
    )
    with pytest.raises(ValueError, match="plain state_dict checkpoint"):
        build_videomamba(cfg)


def test_build_videomamba_requires_channels_attr(tmp_path):
    reference = _small_model()
    ckpt_path = tmp_path / "mini_ckpt.pt"
    torch.save(reference.state_dict(), ckpt_path)
    cfg = SimpleNamespace(
        vision_encoder=SimpleNamespace(
            img_size=8,
            patch_size=4,
            depth=2,
            embed_dim=16,
            in_chans=3,
            drop_path_rate=0.0,
            ssm_cfg={"use_fast_path": False},
            norm_epsilon=1e-5,
            fused_add_norm=False,
            rms_norm=False,
            residual_in_fp32=False,
            bimamba=True,
            pool_type="cls+avg",
            kernel_size=1,
            num_frames=4,
            use_checkpoint=False,
            checkpoint_num=0,
            clip_decoder_embed_dim=16,
            clip_output_dim=8,
            clip_return_layer=1,
            clip_student_return_interval=1,
            pretrained=str(ckpt_path),
            ckpt_num_frame=4,
        )
    )
    with pytest.raises(AttributeError):
        build_videomamba(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_clip_return_layer_zero_no_longer_crashes():
    model = _small_model(clip_return_layer=0).cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    with torch.no_grad():
        x_vis, x_pool, x_clip = model(x, mask=None, use_image=False)
    assert x_vis.shape[0] == 1
    assert x_pool.shape[0] == 1
    assert x_clip is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_masked_forward_handles_clip_dim_mismatch_with_strict_mask_shape():
    model = _small_model(clip_decoder_embed_dim=24, clip_output_dim=10).cuda().eval()
    x = torch.randn(2, 3, 4, 8, 8, device="cuda")
    mask = torch.zeros(2, 1 + 4 * 2 * 2, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        _, _, x_clip = model(x, mask=mask, use_image=False)
    assert x_clip is not None
    assert x_clip.shape[0] == 1
    assert x_clip.shape[1] == 2
    assert x_clip.shape[-1] == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_masked_forward_supports_runtime_temporal_length_mismatch():
    model = _small_model(num_frames=8).cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    mask = torch.zeros(1, 1 + 4 * 2 * 2, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        _, _, x_clip = model(x, mask=mask, use_image=False)
    assert x_clip is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_masked_forward_rejects_legacy_mask_shape():
    model = _small_model().cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    legacy_mask = torch.zeros(1, 4 * 2 * 2, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="mask token length mismatch"):
        model(x, mask=legacy_mask, use_image=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_keep_temporal_cls_cat_avg_concatenates_cls_and_temporal_avg():
    model_add = _small_model(pool_type="cls+avg").cuda().eval()
    model_cat = _small_model(pool_type="cls_cat_avg").cuda().eval()
    model_cat.load_state_dict(model_add.state_dict(), strict=True)
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    temporal_tokens = x.shape[2] // model_add.patch_embed.tubelet_size

    with torch.no_grad():
        _, pool_add, _ = model_add(x, mask=None, use_image=False, keep_temporal=True)
        _, pool_cat, _ = model_cat(x, mask=None, use_image=False, keep_temporal=True)

    assert pool_add.shape == (1, temporal_tokens, model_add.embed_dim)
    assert pool_cat.shape == (1, temporal_tokens + 1, model_cat.embed_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_clip_return_layer_gt_one_final_tap_matches_normalized_output():
    model = _small_model(clip_return_layer=2).cuda().eval()
    x = torch.randn(2, 3, 4, 8, 8, device="cuda")

    with torch.no_grad():
        x_vis, x_clip_vis = model.forward_features(x, mask=None, use_image=False)

    assert x_clip_vis is not None
    assert x_clip_vis.shape[0] == 2
    torch.testing.assert_close(x_clip_vis[-1], x_vis, rtol=1e-4, atol=1e-4)
