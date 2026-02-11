import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from models.criterions import MLMLoss, VTC_VTM_Loss
from models.mask import RandomMaskingGenerator, TubeMaskingGenerator
from models.videomamba import build_videomamba
from models.videomamba.mamba_simple import Mamba
import models.videomamba.videomamba as videomamba_module
from models.videomamba.videomamba import PretrainVideoMamba, create_block, load_state_dict
from utils.config import Config
from utils.config_utils import setup_deepspeed_config, setup_deepspeed_zero_config
from utils.distributed import _parse_slurm_tasks_per_node


class _DummyMultimodalEncoder(torch.nn.Module):
    def forward(
        self,
        encoder_embeds,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        return_dict,
        mode,
    ):
        _ = attention_mask, encoder_hidden_states, encoder_attention_mask, return_dict, mode
        bsz, seqlen, dim = encoder_embeds.shape
        return SimpleNamespace(
            last_hidden_state=torch.randn(bsz, seqlen, dim, device=encoder_embeds.device)
        )


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


def test_load_state_dict_uses_weights_only(tmp_path, monkeypatch):
    model = _small_model()
    ckpt_path = tmp_path / "mini_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)

    seen_kwargs: dict[str, Any] = {}
    original_load = videomamba_module.torch.load

    def _wrapped_load(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(videomamba_module.torch, "load", _wrapped_load)
    load_state_dict(str(ckpt_path), model, ckpt_num_frame=4, num_frames=4)
    assert seen_kwargs.get("weights_only") is True


def test_mamba_forward_requires_cuda_tensor_inputs():
    model = Mamba(
        d_model=8,
        d_state=4,
        d_conv=2,
        expand=2,
        use_fast_path=False,
        layer_idx=0,
    ).eval()
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        model(torch.randn(1, 2, 8))


def test_parse_slurm_tasks_per_node():
    assert _parse_slurm_tasks_per_node("8") == 8
    assert _parse_slurm_tasks_per_node("16(x2)") == 32
    assert _parse_slurm_tasks_per_node("16(x2),8") == 40


def test_setup_deepspeed_zero_config_invalid_stage_raises_value_error():
    with pytest.raises(ValueError, match="Wrong stage for deepspeed 4"):
        setup_deepspeed_zero_config(4)


def test_setup_deepspeed_config_uses_world_size_one_without_dist_init(tmp_path):
    config = SimpleNamespace(
        output_dir=str(tmp_path / "ds_cfg"),
        batch_size=4,
        optimizer=SimpleNamespace(
            lr=1e-4,
            weight_decay=0.01,
            opt_betas=(0.9, 0.999),
        ),
        deepspeed=SimpleNamespace(stage=1, enable=True),
        fp16=True,
        bf16=True,
    )
    config.get = lambda key, default=None: getattr(config, key, default)

    setup_deepspeed_config(config)

    with open(config.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    assert ds_config["train_batch_size"] == 4
    assert ds_config["train_micro_batch_size_per_gpu"] == 4


def test_config_from_file_python_module_cache_does_not_collide(tmp_path):
    cfg_a_dir = tmp_path / "a"
    cfg_b_dir = tmp_path / "b"
    cfg_a_dir.mkdir()
    cfg_b_dir.mkdir()
    cfg_a = cfg_a_dir / "cfg.py"
    cfg_b = cfg_b_dir / "cfg.py"
    cfg_a.write_text("value = 1\n", encoding="utf-8")
    cfg_b.write_text("value = 2\n", encoding="utf-8")

    loaded_a = Config.from_file(str(cfg_a))
    loaded_b = Config.from_file(str(cfg_b))

    assert loaded_a.value == 1
    assert loaded_b.value == 2


def test_vtc_loss_default_all_gather_path_is_safe_without_distributed():
    loss_module = VTC_VTM_Loss(vtm_hard_neg=True)
    vision_proj = torch.randn(3, 2, 8)
    text_proj = torch.randn(3, 8)
    idx = torch.arange(3)
    default_loss = loss_module.vtc_loss(vision_proj, text_proj, idx, temp=0.07)
    local_loss = loss_module.vtc_loss(
        vision_proj, text_proj, idx, temp=0.07, all_gather=False
    )
    torch.testing.assert_close(default_loss, local_loss)


def test_vtm_loss_batch_size_one_returns_zero_loss():
    loss_module = VTC_VTM_Loss(vtm_hard_neg=True)
    encoder = _DummyMultimodalEncoder()
    vtm_head = torch.nn.Linear(8, 2)
    vision_embeds = torch.randn(1, 2, 3, 8, requires_grad=True)
    loss = loss_module.vtm_loss(
        multimodal_encoder=encoder,
        vtm_head=vtm_head,
        temp=0.07,
        vision_embeds=vision_embeds,
        text_embeds=torch.randn(1, 3, 8),
        vision_proj=torch.randn(1, 2, 8),
        text_proj=torch.randn(1, 8),
        text_atts=torch.ones(1, 3, dtype=torch.long),
        idx=torch.arange(1),
    )
    assert loss.shape == ()
    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    loss.backward()
    assert vision_embeds.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this regression")
def test_mlm_mask_accepts_cuda_probability_matrix_without_device_mismatch():
    tokenizer = SimpleNamespace(pad_token_id=0, cls_token_id=101, mask_token_id=103)
    loss_module = MLMLoss(masking_prob=0.15, tokenizer=tokenizer)
    input_ids = torch.tensor([[101, 5, 6, 0], [101, 7, 8, 0]], device="cuda")
    targets = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, 0.15, device="cuda")

    masked_input_ids, masked_targets = loss_module.mask(
        input_ids=input_ids.clone(),
        vocab_size=2048,
        device=input_ids.device,
        targets=targets,
        probability_matrix=probability_matrix,
    )
    assert masked_input_ids.device.type == "cuda"
    assert masked_targets.device.type == "cuda"


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
def test_forward_rejects_frame_count_not_divisible_by_tubelet():
    model = _small_model(kernel_size=2, num_frames=8).cuda().eval()
    x = torch.randn(1, 3, 5, 8, 8, device="cuda")
    with pytest.raises(ValueError, match="must be divisible by tubelet size"):
        model(x, mask=None, use_image=False)
    with pytest.raises(ValueError, match="must be divisible by tubelet size"):
        model.forward_features(x, mask=None, use_image=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_masked_forward_rejects_legacy_mask_shape():
    model = _small_model().cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    legacy_mask = torch.zeros(1, 4 * 2 * 2, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="mask token length mismatch"):
        model(x, mask=legacy_mask, use_image=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_masked_forward_rejects_masked_cls_token():
    model = _small_model().cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    mask = torch.zeros(1, 1 + 4 * 2 * 2, dtype=torch.bool, device="cuda")
    mask[:, 0] = True
    with pytest.raises(ValueError, match="CLS token visible"):
        model(x, mask=mask, use_image=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_masked_forward_rejects_all_patch_tokens_for_avg_pool():
    model = _small_model(pool_type="cls+avg").cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    mask = torch.ones(1, 1 + 4 * 2 * 2, dtype=torch.bool, device="cuda")
    mask[:, 0] = False

    with pytest.raises(ValueError, match="at least one patch token visible"):
        model(x, mask=mask, use_image=False, keep_temporal=False)


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
def test_keep_temporal_masked_forward_supports_nonuniform_visible_per_frame():
    model = _small_model(pool_type="cls+avg").cuda().eval()
    x = torch.randn(2, 3, 4, 8, 8, device="cuda")

    mask = torch.ones(2, 1 + 4 * 2 * 2, dtype=torch.bool, device="cuda")
    # Keep cls and patch tokens with per-frame visible counts [1, 2, 1, 3].
    visible_positions = torch.tensor([0, 1, 5, 6, 9, 13, 14, 15], device="cuda")
    mask[:, visible_positions] = False

    with torch.no_grad():
        _, x_pool, _ = model(x, mask=mask, use_image=False, keep_temporal=True)

    assert x_pool.shape == (2, 4, model.embed_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_keep_temporal_masked_forward_requires_visible_tokens_in_each_frame():
    model = _small_model(pool_type="cls+avg").cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    mask = torch.ones(1, 1 + 4 * 2 * 2, dtype=torch.bool, device="cuda")
    # Keep cls and only frame-0 patch tokens.
    visible_positions = torch.tensor([0, 1, 2], device="cuda")
    mask[:, visible_positions] = False

    with pytest.raises(ValueError, match="at least one visible patch token"):
        model(x, mask=mask, use_image=False, keep_temporal=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_clip_return_layer_gt_one_final_tap_matches_normalized_output():
    model = _small_model(clip_return_layer=2).cuda().eval()
    x = torch.randn(2, 3, 4, 8, 8, device="cuda")

    with torch.no_grad():
        x_vis, x_clip_vis = model.forward_features(x, mask=None, use_image=False)

    assert x_clip_vis is not None
    assert x_clip_vis.shape[0] == 2
    torch.testing.assert_close(x_clip_vis[-1], x_vis, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_use_image_multiframe_masked_path_keeps_batch_dimension():
    model = _small_model().cuda().eval()
    x = torch.randn(2, 3, 3, 8, 8, device="cuda")
    mask = torch.zeros(2, 1 + 3 * 2 * 2, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        x_vis_unmasked, x_pool_unmasked, _ = model(x, mask=None, use_image=True)
        x_vis_masked, x_pool_masked, x_clip = model(x, mask=mask, use_image=True)

    assert x_vis_unmasked.shape == (2, 3 * 2 * 2, model.embed_dim)
    assert x_pool_unmasked.shape[0] == 2
    assert x_vis_masked.shape == (2, 3 * 2 * 2, model.embed_dim)
    assert x_pool_masked.shape[0] == 2
    assert x_clip is not None
    assert x_clip.shape[1] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_masked_forward_rejects_variable_visible_token_counts():
    model = _small_model().cuda().eval()
    x = torch.randn(2, 3, 4, 8, 8, device="cuda")
    mask = torch.zeros(2, 1 + 4 * 2 * 2, dtype=torch.bool, device="cuda")
    mask[0, 3:7] = True
    mask[1, 3:11] = True

    with pytest.raises(ValueError, match="same number of visible tokens"):
        model(x, mask=mask, use_image=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_use_image_mask_length_uses_post_tubelet_temporal_tokens():
    model = _small_model(kernel_size=2, num_frames=4).cuda().eval()
    x = torch.randn(1, 3, 4, 8, 8, device="cuda")
    temporal_tokens = x.shape[2] // model.patch_embed.tubelet_size
    mask = torch.zeros(
        1, 1 + temporal_tokens * model.patch_embed.num_patches, dtype=torch.bool, device="cuda"
    )

    with torch.no_grad():
        x_vis, x_pool, x_clip = model(x, mask=mask, use_image=True)

    assert x_vis.shape == (1, temporal_tokens * model.patch_embed.num_patches, model.embed_dim)
    assert x_pool.shape == (1, 1, model.embed_dim)
    assert x_clip is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_inference_cache_resizes_when_batch_size_changes():
    model = Mamba(
        d_model=8,
        d_state=4,
        d_conv=2,
        expand=2,
        use_fast_path=False,
        layer_idx=0,
    ).cuda().eval()
    cache = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})

    with torch.no_grad():
        out_a = model(torch.randn(2, 1, 8, device="cuda"), inference_params=cache)
        cache.seqlen_offset = 1
        out_b = model(torch.randn(1, 1, 8, device="cuda"), inference_params=cache)

    conv_state, ssm_state = cache.key_value_memory_dict[0]
    assert out_a.shape == (2, 1, 8)
    assert out_b.shape == (1, 1, 8)
    assert conv_state.shape[0] == 1
    assert ssm_state.shape[0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by causal-conv1d")
def test_block_return_state_flag_is_respected():
    block = create_block(
        d_model=16,
        ssm_cfg={"use_fast_path": False},
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
        bimamba=True,
        layer_idx=0,
    ).cuda()
    x = torch.randn(2, 3, 16, device="cuda")
    state = block.mixer.allocate_state(batch_size=2, dtype=x.dtype, device=x.device)

    with torch.no_grad():
        out_without_state = block(x, state=state, return_state=False)
        out_with_state = block(x, state=state, return_state=True)

    assert len(out_without_state) == 2
    assert len(out_with_state) == 3
