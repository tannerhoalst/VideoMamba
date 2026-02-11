# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import logging
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import _cfg, _load_weights
from torch import Tensor

from .mamba_simple import InferenceParamsLike, Mamba

logger = logging.getLogger(__name__)

LayerState = Union[Tensor, Tuple[Tensor, Tensor]]
StateCollection = Union[List[LayerState], Tuple[LayerState, ...], Dict[int, LayerState]]


class NormLayerProtocol(Protocol):
    weight: Tensor
    bias: Optional[Tensor]
    eps: float

    def __call__(self, __x: Tensor) -> Tensor:
        ...


class MixerProtocol(Protocol):
    def __call__(
        self,
        hidden_states: Tensor,
        inference_params: Optional[InferenceParamsLike] = None,
        ssm_state: Optional[Tensor] = None,
        state: Optional[Tuple[Tensor, Tensor]] = None,
        return_state: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        ...

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        ...

    def allocate_state(self, batch_size: int, dtype=None, device=None) -> Tuple[Tensor, Tensor]:
        ...


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mixer_cls: Callable[[int], MixerProtocol],
        norm_cls: Callable[[int], nn.Module] = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = cast(MixerProtocol, mixer_cls(dim))
        self.norm = cast(NormLayerProtocol, norm_cls(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params: Optional[InferenceParamsLike] = None,
        use_checkpoint: bool = False,
        ssm_state: Optional[Tensor] = None,
        state: Optional[Tuple[Tensor, Tensor]] = None,
        return_state: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]]:
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            state: Optional tuple (conv_state, ssm_state) for streaming training.
            return_state: Whether to return the updated (conv_state, ssm_state).
        """
        if state is not None and ssm_state is not None:
            raise ValueError("Pass either state or ssm_state, not both.")
        if not self.fused_add_norm:
            residual = (
                (residual + self.drop_path(hidden_states))
                if residual is not None
                else hidden_states
            )
            assert residual is not None
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = cast(
                Tuple[Tensor, Tensor],
                fused_add_norm_fn(
                    hidden_states if residual is None else self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                ),
            )
        new_state: Optional[Tuple[Tensor, Tensor]] = None
        if use_checkpoint:
            if state is not None:
                if return_state:
                    hidden_states, new_state = cast(
                        Tuple[Tensor, Tuple[Tensor, Tensor]],
                        checkpoint.checkpoint(
                            self.mixer,
                            hidden_states,
                            inference_params,
                            None,
                            state,
                            True,
                            use_reentrant=False,
                        ),
                    )
                else:
                    hidden_states = cast(
                        Tensor,
                        checkpoint.checkpoint(
                            self.mixer,
                            hidden_states,
                            inference_params,
                            None,
                            state,
                            False,
                            use_reentrant=False,
                        ),
                    )
            else:
                hidden_states = cast(
                    Tensor,
                    checkpoint.checkpoint(
                        self.mixer,
                        hidden_states,
                        inference_params,
                        ssm_state,
                        use_reentrant=False,
                    ),
                )
        else:
            if state is not None:
                if return_state:
                    hidden_states, new_state = cast(
                        Tuple[Tensor, Tuple[Tensor, Tensor]],
                        self.mixer(
                            hidden_states,
                            inference_params=inference_params,
                            state=state,
                            return_state=True,
                        ),
                    )
                else:
                    hidden_states = cast(
                        Tensor,
                        self.mixer(
                            hidden_states,
                            inference_params=inference_params,
                            state=state,
                            return_state=False,
                        ),
                    )
            else:
                hidden_states = cast(
                    Tensor,
                    self.mixer(
                        hidden_states,
                        inference_params=inference_params,
                        ssm_state=ssm_state,
                    ),
                )
        if state is not None and return_state:
            assert new_state is not None
            assert residual is not None
            return hidden_states, residual, new_state
        if state is not None:
            assert residual is not None
            return hidden_states, residual
        assert residual is not None
        return hidden_states, residual

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model: int,
    ssm_cfg: Optional[Dict[str, object]] = None,
    norm_epsilon: float = 1e-5,
    drop_path: float = 0.0,
    rms_norm: bool = True,
    residual_in_fp32: bool = True,
    fused_add_norm: bool = True,
    layer_idx: Optional[int] = None,
    bimamba: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    factory_kwargs: Dict[str, object] = {}
    if device is not None:
        factory_kwargs["device"] = device
    if dtype is not None:
        factory_kwargs["dtype"] = dtype
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(
        Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs
    )
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    object.__setattr__(block, "layer_idx", layer_idx)
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        kernel_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size_2d = cast(Tuple[int, int], to_2tuple(img_size))
        patch_size_2d = cast(Tuple[int, int], to_2tuple(patch_size))
        num_patches = (img_size_2d[1] // patch_size_2d[1]) * (
            img_size_2d[0] // patch_size_2d[0]
        )
        self.img_size = img_size_2d
        self.patch_size = patch_size_2d
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(kernel_size, patch_size_2d[0], patch_size_2d[1]),
            stride=(kernel_size, patch_size_2d[0], patch_size_2d[1]),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 768,
        embed_dim: int = 768,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.head = nn.Linear(embed_dim, output_dim)
        self.norm = norm_layer(output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


def get_sinusoid_encoding_table_torch(
    n_position: int,
    d_hid: int,
    offset: int = 0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if n_position <= 0:
        raise ValueError("n_position must be positive.")
    if d_hid <= 0:
        raise ValueError("d_hid must be positive.")
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    angles = (
        torch.arange(offset, offset + n_position, device=device, dtype=torch.float32)
        .unsqueeze(1)
        .div(
            torch.pow(
                10000.0,
                (
                    2
                    * torch.floor(
                        torch.arange(d_hid, device=device, dtype=torch.float32) / 2
                    )
                    / d_hid
                ),
            )
        )
    )
    sinusoid_table = torch.empty_like(angles)
    sinusoid_table[:, 0::2] = torch.sin(angles[:, 0::2])
    sinusoid_table[:, 1::2] = torch.cos(angles[:, 1::2])
    return sinusoid_table.unsqueeze(0).to(dtype=torch.float32 if dtype is None else dtype)


class PretrainVideoMamba(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 24,
        embed_dim: int = 192,
        channels: int = 3,
        drop_path_rate: float = 0.0,
        ssm_cfg: Optional[Dict[str, object]] = None,
        norm_epsilon: float = 1e-5,
        initializer_cfg: Optional[Dict[str, object]] = None,
        fused_add_norm: bool = True,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        bimamba: bool = True,
        pool_type: str = "cls+avg",
        # video
        kernel_size: int = 1,
        num_frames: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # checkpoint
        use_checkpoint: bool = False,
        checkpoint_num: int = 0,
        # clip,
        clip_decoder_embed_dim: Optional[int] = None,
        clip_output_dim: int = 512,
        clip_return_layer: int = 1,
        clip_student_return_interval: int = 1,
        add_pool_norm: bool = True,
    ):
        factory_kwargs: Dict[str, object] = {}  # follow MambaLMHeadModel
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        super().__init__()
        if not bimamba:
            raise NotImplementedError(
                "This minimal VideoMamba package only supports bimamba=True."
            )
        if clip_return_layer < 0:
            raise ValueError("clip_return_layer must be >= 0.")
        if clip_student_return_interval <= 0:
            raise ValueError("clip_student_return_interval must be > 0.")
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.clip_decoder_embed_dim = (
            embed_dim if clip_decoder_embed_dim is None else clip_decoder_embed_dim
        )
        logger.info(f"Use checkpoint: {use_checkpoint}")
        logger.info(f"Checkpoint number: {checkpoint_num}")
        self.return_index = []
        for i in range(clip_return_layer):
            layer_index = depth - int(i * clip_student_return_interval) - 1
            if layer_index < 0:
                raise ValueError(
                    "clip_return_layer and clip_student_return_interval exceed model depth."
                )
            self.return_index.append(layer_index)
        logger.info(f"Student return index: {self.return_index}")
        self.depth = depth
        self.pool_type = pool_type
        logger.info(f"Pool type: {pool_type}")

        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames // kernel_size, embed_dim)
        )
        self.clip_input_proj: nn.Module
        if self.clip_decoder_embed_dim == embed_dim:
            self.clip_input_proj = nn.Identity()
        else:
            self.clip_input_proj = nn.Linear(
                embed_dim,
                self.clip_decoder_embed_dim,
                **cast(Dict[str, Any], factory_kwargs),
            )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **cast(Dict[str, Any], factory_kwargs),
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm = cast(
            NormLayerProtocol,
            (nn.LayerNorm if not rms_norm else RMSNorm)(
                embed_dim, eps=norm_epsilon, **cast(Dict[str, Any], factory_kwargs)
            ),
        )

        # CLIP decoder
        self.clip_decoder = nn.ModuleList(
            [
                Linear_Decoder(
                    output_dim=clip_output_dim,
                    embed_dim=self.clip_decoder_embed_dim,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(clip_return_layer)
            ]
        )

        self.clip_pos_embed = get_sinusoid_encoding_table(
            num_patches * num_frames // kernel_size + 1, self.clip_decoder_embed_dim
        )
        self.clip_img_pos_embed = get_sinusoid_encoding_table(
            num_patches + 1, self.clip_decoder_embed_dim
        )

        self.add_pool_norm = add_pool_norm
        if add_pool_norm:
            self.pool_norm = nn.LayerNorm(embed_dim)

        # original init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None, **kwargs
    ) -> Dict[int, Tuple[Tensor, Tensor]]:
        return {
            i: cast(Block, layer_module).allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer_module in enumerate(self.layers)
        }

    def init_ssm_state(
        self, batch_size: int, dtype=None, device=None, as_dict: bool = False
    ) -> Union[List[Tensor], Dict[int, Tensor]]:
        if as_dict:
            states_dict: Dict[int, Tensor] = {}
            for idx, layer_module in enumerate(self.layers):
                layer = cast(Block, layer_module)
                _, layer_state = layer.allocate_inference_cache(
                    batch_size, max_seqlen=1, dtype=dtype
                )
                if device is not None:
                    layer_state = layer_state.to(device=device)
                states_dict[idx] = layer_state
            return states_dict
        states_list: List[Tensor] = []
        for _, layer_module in enumerate(self.layers):
            layer = cast(Block, layer_module)
            _, layer_state = layer.allocate_inference_cache(
                batch_size, max_seqlen=1, dtype=dtype
            )
            if device is not None:
                layer_state = layer_state.to(device=device)
            states_list.append(layer_state)
        return states_list

    def init_state(
        self, batch_size: int, dtype=None, device=None, as_dict: bool = False
    ) -> Union[List[Tuple[Tensor, Tensor]], Dict[int, Tuple[Tensor, Tensor]]]:
        if as_dict:
            states_dict: Dict[int, Tuple[Tensor, Tensor]] = {}
            for idx, layer_module in enumerate(self.layers):
                layer = cast(Block, layer_module)
                mixer = layer.mixer
                states_dict[idx] = mixer.allocate_state(
                    batch_size, dtype=dtype, device=device
                )
            return states_dict
        states_list: List[Tuple[Tensor, Tensor]] = []
        for _, layer_module in enumerate(self.layers):
            layer = cast(Block, layer_module)
            mixer = layer.mixer
            states_list.append(
                mixer.allocate_state(batch_size, dtype=dtype, device=device)
            )
        return states_list

    @torch.jit.ignore()
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(cast(Any, self), checkpoint_path, prefix)

    def _get_layer_state(
        self, state: Optional[StateCollection], layer_idx: int
    ) -> Optional[LayerState]:
        if state is None:
            return None
        if isinstance(state, dict):
            return state.get(layer_idx)
        if isinstance(state, (list, tuple)):
            return state[layer_idx]
        raise TypeError("state must be a list, tuple, or dict indexed by layer id")

    def _validate_temporal_length(self, frame_count: int) -> int:
        tubelet = self.patch_embed.tubelet_size
        if frame_count <= 0:
            raise ValueError("Input must contain at least one frame.")
        if frame_count % tubelet != 0:
            raise ValueError(
                f"Input frame count ({frame_count}) must be divisible by tubelet size ({tubelet})."
            )
        return frame_count // tubelet

    def _get_temporal_pos_embedding(
        self, seqlen: int, offset: int = 0, dtype=None, device=None
    ) -> Tensor:
        """Return temporal positional embeddings for a slice starting at offset."""
        if offset < 0:
            raise ValueError("temporal_pos_offset must be non-negative.")
        if device is None:
            device = self.temporal_pos_embedding.device
        if dtype is None:
            dtype = self.temporal_pos_embedding.dtype
        pos_embed = self.temporal_pos_embedding.to(device=device, dtype=dtype)
        pos_len = pos_embed.shape[1]
        end = offset + seqlen
        if end <= pos_len:
            return pos_embed[:, offset:end]
        pos = pos_embed.permute(0, 2, 1)
        pos = torch.nn.functional.interpolate(
            pos.float(), size=end, mode="linear", align_corners=False
        )
        pos = pos.permute(0, 2, 1).to(dtype=dtype)
        return pos[:, offset:end]

    def _normalize_mask(
        self, mask: Optional[Tensor], batch_size: int, token_count: int, device: torch.device
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.ndim != 2:
            raise ValueError("mask must be 2D with shape [B, N].")
        if mask.shape[0] != batch_size:
            raise ValueError(
                f"mask batch size mismatch: expected {batch_size}, got {mask.shape[0]}."
            )
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.shape[1] != token_count:
            raise ValueError(
                f"mask token length mismatch: expected {token_count}, got {mask.shape[1]}."
            )
        if token_count > 0 and torch.any(mask[:, 0]):
            raise ValueError("mask must keep CLS token visible (mask[:, 0] must be False).")
        return mask

    def _masked_temporal_average(
        self,
        patch_tokens: Tensor,
        visible_positions: Tensor,
        temporal_tokens: int,
    ) -> Tensor:
        """Average visible patch tokens per temporal slice under arbitrary masking."""
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [B, N, C].")
        if visible_positions.ndim != 2:
            raise ValueError("visible_positions must have shape [B, N_total_visible].")
        if patch_tokens.shape[0] != visible_positions.shape[0]:
            raise ValueError("Batch size mismatch between patch_tokens and visible_positions.")
        if visible_positions.shape[1] != patch_tokens.shape[1] + 1:
            raise ValueError(
                "visible_positions must include one CLS position in addition to patch tokens."
            )
        if visible_positions.numel() > 0 and not torch.all(visible_positions[:, 0] == 0):
            raise ValueError("mask must keep CLS token visible for temporal pooling.")

        tokens_per_frame = self.patch_embed.num_patches
        patch_positions = visible_positions[:, 1:] - 1
        frame_indices = torch.div(
            patch_positions, tokens_per_frame, rounding_mode="floor"
        ).to(dtype=torch.long)

        B, _, C = patch_tokens.shape
        temporal_sum = torch.zeros(
            B, temporal_tokens, C, device=patch_tokens.device, dtype=patch_tokens.dtype
        )
        temporal_sum.scatter_add_(
            1, frame_indices.unsqueeze(-1).expand(-1, -1, C), patch_tokens
        )

        temporal_counts = torch.zeros(
            B, temporal_tokens, 1, device=patch_tokens.device, dtype=patch_tokens.dtype
        )
        ones = torch.ones(
            B, patch_tokens.shape[1], 1, device=patch_tokens.device, dtype=patch_tokens.dtype
        )
        temporal_counts.scatter_add_(1, frame_indices.unsqueeze(-1), ones)

        if torch.any(temporal_counts == 0):
            raise ValueError(
                "keep_temporal with masking requires at least one visible patch token "
                "for each temporal slice."
            )
        return temporal_sum / temporal_counts

    def _visible_token_positions(
        self, mask: Optional[Tensor], batch_size: int, token_count: int, device: torch.device
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Normalize mask and return visible token indices per sample."""
        normalized_mask = self._normalize_mask(mask, batch_size, token_count, device)
        if normalized_mask is None:
            return None, None

        visible_mask = ~normalized_mask
        visible_counts = visible_mask.sum(dim=1)
        if visible_counts.numel() > 0 and not torch.all(visible_counts == visible_counts[0]):
            raise ValueError(
                "mask must keep the same number of visible tokens per sample; "
                f"got per-sample counts: {visible_counts.tolist()}."
            )
        if visible_counts.numel() > 0 and int(visible_counts[0].item()) <= 0:
            raise ValueError("mask must keep at least one visible token per sample.")

        token_positions = torch.arange(token_count, device=device).unsqueeze(0).expand(
            batch_size, -1
        )
        token_positions = token_positions.masked_fill(~visible_mask, token_count)
        num_visible = int(visible_counts[0].item()) if visible_counts.numel() > 0 else 0
        visible_positions = torch.sort(token_positions, dim=1).values[:, :num_visible]
        return normalized_mask, visible_positions

    def _get_clip_pos_embedding(
        self,
        batch_size: int,
        temporal_tokens: int,
        use_image: bool,
        temporal_pos_offset: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if use_image:
            tokens_per_frame = self.patch_embed.num_patches
            expected_tokens = 1 + temporal_tokens * tokens_per_frame
            if temporal_tokens == 1 and temporal_pos_offset == 0:
                pos_embed = self.clip_img_pos_embed.to(device=device, dtype=dtype)
                if expected_tokens > pos_embed.shape[1]:
                    pos_embed = get_sinusoid_encoding_table_torch(
                        expected_tokens,
                        self.clip_decoder_embed_dim,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    pos_embed = pos_embed[:, :expected_tokens]
                return pos_embed.repeat(batch_size, 1, 1)

        if temporal_pos_offset < 0:
            raise ValueError("temporal_pos_offset must be non-negative.")
        tokens_per_frame = self.patch_embed.num_patches
        start_token = 1 + temporal_pos_offset * tokens_per_frame
        end_token = start_token + temporal_tokens * tokens_per_frame
        pos_embed = self.clip_pos_embed.to(device=device, dtype=dtype)
        if end_token > pos_embed.shape[1]:
            pos_embed = get_sinusoid_encoding_table_torch(
                end_token,
                self.clip_decoder_embed_dim,
                device=device,
                dtype=dtype,
            )
        cls_token = pos_embed[:, :1]
        spatial_tokens = pos_embed[:, start_token:end_token]
        return torch.cat([cls_token, spatial_tokens], dim=1).repeat(batch_size, 1, 1)

    def forward_features(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        use_image: bool = False,
        ssm_state: Optional[StateCollection] = None,
        temporal_pos_offset: int = 0,
    ) -> Union[
        Tuple[Tensor, Optional[Tensor]],
        Tuple[Tensor, Optional[Tensor], StateCollection],
    ]:
        """Forward features with temporal position slicing.

        Args:
            temporal_pos_offset: Start index in temporal tokens (post-tubelet).
            ssm_state: Optional per-layer state (ssm_state or (conv_state, ssm_state)).
        """
        if x.ndim != 5:
            raise ValueError("x must have shape [B, C, T, H, W].")
        self._validate_temporal_length(x.shape[2])
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape

        if use_image:
            x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            cls_pos = self.pos_embed[:, :1]
            spatial_pos = self.pos_embed[:, 1:].repeat(1, T, 1)
            x = x + torch.cat((cls_pos, spatial_pos), dim=1)
        else:
            x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
            cls_token = self.cls_token.expand(
                x.shape[0], -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed
            # temporal pos
            cls_tokens = x[:B, :1, :]
            x = x[:, 1:]
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
            temporal_pos_embed = self._get_temporal_pos_embedding(
                T,
                offset=temporal_pos_offset,
                dtype=x.dtype,
                device=x.device,
            )
            x = x + temporal_pos_embed
            x = rearrange(x, "(b n) t m -> b (t n) m", b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        # mask
        mask, visible_positions = self._visible_token_positions(mask, B, x.shape[1], x.device)
        if visible_positions is not None:
            x_vis = x.gather(1, visible_positions.unsqueeze(-1).expand(-1, -1, C))
        else:
            x_vis = x
        x_clip_vis_list: List[Tensor] = []

        # mamba impl
        residual = None
        hidden_states = x_vis
        new_states: Optional[Union[Dict[int, LayerState], List[Optional[LayerState]]]] = None
        return_tuple_state = False
        for idx, layer_module in enumerate(self.layers):
            layer = cast(Block, layer_module)
            layer_state = self._get_layer_state(ssm_state, idx)
            is_full_state = isinstance(layer_state, (list, tuple)) and len(layer_state) == 2
            if is_full_state and new_states is None:
                if isinstance(ssm_state, dict):
                    new_states = {}
                else:
                    new_states = [
                        cast(Optional[LayerState], None) for _ in range(len(self.layers))
                    ]
                    return_tuple_state = isinstance(ssm_state, tuple)
            if self.use_checkpoint and idx < self.checkpoint_num:
                if is_full_state:
                    hidden_states, residual, layer_state = layer(
                        hidden_states,
                        residual,
                        inference_params=None,
                        use_checkpoint=True,
                        state=layer_state,
                        return_state=True,
                    )
                else:
                    hidden_states, residual = layer(
                        hidden_states,
                        residual,
                        inference_params=None,
                        use_checkpoint=True,
                        ssm_state=layer_state,
                    )
            else:
                if is_full_state:
                    hidden_states, residual, layer_state = layer(
                        hidden_states,
                        residual,
                        inference_params=None,
                        state=layer_state,
                        return_state=True,
                    )
                else:
                    hidden_states, residual = layer(
                        hidden_states,
                        residual,
                        inference_params=None,
                        ssm_state=layer_state,
                    )
            if new_states is not None:
                if isinstance(new_states, dict):
                    assert layer_state is not None
                    new_states[idx] = layer_state
                else:
                    new_states[idx] = layer_state
            if (idx - 1) in self.return_index:
                assert residual is not None
                x_clip_vis_list.append(
                    self.norm(residual.to(dtype=self.norm.weight.dtype))
                )  # share norm for mask

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states = cast(
                Tensor,
                fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    eps=self.norm.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                ),
            )

        if (self.depth - 1) in self.return_index:
            if residual is None:
                x_clip_vis_list.append(hidden_states)
            else:
                x_clip_vis_list.append(self.norm(residual.to(dtype=self.norm.weight.dtype)))

        x_vis = hidden_states
        x_clip_vis = torch.stack(x_clip_vis_list) if x_clip_vis_list else None

        if new_states is not None and return_tuple_state:
            maybe_states = cast(List[Optional[LayerState]], new_states)
            finalized_states: List[LayerState] = []
            for state_item in maybe_states:
                if state_item is None:
                    raise ValueError("Expected full state for all layers.")
                finalized_states.append(state_item)
            return x_vis, x_clip_vis, tuple(finalized_states)
        if ssm_state is None:
            return x_vis, x_clip_vis
        if new_states is not None:
            if isinstance(new_states, list):
                finalized_states: List[LayerState] = []
                for state_item in new_states:
                    if state_item is None:
                        raise ValueError("Expected full state for all layers.")
                    finalized_states.append(state_item)
                return x_vis, x_clip_vis, finalized_states
            return x_vis, x_clip_vis, new_states
        return x_vis, x_clip_vis, ssm_state

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        use_image: bool = False,
        keep_temporal: bool = False,
        ssm_state: Optional[StateCollection] = None,
        temporal_pos_offset: int = 0,
    ) -> Union[
        Tuple[Tensor, Tensor, Optional[Tensor]],
        Tuple[Tensor, Tensor, Optional[Tensor], StateCollection],
        Tuple[Tensor, Optional[Tensor]],
        Tuple[Tensor, Optional[Tensor], StateCollection],
    ]:
        """Forward with optional temporal position offset.

        Args:
            temporal_pos_offset: Start index in temporal tokens (post-tubelet).
            ssm_state: Optional per-layer state (ssm_state or (conv_state, ssm_state)).
        """
        if x.ndim != 5:
            raise ValueError("x must have shape [B, C, T, H, W].")
        temporal_tokens = self._validate_temporal_length(x.shape[2])
        features = self.forward_features(
            x,
            mask,
            use_image,
            ssm_state=ssm_state,
            temporal_pos_offset=temporal_pos_offset,
        )  # [B, N_vis, C_e]
        if ssm_state is None:
            x_vis, x_clip_vis = cast(Tuple[Tensor, Optional[Tensor]], features)
        else:
            x_vis, x_clip_vis, ssm_state = cast(
                Tuple[Tensor, Optional[Tensor], StateCollection], features
            )

        # align CLIP:
        if mask is not None and x_clip_vis is not None and x_clip_vis.shape[0] > 0:
            K, B, _, _ = x_clip_vis.shape
            full_token_count = 1 + temporal_tokens * self.patch_embed.num_patches
            mask, visible_positions = self._visible_token_positions(
                mask, B, full_token_count, x.device
            )
            assert mask is not None
            assert visible_positions is not None
            x_clip_vis = cast(Tensor, self.clip_input_proj(x_clip_vis))
            C_CLIP = x_clip_vis.shape[-1]
            expand_clip_pos_embed = self._get_clip_pos_embedding(
                batch_size=B,
                temporal_tokens=temporal_tokens,
                use_image=use_image,
                temporal_pos_offset=temporal_pos_offset,
                dtype=x_clip_vis.dtype,
                device=x.device,
            )
            clip_pos_emd_vis = expand_clip_pos_embed.gather(
                1, visible_positions.unsqueeze(-1).expand(-1, -1, C_CLIP)
            ).unsqueeze(0).repeat(K, 1, 1, 1)
            x_clip_full = x_clip_vis + clip_pos_emd_vis  # [K, B, N, C_d_clip]

            x_clip = []
            for idx, clip_decoder in enumerate(self.clip_decoder):
                x_clip.append(clip_decoder(x_clip_full[idx]))
            x_clip = torch.stack(x_clip)  # align and normalize
        else:
            x_clip = None

        if self.add_pool_norm:
            x_vis_cls, x_vis = x_vis[:, :1], x_vis[:, 1:]
            if self.pool_type != "cls" and x_vis.shape[1] == 0:
                raise ValueError(
                    "mask must keep at least one patch token visible when using "
                    f"pool_type='{self.pool_type}'."
                )
            if self.pool_type == "cls":  # only return cls token
                x_pool_vis = self.pool_norm(x_vis_cls)
            else:
                if keep_temporal:
                    B, _, C_CLIP = x_vis.shape
                    if mask is None:
                        temporal_avg = x_vis.view(B, temporal_tokens, -1, C_CLIP).mean(2)
                    else:
                        full_token_count = 1 + temporal_tokens * self.patch_embed.num_patches
                        _, visible_positions = self._visible_token_positions(
                            mask, B, full_token_count, x.device
                        )
                        assert visible_positions is not None
                        temporal_avg = self._masked_temporal_average(
                            x_vis,
                            visible_positions,
                            temporal_tokens,
                        )
                    if self.pool_type == "cls+avg":
                        x_pool_vis = self.pool_norm(x_vis_cls + temporal_avg)
                    elif self.pool_type == "cls_cat_avg":
                        x_pool_vis = self.pool_norm(
                            torch.cat([x_vis_cls, temporal_avg], dim=1)
                        )
                    elif self.pool_type == "avg":
                        x_pool_vis = self.pool_norm(temporal_avg)
                    else:
                        raise ValueError(f"Unsupported pool_type: {self.pool_type}")
                else:
                    if self.pool_type == "cls+avg":
                        x_pool_vis = self.pool_norm(
                            x_vis_cls + x_vis.mean(1, keepdim=True)
                        )
                    elif self.pool_type == "cls_cat_avg":
                        x_pool_vis = self.pool_norm(
                            torch.cat([x_vis_cls, x_vis.mean(1, keepdim=True)], dim=1)
                        )
                    elif self.pool_type == "avg":
                        x_pool_vis = self.pool_norm(x_vis.mean(1, keepdim=True))
                    else:
                        raise ValueError(f"Unsupported pool_type: {self.pool_type}")

            if ssm_state is None:
                return x_vis, x_pool_vis, x_clip
            return x_vis, x_pool_vis, x_clip, ssm_state
        else:
            if ssm_state is None:
                return x_vis, x_clip
            return x_vis, x_clip, ssm_state


def load_state_dict(pretrained_path, model, ckpt_num_frame, num_frames):
    logger.info(f"Loading pretrained weights from {pretrained_path}")
    try:
        checkpoint_model = torch.load(
            pretrained_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint_model = torch.load(pretrained_path, map_location="cpu")
    if not isinstance(checkpoint_model, dict):
        raise TypeError("Expected a plain state_dict (dict) checkpoint.")
    if "model" in checkpoint_model or "module" in checkpoint_model:
        raise ValueError(
            "Checkpoint wrapper keys ('model'/'module') are not supported. "
            "Pass a plain state_dict checkpoint."
        )

    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
    num_patches = model.patch_embed.num_patches  #
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)

    if orig_size != new_size:
        logger.info(
            "Position interpolate from %dx%d to %dx%d"
            % (orig_size, orig_size, new_size, new_size)
        )
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # B, L, C -> B, H, W, C -> B, C, H, W
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        # B, C, H, W -> B, H, W, C ->  B, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
            -1, new_size, new_size, embedding_size
        )
        pos_tokens = pos_tokens.flatten(1, 2)  # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

    # we use 8 frames for pretraining
    temporal_pos_embed = checkpoint_model["temporal_pos_embedding"]
    if ckpt_num_frame is None or ckpt_num_frame <= 0:
        raise ValueError(
            "ckpt_num_frame must be a positive integer when loading pretrained weights."
        )
    orig_t_size = ckpt_num_frame // model.patch_embed.tubelet_size
    new_t_size = num_frames // model.patch_embed.tubelet_size
    # height (== width) for the checkpoint position embedding
    if orig_t_size != new_t_size:
        logger.info(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        temporal_pos_embed = torch.nn.functional.interpolate(
            temporal_pos_embed, size=(new_t_size,), mode="linear", align_corners=False
        )
        temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
        checkpoint_model["temporal_pos_embedding"] = temporal_pos_embed

    msg = model.load_state_dict(checkpoint_model, strict=True)
    logger.info(msg)


def build_videomamba(config, add_pool_norm=True):
    vision_cfg = config.vision_encoder
    channels = vision_cfg.channels
    img_size = vision_cfg.img_size
    patch_size = vision_cfg.patch_size
    depth = vision_cfg.depth
    embed_dim = vision_cfg.embed_dim
    drop_path_rate = vision_cfg.drop_path_rate
    ssm_cfg = vision_cfg.ssm_cfg
    norm_epsilon = vision_cfg.norm_epsilon
    fused_add_norm = vision_cfg.fused_add_norm
    rms_norm = vision_cfg.rms_norm
    residual_in_fp32 = vision_cfg.residual_in_fp32
    bimamba = vision_cfg.bimamba
    pool_type = vision_cfg.pool_type
    kernel_size = vision_cfg.kernel_size
    num_frames = vision_cfg.num_frames
    use_checkpoint = vision_cfg.use_checkpoint
    checkpoint_num = vision_cfg.checkpoint_num
    clip_output_dim = vision_cfg.clip_output_dim
    clip_return_layer = vision_cfg.clip_return_layer
    clip_student_return_interval = vision_cfg.clip_student_return_interval
    model = PretrainVideoMamba(
        img_size=img_size,
        patch_size=patch_size,
        depth=depth,
        embed_dim=embed_dim,
        channels=channels,
        drop_path_rate=drop_path_rate,
        ssm_cfg=ssm_cfg,
        norm_epsilon=norm_epsilon,
        fused_add_norm=fused_add_norm,
        rms_norm=rms_norm,
        residual_in_fp32=residual_in_fp32,
        bimamba=bimamba,
        pool_type=pool_type,
        kernel_size=kernel_size,
        num_frames=num_frames,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        clip_decoder_embed_dim=vision_cfg.clip_decoder_embed_dim,
        clip_output_dim=clip_output_dim,
        clip_return_layer=clip_return_layer,
        clip_student_return_interval=clip_student_return_interval,
        add_pool_norm=add_pool_norm,
    )
    object.__setattr__(model, "default_cfg", _cfg())
    pretrained_path = vision_cfg.pretrained
    if pretrained_path is not None:
        load_state_dict(
            pretrained_path=pretrained_path,
            model=model,
            ckpt_num_frame=vision_cfg.ckpt_num_frame,
            num_frames=num_frames,
        )
    else:
        logger.info("No pretrained weights!!!")
    return model


if __name__ == "__main__":

    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8

    config = {
        "vision_encoder": {
            "img_size": 224,
            "patch_size": 16,
            "depth": 32,
            "embed_dim": 576,
            "channels": 3,
            "drop_path_rate": 0.0,
            "ssm_cfg": None,
            "norm_epsilon": 1e-5,
            "fused_add_norm": True,
            "rms_norm": True,
            "residual_in_fp32": True,
            "bimamba": True,
            "pool_type": "cls",
            "kernel_size": 1,
            "num_frames": 8,
            "use_checkpoint": False,
            "checkpoint_num": 0,
            "clip_decoder_embed_dim": 576,
            "clip_output_dim": 512,
            "clip_return_layer": 1,
            "clip_student_return_interval": 1,
            "pretrained": "your_model_path/videomamba_m16_k400_mask_pt_f8_res224.pth",
        }
    }
    from utils.easydict import EasyDict

    channels = config["vision_encoder"].get("channels", 3)
    model = build_videomamba(EasyDict(config)).cuda()

    # flops = FlopCountAnalysis(model, torch.rand(1, channels, num_frames, 224, 224))
    # s = time.time()
    # logger.info(flop_count_table(flops, max_depth=1))
    # logger.info(time.time()-s)
    mask_token = num_frames * int(14 * 14 * 0.8)
    mask = (
        torch.cat(
            [
                torch.zeros(1, num_frames * 14 * 14 + 1 - mask_token),
                torch.ones(1, mask_token),
            ],
            dim=-1,
        )
        .to(torch.bool)
        .cuda()
    )
    logger.info(
        model(torch.rand(1, channels, num_frames, 224, 224).cuda(), mask)[1].shape
    )
