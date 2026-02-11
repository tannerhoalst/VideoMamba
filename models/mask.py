import numpy as np
import torch


def _resolve_device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _add_cls(mask: torch.Tensor) -> torch.Tensor:
    cls = torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
    return torch.cat([cls, mask], dim=1)


def TubeMaskingGenerator(
    input_size,
    mask_ratio,
    batch,
    device=None,
):
    frames, height, width = input_size
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1].")
    num_patches_per_frame = height * width
    num_masks_per_frame = int(mask_ratio * num_patches_per_frame)

    mask_list = []
    for _ in range(batch):
        mask_per_frame = np.hstack(
            [
                np.zeros(num_patches_per_frame - num_masks_per_frame),
                np.ones(num_masks_per_frame),
            ]
        )
        np.random.shuffle(mask_per_frame)
        mask_list.append(np.tile(mask_per_frame, (frames, 1)).flatten())
    device = _resolve_device(device)
    mask = torch.tensor(np.array(mask_list), dtype=torch.bool, device=device)
    return _add_cls(mask)


def RandomMaskingGenerator(
    input_size,
    mask_ratio,
    batch,
    device=None,
):
    frames, height, width = input_size
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1].")

    num_patches = frames * height * width  # 8x14x14
    num_mask = int(mask_ratio * num_patches)

    mask_list = []
    for _ in range(batch):
        mask = np.hstack(
            [
                np.zeros(num_patches - num_mask),
                np.ones(num_mask),
            ]
        )
        np.random.shuffle(mask)
        mask_list.append(mask)
    device = _resolve_device(device)
    mask = torch.tensor(np.array(mask_list), dtype=torch.bool, device=device)
    return _add_cls(mask)
