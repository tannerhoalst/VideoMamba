import argparse
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class DeterminismConfig:
    seed: int = 0
    deterministic: bool = False
    warn_only: bool = True
    cudnn_benchmark: bool = True
    allow_tf32: bool = True


def configure_determinism(
    seed: int,
    deterministic: bool,
    warn_only: bool = True,
    cudnn_benchmark: Optional[bool] = None,
    allow_tf32: Optional[bool] = None,
) -> DeterminismConfig:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cudnn_benchmark is None:
        cudnn_benchmark = not deterministic
    if allow_tf32 is None:
        allow_tf32 = not deterministic

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=warn_only)

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    return DeterminismConfig(
        seed=seed,
        deterministic=bool(deterministic),
        warn_only=bool(warn_only),
        cudnn_benchmark=bool(cudnn_benchmark),
        allow_tf32=bool(allow_tf32),
    )


def add_determinism_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch algorithms and cuDNN mode.",
    )
    parser.add_argument(
        "--deterministic-warn-only",
        action="store_true",
        help="Use warn-only mode for deterministic algorithm enforcement.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        choices=["on", "off", "auto"],
        default="auto",
        help="cuDNN benchmark mode. auto => inverse of --deterministic.",
    )
    parser.add_argument(
        "--allow-tf32",
        choices=["on", "off", "auto"],
        default="auto",
        help="TF32 matmul/convolution mode. auto => inverse of --deterministic.",
    )
    return parser


def _tri_state_to_bool(value: str) -> Optional[bool]:
    if value == "on":
        return True
    if value == "off":
        return False
    return None


def configure_determinism_from_args(args: argparse.Namespace) -> DeterminismConfig:
    return configure_determinism(
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        warn_only=bool(args.deterministic_warn_only),
        cudnn_benchmark=_tri_state_to_bool(args.cudnn_benchmark),
        allow_tf32=_tri_state_to_bool(args.allow_tf32),
    )
