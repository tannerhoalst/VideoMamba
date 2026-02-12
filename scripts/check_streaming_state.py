import argparse

import torch

from video_mamba import (
    STREAMING_CONTRACT_VERSION,
    add_determinism_args,
    configure_determinism_from_args,
)
from video_mamba.mamba_simple import Mamba


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate VideoMamba streaming state path.")
    add_determinism_args(parser)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=12)
    parser.add_argument("--split", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=16)
    return parser


def main():
    args = _build_arg_parser().parse_args()
    configure_determinism_from_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "check_streaming_state.py requires CUDA because VideoMamba kernels are CUDA-only."
        )
    device = torch.device("cuda")

    model = Mamba(
        d_model=args.d_model,
        d_state=8,
        d_conv=4,
        expand=2,
        use_fast_path=False,
    ).to(device)

    batch_size = args.batch_size
    seqlen = args.seqlen
    split = args.split
    if split <= 0 or split >= seqlen:
        raise ValueError("--split must be in range [1, seqlen-1].")

    x = torch.randn(batch_size, seqlen, args.d_model, device=device, requires_grad=True)

    out_full = model(x)

    x1, x2 = x[:, :split], x[:, split:]
    out1, state = model(x1, return_state=True)
    out2, _ = model(x2, state=state, return_state=True)
    out_chunked = torch.cat([out1, out2], dim=1)

    torch.testing.assert_close(out_full, out_chunked, rtol=1e-4, atol=1e-4)

    out_chunked.sum().backward()
    if x.grad is None:
        raise RuntimeError("Missing gradients for streaming path.")

    print(f"Streaming state check passed. contract={STREAMING_CONTRACT_VERSION}")


if __name__ == "__main__":
    main()
