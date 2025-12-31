import torch

from models.videomamba.mamba_simple import Mamba


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mamba(
        d_model=16,
        d_state=8,
        d_conv=4,
        expand=2,
        use_fast_path=False,
    ).to(device)

    batch_size = 2
    seqlen = 12
    split = 5

    x = torch.randn(batch_size, seqlen, 16, device=device, requires_grad=True)

    out_full = model(x)

    x1, x2 = x[:, :split], x[:, split:]
    out1, state = model(x1, return_state=True)
    out2, _ = model(x2, state=state, return_state=True)
    out_chunked = torch.cat([out1, out2], dim=1)

    torch.testing.assert_close(out_full, out_chunked, rtol=1e-4, atol=1e-4)

    out_chunked.sum().backward()
    if x.grad is None:
        raise RuntimeError("Missing gradients for streaming path.")

    print("Streaming state check passed.")


if __name__ == "__main__":
    main()
