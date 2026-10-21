"""Compare the owned Qwen convolution with PyTorch through two derivatives.

Uses Qwen 3.5's 8,192-channel, kernel-size-four shape on CUDA; no model download.
Times are local measurements of this isolated operator, not training speedups.
"""

import json
import time

import torch

from darkmagic.qwen import depthwise_conv1d


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This validation requires CUDA; CPU tests are in tests/test_qwen.py")
    torch.manual_seed(123)
    torch.backends.cudnn.allow_tf32 = False
    x = torch.randn(1, 8192, 256, device="cuda", requires_grad=True)
    w = torch.randn(8192, 1, 4, device="cuda", requires_grad=True)
    directions = [torch.randn_like(t) for t in (x, w)]

    def evaluate(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = fn(x, w)
        grads = torch.autograd.grad(y.sin().sum(), (x, w), create_graph=True)
        second = torch.autograd.grad(sum((a * b).sum() for a, b in zip(grads, directions)), (x, w))
        torch.cuda.synchronize()
        return y.detach(), tuple(g.detach() for g in grads), second, time.perf_counter() - start

    reference = evaluate(lambda x, w: torch.nn.functional.conv1d(x, w, padding=3, groups=8192))
    actual = evaluate(lambda x, w: depthwise_conv1d(x, w, padding=3))
    errors = []
    for a, b in [
        (actual[0], reference[0]),
        *zip(actual[1], reference[1]),
        *zip(actual[2], reference[2]),
    ]:
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-4)
        errors.append(float((a - b).abs().max()))
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": [1, 8192, 256],
                "max_absolute_errors": errors,
                "reference_seconds": reference[3],
                "shifted_seconds": actual[3],
                "result": "forward, gradient and Hessian-vector product match",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
