from __future__ import annotations

"""Lab 04 (skeleton): Markov Random Field (MRF) image restoration."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

PenaltyType = Literal["quadratic", "huber"]


def mrf_energy(
    x: np.ndarray,
    y: np.ndarray,
    lambda_smooth: float,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> float:
    """
    Compute pairwise MRF energy for grayscale image restoration.

    Energy:
        E(x) = sum_p (x_p - y_p)^2 + lambda * sum_(p,q in N) rho(x_p - x_q)

    Args:
        x: Restored image candidate `(H,W)`.
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Scalar energy.
    """
    # Data term: fidelity to noisy observation
    data_term = np.sum((x-y) ** 2)

    # Smoothness term: pairwise differences on 4-connected neighborhood
    diff_h = x[:,1:] - x[:, :-1]   # horizontal
    diff_v = x[1:, :] - x[:-1, :]  # vertical

    if penalty == "quadratic":
        smooth_term = np.sum(diff_h ** 2) + np.sum(diff_v ** 2)
    elif penalty == "huber":
        def huber_penalty(d, delta):
            abs_d = np.abs(d)
            return np.where(abs_d <= delta,
                            0.5 * d ** 2,
                            delta * (abs_d - 0.5 * delta))

        smooth_term = np.sum(huber_penalty(diff_h, huber_delta)) + \
                      np.sum(huber_penalty(diff_v, huber_delta))
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

    return data_term + lambda_smooth * smooth_term


def mrf_denoise(
    y: np.ndarray,
    lambda_smooth: float,
    num_iters: int,
    step: float = 0.1,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> np.ndarray:
    """
    Restore grayscale image by minimizing MRF energy.

    Args:
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        num_iters: Number of optimization iterations.
        step: Optimization step size.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Restored image with the same shape as `y`.
    """
    x = y.copy().astype(np.float32)

    current_step = step * 0.5 if penalty == "quadratic" else step

    for it in range(num_iters):
        grad_data = 2.0 * (x - y)
        grad_smooth = np.zeros_like(x, dtype=np.float32)

        # Horizontal Smoothness (x_right - x_left)
        diff_h = x[:, 1:] - x[:, :-1]
        grad_h = 2.0 * diff_h if penalty == "quadratic" else np.where(
            np.abs(diff_h) <= huber_delta, diff_h, huber_delta * np.sign(diff_h)
        )
        grad_smooth[:, :-1] -= grad_h
        grad_smooth[:, 1:] += grad_h

        # Vertical Smoothness (x_down - x_up)
        diff_v = x[1:, :] - x[:-1, :]
        grad_v = 2.0 * diff_v if penalty == "quadratic" else np.where(
            np.abs(diff_v) <= huber_delta, diff_v, huber_delta * np.sign(diff_v)
        )
        grad_smooth[:-1, :] -= grad_v
        grad_smooth[1:, :] += grad_v

        grad = grad_data + lambda_smooth * grad_smooth
        x -= current_step * grad

        x = np.clip(x, 0.0, 255.0)

    return x


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0,255] uint8 for visualization."""
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.uint8)
    normalized = (x - x_min) / (x_max - x_min) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def main() -> int:
    """
    Lab 04 demo (skeleton).

    Expected behavior after implementation:
    - load grayscale image from `./imgs/`
    - add Gaussian noise (deterministic seed)
    - denoise with MRF (quadratic and/or huber)
    - save side-by-side result to `./out/lab04/mrf_denoise.png`
    """
    parser = argparse.ArgumentParser(description="Lab 04 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab04", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    try:
        clean = img.astype(np.float32)
        rng = np.random.default_rng(0)
        noisy = clean + rng.normal(0.0, 18.0, size=clean.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)

        den_quad = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="quadratic")
        den_hub = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="huber", huber_delta=8.0)

        e_noisy_q = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_quad = mrf_energy(den_quad, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_noisy_h = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)
        e_hub = mrf_energy(den_hub, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)

        plt.figure(figsize=(12, 4))
        panels = [
            ("Original", clean),
            ("Noisy (seed=0)", noisy),
            (f"MRF quadratic\nE: {e_noisy_q:.1f} -> {e_quad:.1f}", den_quad),
            (f"MRF huber\nE: {e_noisy_h:.1f} -> {e_hub:.1f}", den_hub),
        ]
        for i, (title, im) in enumerate(panels, start=1):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "mrf_denoise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 04 demo is incomplete. Implement the TODO functions in labs/lab04_mrf_restoration.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())