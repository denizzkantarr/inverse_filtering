"""
utils.py — Helper functions: image I/O, metrics, and result visualisation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ── image loading ──────────────────────────────────────────────────────────────

def load_test_image() -> np.ndarray:
    """
    Load the classic 512×512 'cameraman' grayscale test image from scikit-image.

    Returns:
        image : 2-D float64 array with values in [0, 1].
    """
    return img_as_float(data.camera())


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Linearly rescale any image array to the [0, 1] range.

    Useful when a restored image has values slightly outside [0, 1]
    due to ringing artefacts before final clipping.

    Args:
        image : numpy array of any shape and range.
    Returns:
        Normalised array in [0, 1] with the same shape and dtype.
    """
    lo, hi = image.min(), image.max()
    if hi == lo:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)


# ── quality metrics ────────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """
    Compute standard full-reference image quality metrics.

    PSNR  (Peak Signal-to-Noise Ratio) — higher is better; measures
          pixel-level fidelity in dB.
    SSIM  (Structural Similarity Index) — higher is better (max 1.0);
          measures perceptual similarity in luminance, contrast, structure.

    Args:
        original  : 2-D ground-truth image in [0, 1].
        processed : 2-D restored/degraded image in [0, 1].
    Returns:
        dict with keys 'PSNR' (float, dB) and 'SSIM' (float, 0–1).
    """
    return {
        'PSNR': psnr(original, processed, data_range=1.0),
        'SSIM': ssim(original, processed, data_range=1.0),
    }


def print_metrics_table(blurred_m: dict, inv_m: dict, wiener_m: dict) -> None:
    """
    Print a formatted table comparing quality metrics across methods.

    Args:
        blurred_m : metrics for the degraded (blurred+noisy) image.
        inv_m     : metrics for the inverse-filter output.
        wiener_m  : metrics for the Wiener-filter output.
    """
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {'Method':<22}  {'PSNR (dB)':>10}  {'SSIM':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}")
    print(f"  {'Blurred + Noise':<22}  {blurred_m['PSNR']:>10.2f}  {blurred_m['SSIM']:>8.4f}")
    print(f"  {'Inverse Filter':<22}  {inv_m['PSNR']:>10.2f}  {inv_m['SSIM']:>8.4f}")
    print(f"  {'Wiener Filter':<22}  {wiener_m['PSNR']:>10.2f}  {wiener_m['SSIM']:>8.4f}")
    print(f"{sep}\n")


# ── figure saving ──────────────────────────────────────────────────────────────

def save_comparison(images: list,
                    titles: list,
                    filename: str,
                    suptitle: str = "") -> None:
    """
    Save a row of grayscale images as a single comparison figure.

    Args:
        images   : list of 2-D arrays in [0, 1].
        titles   : list of subplot title strings (same length as images).
        filename : output file path (PNG recommended).
        suptitle : optional figure-level title.
    """
    n   = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {filename}")


def save_psf_figure(psf: np.ndarray, filename: str) -> None:
    """
    Save a visualisation of the PSF kernel with a colour-bar.

    Args:
        psf      : 2-D PSF array.
        filename : output file path.
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(psf, cmap="hot", interpolation="nearest")
    ax.set_title("Motion Blur PSF Kernel", fontsize=12)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {filename}")


def save_full_report(original: np.ndarray,
                     blurred: np.ndarray,
                     noisy_blurred: np.ndarray,
                     inv_restored: np.ndarray,
                     wiener_restored: np.ndarray,
                     filename: str) -> tuple:
    """
    Save a 5-panel summary figure with PSNR/SSIM annotations and return metrics.

    Panels (left → right):
      1. Original clean image
      2. Motion-blurred (no noise)
      3. Blurred + noise  — the actual input to the restoration algorithms
      4. Inverse-filter output
      5. Wiener-filter output

    Args:
        original        : ground-truth clean image.
        blurred         : image after motion blur only.
        noisy_blurred   : blurred image after adding noise.
        inv_restored    : output of inverse_filter().
        wiener_restored : output of wiener_filter().
        filename        : output file path.

    Returns:
        (inv_metrics, wiener_metrics, blurred_metrics) — each a dict with
        keys 'PSNR' and 'SSIM'.
    """
    blurred_m = compute_metrics(original, noisy_blurred)
    inv_m     = compute_metrics(original, inv_restored)
    wiener_m  = compute_metrics(original, wiener_restored)

    images = [original, blurred, noisy_blurred, inv_restored, wiener_restored]
    titles = [
        "Original",
        "Motion Blurred\n(noise-free)",
        f"Blurred + Noise\nPSNR {blurred_m['PSNR']:.1f} dB",
        f"Inverse Filter\nPSNR {inv_m['PSNR']:.1f} dB  SSIM {inv_m['SSIM']:.3f}",
        f"Wiener Filter\nPSNR {wiener_m['PSNR']:.1f} dB  SSIM {wiener_m['SSIM']:.3f}",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.suptitle(
        "Motion Blur Removal — Inverse Filter vs. Wiener Filter",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {filename}")

    return inv_m, wiener_m, blurred_m


def save_frequency_magnitude(image: np.ndarray, filename: str, title: str = "") -> None:
    """
    Save a log-scale magnitude spectrum plot of an image.

    Visualising the spectrum helps understand which frequencies the
    blur or restoration has affected.

    Args:
        image    : 2-D float64 image.
        filename : output file path.
        title    : optional axes title.
    """
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(image)))
    log_spectrum = np.log1p(spectrum)          # log(1 + |F|) for display

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(log_spectrum, cmap="inferno")
    ax.set_title(title or "Log Magnitude Spectrum", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {filename}")
