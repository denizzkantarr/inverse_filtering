"""
main.py — Entry point for Homework 2: Inverse Filtering for Motion Blur Removal.

Pipeline
────────
  1. Load the standard 512×512 'cameraman' grayscale test image.
  2. Simulate horizontal motion blur with a PSF of configurable length/angle.
  3. Add Gaussian noise to the blurred image (realistic degradation model).
  4. Restore with the regularised inverse filter  (OpenCV DFT).
  5. Restore with the Wiener filter               (OpenCV DFT).
  6. Save comparison figures and print quality metrics (PSNR, SSIM).

All parameters are grouped at the top of main() for easy experimentation.

Dependencies: opencv-python, numpy, scikit-image, scipy, matplotlib
"""

import os
import numpy as np

from blur    import create_motion_blur_psf, apply_motion_blur, add_gaussian_noise
from restore import inverse_filter, wiener_filter
from utils   import (
    load_test_image,
    save_comparison,
    save_full_report,
    save_psf_figure,
    save_frequency_magnitude,
    print_metrics_table,
)


def main() -> None:
    # ── Hyper-parameters ────────────────────────────────────────────────────────
    PSF_LENGTH  = 20      # pixels of motion blur (try 10–40)
    PSF_ANGLE   = 0       # degrees; 0 = horizontal, 45 = diagonal
    NOISE_SIGMA = 0.01    # Gaussian noise std-dev (0 = noise-free)

    INV_EPSILON = 1e-3    # inverse-filter regularisation (ε²); smaller = more
                          #   aggressive deblur but noisier output
    WIENER_K    = 0.005   # Wiener NSR constant K; larger = smoother but blurrier
    # ────────────────────────────────────────────────────────────────────────────

    print("=" * 58)
    print("  Motion Blur Removal — Inverse Filter & Wiener Filter")
    print("=" * 58)

    # ── 1. Setup ────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    np.random.seed(42)          # reproducible noise

    # ── 2. Load image ───────────────────────────────────────────────────────────
    print("\n[1] Loading test image …")
    image = load_test_image()
    print(f"    Shape: {image.shape}  dtype: {image.dtype}")

    # ── 3. Build PSF and blur the image ─────────────────────────────────────────
    print("\n[2] Simulating motion blur …")
    psf = create_motion_blur_psf(length=PSF_LENGTH, angle=PSF_ANGLE)
    print(f"    PSF: {PSF_LENGTH}px  angle={PSF_ANGLE}°  kernel={psf.shape}")

    blurred       = apply_motion_blur(image, psf)
    noisy_blurred = add_gaussian_noise(blurred, sigma=NOISE_SIGMA)
    print(f"    Noise σ = {NOISE_SIGMA}")

    # ── 4. Inverse filtering ────────────────────────────────────────────────────
    print("\n[3] Inverse filtering …")
    inv_restored = inverse_filter(noisy_blurred, psf, epsilon=INV_EPSILON)
    print(f"    ε = {INV_EPSILON}  →  regularisation = ε² = {INV_EPSILON**2:.2e}")

    # ── 5. Wiener filtering ─────────────────────────────────────────────────────
    print("\n[4] Wiener filtering …")
    wiener_restored = wiener_filter(noisy_blurred, psf, K=WIENER_K)
    print(f"    K = {WIENER_K}")

    # ── 6. Save figures ─────────────────────────────────────────────────────────
    print("\n[5] Saving output figures …")

    # PSF visualisation
    save_psf_figure(psf, "results/psf_kernel.png")

    # Blur pipeline: original → blurred → noisy
    save_comparison(
        [image, blurred, noisy_blurred],
        ["Original", "Motion Blurred", f"Blurred + Noise (σ={NOISE_SIGMA})"],
        filename="results/blur_stages.png",
        suptitle="Degradation Pipeline",
    )

    # Restoration comparison
    save_comparison(
        [noisy_blurred, inv_restored, wiener_restored],
        [
            "Degraded Input",
            f"Inverse Filter (ε={INV_EPSILON})",
            f"Wiener Filter (K={WIENER_K})",
        ],
        filename="results/restoration_comparison.png",
        suptitle="Restoration Methods",
    )

    # Full 5-panel summary with embedded metrics
    inv_m, wiener_m, blurred_m = save_full_report(
        image, blurred, noisy_blurred, inv_restored, wiener_restored,
        filename="results/full_comparison.png",
    )

    # Frequency-domain spectra (computed via cv2.dft inside save_frequency_magnitude)
    save_frequency_magnitude(image,           "results/spectrum_original.png",  "Spectrum — Original")
    save_frequency_magnitude(noisy_blurred,   "results/spectrum_blurred.png",   "Spectrum — Blurred + Noise")
    save_frequency_magnitude(wiener_restored, "results/spectrum_wiener.png",    "Spectrum — Wiener Restored")

    # ── 7. Print metrics table ──────────────────────────────────────────────────
    print("\n[6] Image Quality Metrics (vs. original):")
    print_metrics_table(blurred_m, inv_m, wiener_m)

    print("All results saved to  results/")


if __name__ == "__main__":
    main()
