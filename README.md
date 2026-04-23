# Inverse Filtering for Motion Blur Removal

**CENG 503 — Homework 2**

This project simulates motion blur on a grayscale image and restores it using two classical frequency-domain methods implemented with **OpenCV DFT**.

---

## Overview

The degradation model is:

```
g(x,y) = h(x,y) * f(x,y) + n(x,y)
```

| Symbol | Meaning |
|--------|---------|
| `f` | Original clean image |
| `h` | PSF (motion blur kernel) |
| `g` | Observed (blurred + noisy) image |
| `n` | Additive Gaussian noise |

Both restoration methods solve for `f` given `g` and `h` in the Fourier domain.

---

## Methods

### 1. Inverse Filter (Regularised)

$$\hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + \varepsilon^2} \cdot G(u,v)$$

A small ε² prevents division by zero at frequencies where the PSF has near-zero response. Without regularisation this method amplifies noise catastrophically.

### 2. Wiener Filter

$$\hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K} \cdot G(u,v)$$

`K` is the noise-to-signal power ratio. A larger `K` suppresses more noise at the cost of less sharpening. This is the minimum mean-square-error (MMSE) estimator under stationary signal/noise assumptions.

---

## Project Structure

```
inverse_filtering/
├── main.py        # Entry point — run this
├── blur.py        # PSF creation, motion blur, Gaussian noise
├── restore.py     # Inverse filter & Wiener filter (OpenCV DFT)
├── utils.py       # Image I/O, PSNR/SSIM metrics, figure saving
└── results/       # Output figures (generated on run)
```

---

## Results

Test image: **cameraman** (512×512, scikit-image) with 20-pixel horizontal motion blur and σ = 0.01 Gaussian noise.

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Blurred + Noise | 22.03 | 0.626 |
| Inverse Filter | 7.32 | 0.026 |
| **Wiener Filter** | **23.61** | **0.399** |

The inverse filter collapses under even mild noise, demonstrating why regularisation (Wiener) is necessary in practice.

### Output Figures

| File | Description |
|------|-------------|
| `full_comparison.png` | 5-panel report with embedded PSNR/SSIM |
| `restoration_comparison.png` | Degraded input vs. both restored outputs |
| `blur_stages.png` | Original → Blurred → Blurred+Noise |
| `psf_kernel.png` | PSF kernel heatmap |
| `spectrum_*.png` | Log-magnitude Fourier spectra |

---

## Installation

```bash
pip install opencv-python numpy scikit-image scipy matplotlib
```

Python 3.9+ recommended.

---

## Usage

```bash
python3 main.py
```

All tunable parameters are at the top of `main()` in [main.py](main.py):

```python
PSF_LENGTH  = 20      # pixels of motion (try 10–40)
PSF_ANGLE   = 0       # degrees; 0=horizontal, 45=diagonal
NOISE_SIGMA = 0.01    # Gaussian noise std-dev

INV_EPSILON = 1e-3    # inverse filter regularisation
WIENER_K    = 0.005   # Wiener NSR constant
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | DFT-based convolution, filtering, and image I/O |
| `numpy` | Array operations |
| `scikit-image` | Test image, PSNR/SSIM metrics |
| `scipy` | PSF kernel rotation |
| `matplotlib` | Result visualisation |
