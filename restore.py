"""
restore.py — Frequency-domain image restoration using OpenCV DFT.

Both methods work in the Fourier domain where convolution becomes
pointwise multiplication.  The degradation model is:

    G(u,v) = H(u,v) · F(u,v) + N(u,v)

where G is the observed spectrum, H the OTF, F the ideal image spectrum,
and N the noise spectrum.  The goal is to estimate F from G and H.

──────────────────────────────────────────────────────────────────────────
Method            Formula                         Notes
──────────────────────────────────────────────────────────────────────────
Inverse filter    F̂ = G / H                       Amplifies noise badly
Wiener filter     F̂ = [H* / (|H|² + K)] · G     K trades deblur vs. noise
──────────────────────────────────────────────────────────────────────────

cv2.dft / cv2.idft are used for all DFT operations.  cv2.dft returns a
two-channel float64 array [H, W, 2]; we convert to complex128 numpy arrays
for the filter arithmetic, then pack back to two channels before cv2.idft.
"""

import cv2
import numpy as np


# ── shared helper ──────────────────────────────────────────────────────────────

def _otf_from_psf(psf: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Pad the PSF to the image size and compute its Optical Transfer Function.

    The PSF is zero-padded then circularly shifted so its centre aligns
    with pixel (0, 0), which is the FFT convention for linear (non-circular)
    convolution.

    Args:
        psf         : 2-D PSF array.
        image_shape : (H, W) of the target image.
    Returns:
        OTF as a complex128 numpy array with shape image_shape.
    """
    psf_padded = np.zeros(image_shape, dtype=np.float64)
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf

    # Roll PSF centre to index (0, 0) for correct FFT-phase alignment
    psf_padded = np.roll(psf_padded, -(ph // 2), axis=0)
    psf_padded = np.roll(psf_padded, -(pw // 2), axis=1)

    # cv2.dft: float64 input → [H, W, 2] output (real, imag channels)
    dft_result = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    return dft_result[..., 0] + 1j * dft_result[..., 1]


def _idft_real(complex_spectrum: np.ndarray) -> np.ndarray:
    """
    Inverse DFT of a complex spectrum, returning only the real part.

    Args:
        complex_spectrum : 2-D complex128 array (frequency domain).
    Returns:
        Spatial-domain result as a 2-D float64 array.
    """
    # Pack complex array to the [H, W, 2] two-channel format cv2.idft expects
    two_ch = np.stack([complex_spectrum.real, complex_spectrum.imag], axis=-1)
    # DFT_SCALE: divide by N*M; DFT_REAL_OUTPUT: drop the (zero) imaginary part
    return cv2.idft(two_ch, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)


# ── inverse filter ─────────────────────────────────────────────────────────────

def inverse_filter(blurred: np.ndarray,
                   psf: np.ndarray,
                   epsilon: float = 1e-3) -> np.ndarray:
    """
    Restore an image via the regularised inverse filter.

    The ideal inverse filter is simply F̂ = G / H, but it amplifies noise
    wherever |H| is small (i.e., frequencies the blur suppressed heavily).
    Regularisation replaces the bare division with:

        F̂(u,v) = G(u,v) · H*(u,v) / (|H(u,v)|² + ε²)

    which is mathematically equivalent to Wiener filtering with K = ε²,
    but the intent here is only to prevent numerical singularities
    (ε² ≪ 1), not to suppress noise.  Wiener filtering (below) uses a
    larger K tuned to the actual noise level.

    Args:
        blurred : 2-D float64 array in [0, 1], degraded image.
        psf     : 2-D PSF kernel used to create the blur.
        epsilon : Regularisation constant; set just large enough to prevent
                  division by zero  (e.g. 1e-3 for mild noise).
    Returns:
        restored : 2-D float64 array clipped to [0, 1].
    """
    # Forward DFT of the blurred image
    G_2ch = cv2.dft(blurred.astype(np.float64), flags=cv2.DFT_COMPLEX_OUTPUT)
    G = G_2ch[..., 0] + 1j * G_2ch[..., 1]

    H = _otf_from_psf(psf, blurred.shape)

    H_conj  = np.conj(H)
    H_power = np.abs(H) ** 2        # |H|², real-valued
    eps_sq  = epsilon ** 2

    # Regularised inverse: avoids amplifying frequencies where |H| ≈ 0
    F_hat = G * H_conj / (H_power + eps_sq)

    restored = _idft_real(F_hat)
    return np.clip(restored, 0.0, 1.0)


# ── Wiener filter ──────────────────────────────────────────────────────────────

def wiener_filter(blurred: np.ndarray,
                  psf: np.ndarray,
                  K: float = 0.01) -> np.ndarray:
    """
    Restore an image with the Wiener (minimum mean-square-error) filter.

    The Wiener filter minimises E[|F̂ - F|²] under the assumption of
    stationary signal and noise.  When the signal and noise power spectra
    are unknown, the simplified parametric form is used:

        W(u,v) = H*(u,v) / (|H(u,v)|² + K)

        F̂(u,v) = W(u,v) · G(u,v)

    K ≈ σ_noise² / σ_signal² (noise-to-signal power ratio).
    ─────────────────────────────────────────────────────────────────────
    Choosing K:
      K → 0   : approaches the pure inverse filter (aggressive deblur,
                  very sensitive to noise).
      K large : heavily damps all frequencies, producing a blurry but
                  smooth result.
      Typical range: 1e-4 … 1e-1 depending on noise level.
    ─────────────────────────────────────────────────────────────────────

    Args:
        blurred : 2-D float64 array in [0, 1], degraded image.
        psf     : 2-D PSF kernel used to create the blur.
        K       : Noise-to-signal ratio (regularisation strength).
    Returns:
        restored : 2-D float64 array clipped to [0, 1].
    """
    # Forward DFT of the blurred image
    G_2ch = cv2.dft(blurred.astype(np.float64), flags=cv2.DFT_COMPLEX_OUTPUT)
    G = G_2ch[..., 0] + 1j * G_2ch[..., 1]

    H = _otf_from_psf(psf, blurred.shape)

    H_conj  = np.conj(H)
    H_power = np.abs(H) ** 2        # |H|², real-valued

    # Wiener kernel: H* / (|H|² + K)
    W     = H_conj / (H_power + K)
    F_hat = G * W

    restored = _idft_real(F_hat)
    return np.clip(restored, 0.0, 1.0)
