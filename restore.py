"""
restore.py — Frequency-domain image restoration using TensorFlow FFT.

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
"""

import numpy as np
import tensorflow as tf


# ── shared helper ──────────────────────────────────────────────────────────────

def _otf_from_psf(psf: np.ndarray, image_shape: tuple) -> tf.Tensor:
    """
    Pad the PSF to the image size and compute its Optical Transfer Function.

    The PSF is zero-padded then circularly shifted so its centre aligns
    with pixel (0, 0), which is the FFT convention for linear (non-circular)
    convolution.

    Args:
        psf         : 2-D PSF array.
        image_shape : (H, W) of the target image.
    Returns:
        OTF as a complex128 tensor with shape image_shape.
    """
    psf_padded = np.zeros(image_shape, dtype=np.float64)
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf

    # Roll PSF centre to index (0,0) for correct FFT-phase alignment
    psf_padded = np.roll(psf_padded, -(ph // 2), axis=0)
    psf_padded = np.roll(psf_padded, -(pw // 2), axis=1)

    return tf.signal.fft2d(tf.cast(psf_padded, tf.complex128))


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
    G = tf.signal.fft2d(tf.cast(blurred, tf.complex128))
    H = _otf_from_psf(psf, blurred.shape)

    H_conj  = tf.math.conj(H)
    # |H|² as a real tensor, then cast to complex for the division
    H_power = tf.cast(tf.abs(H) ** 2, tf.complex128)
    eps_sq  = tf.cast(epsilon ** 2, tf.complex128)

    F_hat = G * H_conj / (H_power + eps_sq)

    restored = tf.math.real(tf.signal.ifft2d(F_hat)).numpy()
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
    G = tf.signal.fft2d(tf.cast(blurred, tf.complex128))
    H = _otf_from_psf(psf, blurred.shape)

    H_conj  = tf.math.conj(H)
    H_power = tf.cast(tf.abs(H) ** 2, tf.complex128)
    K_c     = tf.cast(K, tf.complex128)

    # Wiener kernel: H* / (|H|² + K)
    W     = H_conj / (H_power + K_c)
    F_hat = G * W

    restored = tf.math.real(tf.signal.ifft2d(F_hat)).numpy()
    return np.clip(restored, 0.0, 1.0)
