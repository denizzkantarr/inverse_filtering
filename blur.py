"""
blur.py — Motion blur simulation using a PSF (Point Spread Function).

The degradation model is:  g = h * f + n
  g : observed (blurred+noisy) image
  h : PSF kernel (describes the blur)
  f : original clean image
  n : additive Gaussian noise
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate


def create_motion_blur_psf(length: int = 15, angle: float = 0, size: int = None) -> np.ndarray:
    """
    Build a motion-blur PSF kernel for linear (camera/object) motion.

    A horizontal line of uniform weights represents motion along the x-axis.
    Rotating it to `angle` degrees gives arbitrary-direction blur.

    Args:
        length : Number of pixels the motion spans.
        angle  : Direction of motion in degrees (0 = horizontal, CCW positive).
        size   : Side length of the square output array.
                 Defaults to the next odd number >= length so the kernel fits.
    Returns:
        psf : 2-D float64 array with values summing to 1.
    """
    if size is None:
        size = length if length % 2 == 1 else length + 1

    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    half   = length // 2

    # Horizontal motion: a row of ones centred in the kernel
    kernel[center, center - half : center + half + 1] = 1.0

    # Rotate to the requested angle (bilinear interpolation, no reshape)
    if angle != 0:
        kernel = rotate(kernel, angle, reshape=False, order=1)

    # Normalise so convolution preserves image brightness
    kernel /= kernel.sum()
    return kernel


def _psf_to_otf(psf: np.ndarray, image_shape: tuple) -> tf.Tensor:
    """
    Pad the PSF to the image dimensions and compute its Optical Transfer
    Function (OTF) via FFT.

    The PSF is placed at the top-left corner, then circularly shifted so
    that its centre sits at position (0, 0), which is the FFT convention
    for zero-phase linear convolution.

    Args:
        psf         : 2-D PSF kernel.
        image_shape : (H, W) of the target image.
    Returns:
        OTF as a complex128 TensorFlow tensor of shape image_shape.
    """
    psf_padded = np.zeros(image_shape, dtype=np.float64)
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf

    # Circular shift: move PSF centre to (0, 0) for correct FFT phase
    psf_padded = np.roll(psf_padded, -(ph // 2), axis=0)
    psf_padded = np.roll(psf_padded, -(pw // 2), axis=1)

    return tf.signal.fft2d(tf.cast(psf_padded, tf.complex128))


def apply_motion_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Blur a grayscale image by convolving it with a PSF in the frequency domain.

    Frequency-domain multiplication is equivalent to spatial-domain circular
    convolution and is computationally efficient for large kernels.

    Args:
        image : 2-D float64 array in [0, 1].
        psf   : 2-D PSF kernel (output of create_motion_blur_psf).
    Returns:
        blurred : 2-D float64 array in [0, 1].
    """
    G  = tf.signal.fft2d(tf.cast(image, tf.complex128))   # Image spectrum
    H  = _psf_to_otf(psf, image.shape)                    # PSF spectrum (OTF)

    # Convolution theorem: multiply spectra, then invert
    blurred = tf.math.real(tf.signal.ifft2d(G * H)).numpy()
    return np.clip(blurred, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    Corrupt an image with zero-mean Gaussian (AWGN) noise.

    Additive white Gaussian noise is the standard model for sensor noise and
    quantisation errors. A small sigma (< 0.02) is typical for "light" noise.

    Args:
        image : 2-D float64 array in [0, 1].
        sigma : Standard deviation of the noise; controls noise strength.
    Returns:
        noisy : 2-D float64 array clipped to [0, 1].
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)
