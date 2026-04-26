"""Synthetic galaxy image generation with configurable noise and blur.

This module provides the :class:`GalaxyImageSimulator` class for creating
realistic synthetic galaxy images that can be used to benchmark the robustness
of machine-learning classifiers under various observational degradations.

Typical usage::

    from simulations.data_generation import GalaxyImageSimulator

    sim = GalaxyImageSimulator(image_size=128, random_seed=42)
    images, labels = sim.generate_batch(
        n_samples=256,
        noise_std=0.05,
        psf_sigma=1.5,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Morphology catalogue – extensible via ``GalaxyImageSimulator.MORPHOLOGIES``
# ---------------------------------------------------------------------------
DEFAULT_MORPHOLOGIES: dict[str, int] = {
    "elliptical": 0,
    "spiral": 1,
    "irregular": 2,
}


@dataclass
class GalaxyImageSimulator:
    """Generate synthetic galaxy images with controllable degradations.

    The simulator draws galaxy light profiles from analytic models and
    applies observational effects (Gaussian noise, PSF convolution) so that
    downstream models can be stress-tested against realistic perturbations.

    Parameters
    ----------
    image_size : int
        Side length (in pixels) of the square output images.  Default 128.
    n_channels : int
        Number of spectral channels to simulate.  Default 1 (monochrome).
    random_seed : int | None
        Seed for the NumPy random generator.  ``None`` disables seeding.
    morphologies : dict[str, int] | None
        Mapping from morphology name to integer label.  Falls back to
        :data:`DEFAULT_MORPHOLOGIES` when ``None``.
    """

    image_size: int = 128
    n_channels: int = 1
    random_seed: Optional[int] = None
    morphologies: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MORPHOLOGIES))

    # -- internal state (not part of the public constructor) -----------------
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)
        logger.info(
            "GalaxyImageSimulator initialised – size=%d, channels=%d, seed=%s",
            self.image_size,
            self.n_channels,
            self.random_seed,
        )

    # ------------------------------------------------------------------
    # Light-profile primitives
    # ------------------------------------------------------------------

    def _sersic_profile(self, n: float, r_eff: float) -> NDArray[np.float64]:
        """Return a 2-D Sérsic profile centred on the image grid.

        Parameters
        ----------
        n : float
            Sérsic index (n=1 → exponential disc, n=4 → de Vaucouleurs).
        r_eff : float
            Effective (half-light) radius in pixels.
        """
        centre = self.image_size / 2.0
        y, x = np.mgrid[0 : self.image_size, 0 : self.image_size]
        r = np.sqrt((x - centre) ** 2 + (y - centre) ** 2) + 1e-8

        # Approximation of b_n (Ciotti & Bertin 1999)
        b_n = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
        intensity = np.exp(-b_n * ((r / r_eff) ** (1.0 / n) - 1.0))
        return intensity / intensity.sum()

    def _add_spiral_arms(
        self,
        image: NDArray[np.float64],
        n_arms: int = 2,
        arm_strength: float = 0.3,
    ) -> NDArray[np.float64]:
        """Superimpose logarithmic spiral arms onto an existing profile."""
        centre = self.image_size / 2.0
        y, x = np.mgrid[0 : self.image_size, 0 : self.image_size]
        theta = np.arctan2(y - centre, x - centre)
        r = np.sqrt((x - centre) ** 2 + (y - centre) ** 2) + 1e-8

        pitch = self._rng.uniform(0.2, 0.5)
        arms = np.cos(n_arms * (theta - pitch * np.log(r)))
        spiral_pattern = np.clip(arms, 0, 1) * arm_strength * image
        return image + spiral_pattern

    # ------------------------------------------------------------------
    # Single-image generator
    # ------------------------------------------------------------------

    def generate_galaxy(
        self,
        morphology: str = "elliptical",
    ) -> NDArray[np.float64]:
        """Synthesise a single clean galaxy image.

        Parameters
        ----------
        morphology : str
            One of the keys in :attr:`morphologies`.

        Returns
        -------
        NDArray[np.float64]
            Image array with shape ``(n_channels, image_size, image_size)``.
        """
        if morphology not in self.morphologies:
            raise ValueError(
                f"Unknown morphology '{morphology}'. "
                f"Available: {list(self.morphologies.keys())}"
            )

        r_eff = self._rng.uniform(0.08, 0.25) * self.image_size

        if morphology == "elliptical":
            n_sersic = self._rng.uniform(3.0, 6.0)
            profile = self._sersic_profile(n_sersic, r_eff)
        elif morphology == "spiral":
            n_sersic = self._rng.uniform(0.8, 2.0)
            profile = self._sersic_profile(n_sersic, r_eff)
            n_arms = int(self._rng.choice([2, 3, 4]))
            profile = self._add_spiral_arms(profile, n_arms=n_arms)
        else:  # irregular
            profile = self._sersic_profile(n=1.0, r_eff=r_eff)
            noise_blob = self._rng.normal(0, 0.1, profile.shape)
            profile = np.clip(profile + noise_blob * profile, 0, None)
            profile /= profile.sum() + 1e-12

        # Broadcast to the requested number of channels
        image = np.stack([profile] * self.n_channels, axis=0)
        return image

    # ------------------------------------------------------------------
    # Degradation operators
    # ------------------------------------------------------------------

    @staticmethod
    def apply_gaussian_noise(
        image: NDArray[np.float64],
        std: float,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Add zero-mean Gaussian noise to *image* in-place.

        Parameters
        ----------
        image : NDArray
            Input image (modified and returned).
        std : float
            Standard deviation of the additive noise.
        rng : Generator | None
            Optional random generator; uses the default if ``None``.
        """
        if std <= 0:
            return image
        rng = rng or np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=std, size=image.shape)
        return np.clip(image + noise, 0.0, 1.0)

    @staticmethod
    def apply_psf_blur(
        image: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64]:
        """Convolve each channel with a Gaussian PSF.

        Parameters
        ----------
        image : NDArray
            Image with shape ``(C, H, W)``.
        sigma : float
            Standard deviation of the Gaussian kernel (in pixels).
        """
        if sigma <= 0:
            return image
        blurred = np.empty_like(image)
        for ch in range(image.shape[0]):
            blurred[ch] = gaussian_filter(image[ch], sigma=sigma)
        return blurred

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        n_samples: int,
        noise_std: float = 0.0,
        psf_sigma: float = 0.0,
        morphology_weights: Optional[dict[str, float]] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Generate a labelled batch of (optionally degraded) galaxy images.

        Parameters
        ----------
        n_samples : int
            Number of images to generate.
        noise_std : float
            Gaussian noise standard deviation (0 → no noise).
        psf_sigma : float
            PSF blur sigma in pixels (0 → no blur).
        morphology_weights : dict[str, float] | None
            Sampling weights per morphology.  Uniform when ``None``.

        Returns
        -------
        images : NDArray, shape (n_samples, C, H, W)
            Batch of galaxy images as a 4-D float64 array.
        labels : NDArray, shape (n_samples,)
            Integer class labels.
        """
        morph_names = list(self.morphologies.keys())

        if morphology_weights is not None:
            weights = np.array([morphology_weights.get(m, 0.0) for m in morph_names])
            weights /= weights.sum()
        else:
            weights = np.ones(len(morph_names)) / len(morph_names)

        chosen = self._rng.choice(morph_names, size=n_samples, p=weights)

        images = np.empty(
            (n_samples, self.n_channels, self.image_size, self.image_size),
            dtype=np.float64,
        )
        labels = np.empty(n_samples, dtype=np.int64)

        logger.info(
            "Generating %d galaxy images (noise_std=%.4f, psf_sigma=%.4f) …",
            n_samples,
            noise_std,
            psf_sigma,
        )

        for idx, morph in enumerate(chosen):
            img = self.generate_galaxy(morphology=morph)
            img = self.apply_psf_blur(img, sigma=psf_sigma)
            img = self.apply_gaussian_noise(img, std=noise_std, rng=self._rng)
            images[idx] = img
            labels[idx] = self.morphologies[morph]

        logger.info("Batch generation complete.")
        return images, labels
