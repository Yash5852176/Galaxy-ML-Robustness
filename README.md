# Galaxy-ML-Robustness: Weak-Lensing Galaxy Analysis 🌌

Analyzing noise and blur resilience in ML-based weak-lensing galaxy image interpretation.

## 📌 Overview
Weak gravitational lensing is a critical probe of the dark sector. However, realistic observational artifacts like sky noise and atmospheric Point Spread Function (PSF) blur can degrade the performance of shape-measurement ML models. This project quantifies this degradation and explores robust training strategies.

## 🚀 Technical Methodology
- **Simulation Engine:** Uses `GalSim` (or similar) to generate synthetic galaxy images with varyng Sersic profiles.
- **Artifact Injection:** Implementation of additive Gaussian noise and Moffat/Gaussian PSF convolution kernels.
- **Robustness Metric:** Evaluation of model MSE and multiplicative/additive bias ($m$ and $c$ values) across signal-to-noise ratios (SNR).

## 🛠️ Architecture
1. `data_gen.py`: Script to generate high-fidelity and artifact-degraded image pairs.
2. `robust_trainer.py`: PyTorch-based training script with data augmentation for noise resilience.
3. `metrics.ipynb`: Analysis of model bias and performance stability.

## 📂 Tech Stack
- **Core:** Python, NumPy, SciPy
- **Astro Tools:** Astropy, GalSim
- **ML:** PyTorch, Matplotlib

---
*Part of the Research Portfolio for Yash Yadav.* ✨