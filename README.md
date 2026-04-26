# Galaxy-ML-Robustness

Benchmarking the robustness of machine-learning models for galaxy morphology classification under realistic observational degradations (noise, PSF blur, and more).

## Project Structure

```
Galaxy-ML-Robustness/
├── requirements.txt              # Python dependencies
├── simulations/
│   ├── __init__.py
│   └── data_generation.py        # Synthetic galaxy image simulator
├── models/
│   ├── __init__.py
│   └── robust_cnn.py             # PyTorch CNN for galaxy classification
└── README.md
```

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/Yash5852176/Galaxy-ML-Robustness.git
cd Galaxy-ML-Robustness
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic data
python -c "
from simulations.data_generation import GalaxyImageSimulator
sim = GalaxyImageSimulator(image_size=128, random_seed=42)
images, labels = sim.generate_batch(n_samples=64, noise_std=0.05, psf_sigma=1.5)
print(f'Generated {images.shape[0]} images, shape={images.shape}')
"

# 3. Instantiate the model
python -c "
import torch
from models.robust_cnn import RobustGalaxyCNN
model = RobustGalaxyCNN(in_channels=1, num_classes=3)
x = torch.randn(4, 1, 128, 128)
print(f'Output shape: {model(x).shape}')
"
```

## Key Components

### `simulations/data_generation.py`
- **`GalaxyImageSimulator`** — generates synthetic galaxy images (elliptical, spiral, irregular) with Sérsic profiles and optional spiral arms.
- Supports configurable **Gaussian noise** and **PSF blur** for robustness benchmarking.
- Reproducible via NumPy random seeding.

### `models/robust_cnn.py`
- **`RobustGalaxyCNN`** — a 4-stage convolutional network with residual blocks, batch normalisation, and dropout.
- Adaptive global pooling allows arbitrary input resolutions.
- Includes `predict()` for inference and `feature_maps()` for intermediate activations.

## License

MIT
