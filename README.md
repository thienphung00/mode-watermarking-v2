# Mode Watermarking

A watermarking system for Stable Diffusion models. Embed imperceptible watermarks during image generation and detect them via a Bayesian pipeline with DDIM inversion.

**License:** MIT  
**Python:** 3.8+

---

## Features

- **Watermark embedding** — Latent-space injection (G-field, masks, scheduling) during diffusion sampling
- **Watermark detection** — Bayesian detection with DDIM inversion, g-value statistics, and calibration
- **Key management** — Per-key watermark secrets; master keys stay on the API service
- **HTTP API** — Key registration, image generation with watermarking, and detection endpoints
- **GPU worker** — Separate service for heavy inference (Stable Diffusion + watermark ops)

---

## Project structure

```
mode-watermarking-restructured/
├── src/                    # Core library
│   ├── core/               # Config, strategies, context, key utils
│   ├── algorithms/         # G-field, masks, alpha scheduling
│   ├── detection/          # Inversion, g-values, statistics, API bindings
│   ├── engine/             # Pipeline, hooks, strategies (e.g. seed bias)
│   ├── models/             # Detector models and layers
│   └── ablation/           # Ablation datasets, family signatures, g-value compute
├── service/                # API + GPU worker (see service/README.md)
│   ├── api/                # FastAPI app, routes, key store, GPU client
│   └── gpu/                # SD pipeline + watermark generation/detection
├── scripts/                # Training, ablation, precompute, evaluation
├── tests/                  # Unit and integration tests
├── pyproject.toml
├── requirements.txt
└── install.sh
```

---

## Installation

### From source

```bash
git clone https://github.com/thienphung00/mode-watermarking.git
cd mode-watermarking

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

### Using the install script

```bash
./install.sh              # pip, default
./install.sh --conda      # conda env + pip
./install.sh --dev        # with dev dependencies
./install.sh --help       # options
```

### Requirements

- Python 3.8+
- PyTorch ≥ 2.0, torchvision, diffusers, transformers
- NumPy, SciPy, Pillow, scikit-learn
- Optional: wandb, tensorboard, kagglehub (for datasets)

See `pyproject.toml` and `requirements.txt` for full dependency lists.

---

## Library usage

From the project root (e.g. after `pip install -e .` or with `PYTHONPATH=.`), use the library for config, strategies, and (lazy-loaded) engine:

```python
from src import (
    AppConfig,
    WatermarkStrategy,
    LatentInjectionStrategy,
    GFieldGenerator,
    MaskGenerator,
    AlphaScheduler,
)
# Engine (lazy): create_pipeline, generate_with_watermark, apply_watermark_hook
```

Detection components (inversion, g-values, statistics) live under `src.detection` and are used by the service; see `src/detection/api.py` and `service/` for integration.

---

## Running the API service

The HTTP API (key registration, generation, detection) and the GPU worker are documented in **service/README.md**.

**Quick start (no GPU):**

```bash
cd service
python -m service.api.main
# In another terminal:
STUB_MODE=true python -m service.gpu.main
```

**With Docker:**

```bash
docker-compose -f service/docker-compose.yml up
```

- API: `http://localhost:8000`  
- Docs: `http://localhost:8000/docs`  
- Demo UI: `http://localhost:8000/demo`

---

## Scripts

- **Ablation / evaluation:** `scripts/run_ablation_*.py`, `scripts/evaluate_bayesian_detector.py`
- **Training / precompute:** `scripts/train_g_likelihoods.py`, `scripts/precompute_inverted_g_values.py`
- **Data:** `scripts/generate_training_images.py`, `scripts/generate_ablation_configs*.py`

Run with `python -m scripts.<script_name>` from the project root (or adjust `PYTHONPATH`).

---

## Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Links

- **Repository:** [github.com/thienphung00/mode-watermarking](https://github.com/thienphung00/mode-watermarking)
- **Service details:** [service/README.md](service/README.md)
