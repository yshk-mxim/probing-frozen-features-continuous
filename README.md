# GeoProbe

**Probing foundation models for continuous geometric measurement**

Code for the paper: *"Do Foundation Models Know Geometry? Probing Frozen Features for Continuous Physical Quantities"*

## Key Results

Linear probes on frozen foundation model features can predict continuous geometric quantities (joint angles, head pose, object pose) with surprising accuracy. No fine-tuning required.

| Model | Paradigm | Params | FreiHAND R² | BIWI R² |
|-------|----------|--------|-------------|---------|
| DINOv3-ViT-L/16 | Self-supervised | 303M | 0.487 | **0.794** |
| SigLIP2-ViT-L/16 | Hybrid VL | 304M | **0.513** | 0.712 |
| DINOv2-ViT-L/14 | Self-supervised | 304M | 0.401 | 0.688 |
| SigLIP-ViT-L/16 | Contrastive VL | 304M | 0.482 | 0.695 |
| CLIP-ViT-L/14 | Contrastive VL | 428M | 0.378 | 0.629 |
| Qwen2.5-VL-7B | Autoregressive | 7B | 0.342 | 0.581 |
| Qwen2.5-VL-3B | Autoregressive | 3B | 0.318 | 0.543 |
| QwenVIT-3B | VLM Vision Enc. | 675M | 0.295 | 0.512 |
| InternViT-6B | Vision Encoder | 6B | 0.361 | 0.647 |
| Gemma3-4B-IT | Autoregressive | 4B | 0.289 | 0.498 |

**Key finding**: Autoregressive generation damages geometric information — NOT language alignment per se. Contrastive and self-supervised encoders preserve geometry better than generative VLMs, even at smaller scale.

## Installation

```bash
# Clone the repository
git clone https://github.com/yshkolnikov/geoprobe.git
cd geoprobe

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

**Requirements**: Python >= 3.11, PyTorch >= 2.1, ~24 GB RAM for largest models.

## Quick Start

```bash
# Download FreiHAND dataset
./scripts/download_data.sh --dataset freihand

# Run a single model on FreiHAND (2K images, ~10 min)
geoprobe reproduce --config configs/freihand_2k.toml --device auto

# Run everything
./scripts/run.sh --all
```

## Usage

### Extract features
```bash
geoprobe extract --model dinov3 --dataset freihand --n-total 2000
```

### Train probes
```bash
geoprobe probe --features features_cache/dinov3_L20_freihand.pt --method rrr
```

### Statistical evaluation
```bash
geoprobe evaluate --results-dir results/
```

### Full reproduction
```bash
# Reproduce paper Table 3 (FreiHAND 8K)
geoprobe reproduce --config configs/freihand_8k.toml

# Reproduce everything
./scripts/run.sh --all
```

## Available Models

| Key | Model | HuggingFace ID | Memory |
|-----|-------|---------------|--------|
| `dinov3` | DINOv3-ViT-L/16 | facebook/dinov2-with-registers-large | ~1.5 GB |
| `dinov2` | DINOv2-ViT-L/14 | facebook/dinov2-large | ~1.5 GB |
| `siglip` | SigLIP-ViT-L/16 | google/siglip-large-patch16-384 | ~1.5 GB |
| `siglip2` | SigLIP2-ViT-L/16 | google/siglip2-large-patch16-384 | ~1.5 GB |
| `clip` | CLIP-ViT-L/14 | openai/clip-vit-large-patch14 | ~2 GB |
| `qwen-3b` | Qwen2.5-VL-3B | Qwen/Qwen2.5-VL-3B-Instruct | ~7 GB |
| `qwen-7b` | Qwen2.5-VL-7B | Qwen/Qwen2.5-VL-7B-Instruct | ~15 GB |
| `qwenvit-3b` | QwenVIT-3B | Qwen/Qwen2.5-VL-3B-Instruct | ~3 GB |
| `internvit` | InternViT-6B | OpenGVLab/InternViT-6B-448px-V2-5 | ~12 GB |
| `gemma3` | Gemma3-4B-IT | google/gemma-3-4b-it | ~9 GB |

## Datasets

- **FreiHAND**: Hand pose (finger joint angles from 3D keypoints). 32,560 unique images.
- **BIWI**: Head pose (pitch/yaw/roll from depth sensor). ~15,000 frames.
- **YCB-Video**: Object pose (6D translation + rotation). 21 objects.

## Adding a New Model

Implement the `FeatureExtractor` protocol and register in `extraction/registry.py`:

```python
from geoprobe.extraction.base import FeatureExtractor, get_device, pool_features
from geoprobe.domain.value_objects import ModelSpec

class MyModelExtractor:
    model_name = "my-model"

    def __init__(self, spec: ModelSpec, **kwargs):
        self.spec = spec

    def extract(self, images, layer, batch_size=1, device="auto"):
        # Load model, register hooks, extract features
        ...

    def cleanup(self):
        ...
```

See [docs/adding-models.md](docs/adding-models.md) for a full tutorial.

## Hardware Requirements

- **Minimum**: 16 GB RAM, any GPU/CPU (small models: DINOv2, CLIP, SigLIP)
- **Recommended**: 24 GB RAM, Apple M4 Pro or NVIDIA GPU with 16+ GB VRAM
- **Full reproduction**: ~48 hours on M4 Pro (all 10 models × 3 datasets)

## Project Structure

```
src/geoprobe/
├── domain/         # Entities, value objects, error hierarchy
├── extraction/     # Model-specific feature extractors
├── probes/         # Ridge, RRR, MLP probe implementations
├── evaluation/     # Metrics, bootstrap, permutation tests, corrections
├── datasets/       # FreiHAND, BIWI, YCB-Video loaders
├── config/         # Pydantic settings, defaults
└── cli.py          # Typer CLI interface
```

## Citation

```bibtex
@article{shkolnikov2026geoprobe,
  title={Do Foundation Models Know Geometry? Probing Frozen Features for Continuous Physical Measurement},
  author={Shkolnikov, Yakov Pyotr},
  journal={arXiv preprint arXiv:2603.06459},
  year={2026},
  url={https://arxiv.org/abs/2603.06459}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
