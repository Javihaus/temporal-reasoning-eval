# Temporal Constraint Processing in Language Models: Empirical Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Transformers](https://img.shields.io/badge/Transformers-4.36+-blue.svg)](https://github.com/huggingface/transformers)
[![Code style: research](https://img.shields.io/badge/code%20style-research-brightgreen.svg)](https://github.com/jmarin/temporal-reasoning-eval)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/temporal-reasoning-eval/graphs/commit-activity)

> Systematic characterization of temporal constraint processing failures in production-scale language models (2.8-8B parameters)

## Overview

This repository accompanies our paper **"Temporal Constraint Processing in Autoregressive Language Models: Empirical Characterization and Architectural Remedies"** and provides complete experimental notebooks documenting systematic evaluation of temporal reasoning capabilities across eight language models.

### Key Findings

- **Bimodal performance**: Models achieve either >95% or <50% accuracy on deadline detection
- **Extreme prompt brittleness**: 30-60 percentage point drops from formatting changes alone
- **Systematic action bias**: Failing models show 100% false positive rates
- **Scale independence**: No correlation between parameters (2.8-8B) and capability
- **Learnability**: Fine-tuning on 200 examples improves partial capability by 12-37pp

## Repository Contents

```
temporal-reasoning-eval/
├── notebooks/
│   ├── temporal_reasoning_experiments.ipynb      # Main evaluation pipeline
│   ├── temporal_reasoning_failure_analysis.ipynb # Response pattern analysis  
│   └── Temporal_experiments.ipynb                # Initial experimental design
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Tested Models

| Model | Parameters | Architecture | Access |
|-------|------------|--------------|--------|
| Qwen2.5-7B-Instruct | 7B | Transformer | Public |
| DeepSeek-R1-Distill | 7B | Transformer | Public |
| Llama-3.1-8B | 8B | Transformer | Gated* |
| Mistral-7B-v0.3 | 7B | Transformer | Public |
| Phi-3-mini | 3.8B | Transformer | Public |
| Mamba-2.8B | 2.8B | State Space | Public |
| Jamba-1.5-mini | 7B | Hybrid | Gated* |
| RWKV-6 | 3B | Recurrent | Public |

*Requires HuggingFace access approval 

## Quick Start

### Requirements

```bash
# Core dependencies
pip install transformers>=4.36.0 torch>=2.1.0 accelerate>=0.25.0
pip install bitsandbytes>=0.42.0 peft>=0.7.0 datasets>=2.16.0

# For Mamba architecture
pip install mamba-ssm causal-conv1d>=1.2.0
```

**Hardware**: Minimum 24GB GPU RAM for 8B models with 8-bit quantization
- Tested on: NVIDIA A100 (40GB)
- Compatible with: A100, V100 (32GB), A6000

### Running Experiments

**Option 1: Google Colab (Recommended)**
1. Open notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmarin/temporal-reasoning-eval/blob/main/notebooks/temporal_reasoning_experiments.ipynb)
2. Runtime → Change runtime type → GPU (A100 if available)
3. Run cells sequentially

**Option 2: Local GPU**
```bash
git clone https://github.com/jmarin/temporal-reasoning-eval.git
cd temporal-reasoning-eval
jupyter notebook notebooks/temporal_reasoning_experiments.ipynb
```

### Test Scenarios

The evaluation uses 8 temporal constraint scenarios (4 open windows, 4 closed):
- Emergency response timing
- Financial trading windows  
- Project deadline management
- Medical treatment windows

Scenarios are embedded in notebook cells 207-320 as Python dictionaries with:
- Explicit temporal information (deadlines, elapsed time)
- Binary ground truth (YES/NO)
- Window status classification (open/closed)

## Reproducibility Notes

### What Will Reproduce
-**Qualitative patterns**: Bimodal distribution, prompt brittleness, action bias  
-**Model rankings**: Relative performance across architectures  
-**Fine-tuning improvements**: Direction and rough magnitude of gains

### What May Vary
**Exact accuracy numbers**: ±5-10 percentage points due to:
- Model version updates on HuggingFace Hub
- GPU non-determinism in generation
- 8-bit quantization variability

### Known Limitations
- Models accessed October 2025
- Non-deterministic despite random seed setting
- Colab session constraints require manual memory management
- Access-gated models need HuggingFace authentication

## Code Philosophy

**These are research artifacts, not production software.**

The notebooks document our actual experimental process, including:
- Interactive model loading and memory management
- Manual parameter exploration
- Conversational commentary explaining reasoning
- Colab-specific workflow adaptations

We believe this transparency serves researchers better than artificially polished code that misrepresents the research process. Independent researchers working with similar constraints can see and adapt our actual workflow.

For production evaluation frameworks, we recommend adapting our methodology to your infrastructure rather than using these notebooks directly.

## Citation

If you use this work, please cite:

```bibtex
@article{marin2025temporal,
  title={Empirical Characterization of Temporal Constraint Processing in LLMs},
  author={Marin, Javier},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Results Summary

### Baseline Performance (Experiment 1)

| Model | Accuracy | False Positive Rate | Performance Class |
|-------|----------|---------------------|-------------------|
| Qwen2.5-7B | 100% | 0% | Perfect |
| DeepSeek-R1 | 100% | 0% | Perfect |
| Phi-3-mini | 100% | 0% | Perfect |
| Mamba-2.8B | 45.5% | 59% | Partial |
| Jamba-1.5 | 0% | 100% | Systematic failure |
| RWKV-6 | 0% | 100% | Systematic failure |

### Fine-Tuning Impact (Experiment 2)

| Model | Before | After | Change |
|-------|--------|-------|--------|
| Llama-3.1-8B | 25.0% | 62.5% | +37.5pp |
| Qwen2.5-7B | 62.5% | 87.5% | +25.0pp |
| Mistral-7B | 62.5% | 75.0% | +12.5pp |
| DeepSeek-R1 | 37.5% | 50.0% | +12.5pp |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Javier Marín**  
Applied AI Consultant | Production AI Systems + Regulatory Compliance
javier@jmarin.info  
[GitHub](https://github.com/jmarin)

## Acknowledgments

Experiments conducted using Google Colab Pro with NVIDIA A100 GPUs. Models accessed via HuggingFace Hub. We thank the open-source ML community for providing accessible infrastructure enabling independent research.

---

**Note**: This research was conducted independently with limited computational resources. The findings document deployment risks in current production-scale models and provide empirical foundation for architectural innovation in temporal processing.
