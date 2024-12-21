# Consolidated Repository for NLP and ML Projects

Welcome to the consolidated repository that houses multiple submodules focused on various NLP and machine learning concepts. Each submodule in this repository represents an individual project, bringing together a spectrum of experiments and implementations.

# Repository Structure:

This repository includes the following submodules:

1. **Positional Embedding**
    -  implementation and analysis of absolute position encoding and rotational position encoding.

2. **Tokenizations**
    - Implementation of byte-pair encoding (BPE) with regex patterns matching for tokenization pattern.


3. **Language Modelling**
    - a gpt2 like transformer-based pretrained model with weight tying and simplified KV cache.

4. **opt_gpt**
    - Extension of Language_modelling repo, explores distributed training and proper pipeline.

5. **Fine-Tuning Concepts**
    - Techniques and methodologies for fine-tuning pre-trained models, includes instruction tuning,quantization and performance alignment.

# Getting Started

To work with this consolidated repository, follow the steps below:

Cloning the Repository
```bash
# Clone the main repository
git clone --recurse-submodules https://github.com/anmol-c03/consolidated_LLMs
```

Setting Up Submodules

Each submodule contains its own README with setup instructions. Navigate to the respective submodule directories for more details.
```bash
cd submodule-name
# Follow the instructions specific to the submodule
```
# Prerequisites

Python 3.8+

pipenv or virtualenv for dependency management

GPU support for deep learning projects (optional but recommended)

# Installation

Ensure dependencies for all submodules are installed:
```bash
pip install -r requirements.txt
```
Note: Some submodules maynot have their own specific requirements. Refer to this  requirements.txt files.

# Contribution Guidelines

Contributions to this repository or its submodules are welcome! Please ensure:
- Proper documentation for any new additions.
- Compliance with the existing coding style and standards.