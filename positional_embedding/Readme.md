# Positional Embeddings in Transformer Models

This project demonstrates the implementation of absolute and rotary positional embeddings (RoPE) in a transformer model using PyTorch. The goal is to explore different positional embeddings mechanisms used to encode positions of tokens along with the context to improve the performance of self-attention mechanisms in transformers.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [References](#references)


## Overview

Positional embeddings are crucial in transformer models as they provide information about the position of tokens in a sequence. This project covers:

1. **Absolute Positional Embeddings**: Traditional method where positional information is added directly to the token embeddings.
2. **Rotary Positional Embeddings (RoPE)**: A method where positional information is applied to the query and key vectors, enhancing the attention mechanism's ability to capture relative positions.

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/anmol-c03/positional_embedding.git
    cd positional-embeddings
    ```

2. Install the required packages:
    ```bash
    pip install torch numpy
    ```

## Refrences

## References

- Su, J., Gao, Y., & Wang, B. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is All You Need](https://arxiv.org/abs/1706.03762).
- [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer/tree/main?tab=readme-ov-file)
- [bojone/bert4keras](https://github.com/bojone/bert4keras/tree/master)
