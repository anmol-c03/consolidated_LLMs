
# Tokenizer
# Table of Contents
1. [Project Overview](#project-overview)
2. [Usage](#Usage)
4. [Modules](#modules)
5. [References](#references)

## Project Overview
This project implemets the tokenization using Binary Pair Encoding. Binary Pair Encoding (BPE) is a subword tokenization technique that addresses the limitations of character-level tokenization, which often loses semantics, fails to capture morphemes, and reduces context. BPE begins by calculating the frequency of consecutive character pairs in a dataset. The most frequent pair is merged into a new token, and this process repeats until the vocabulary reaches a predefined size. This approach balances capturing semantics and syntax while maintaining a manageable vocabulary, making it effective for NLP tasks.

## Usage 
One can train their own tokenizer using train.py.

## Modules 
1. Basic_tokenizer :
This tokenizer solely utilises the BPE  for tokenization. This tokenizer treats dog., dog!, and dog? as spearate tokens. Moreover, this tokenization results in poor handling of numbers and special characters and results in case sensitivity. Hence, regex tokenizer is adopted. It contains 3 different files. One can explore the merges in basic.model and basic.vocab.

2. regex_tokenizer:
To overcome the limitations of basic_tokenization, regex pattern was utilized before using BPE. One can explore more about it through regex_tokenizer.py. For better performance, one can change regex pattern as per their dialect if using for local languages.

## References

BPE https://arxiv.org/pdf/1508.07909v5

utf-8 https://en.wikipedia.org/wiki/UTF-8
