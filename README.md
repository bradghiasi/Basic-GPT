BasicLLM (Basic GPT from Scratch) is a lightweight decoder-only Transformer language model implemented from the ground up in PyTorch. The project recreates the core building blocks of modern LLMs by training a small autoregressive model on Persian literature and generating literary-style Persian sentences and phrases through autoregressive sampling.



### Key Features

Decoder-only Transformer architecture (mini-GPT style)

Character-level tokenization

Context-window batching for training sequences

Cross-entropy loss + AdamW optimization

Text generation via autoregressive decoding/sampling



Tech Stack

Python, PyTorch (CUDA optional)



### Purpose

This repo is designed to demonstrate hands-on understanding of LLM fundamentals—model architecture, training dynamics, and embedding-based sequence modeling—by building an end-to-end implementation without relying on high-level LLM frameworks.

