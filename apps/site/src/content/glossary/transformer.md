---
aliases:
- transformer model
- transformer architecture
category: deep-learning
difficulty: advanced
related:
- attention-mechanism
- bert
- gpt
- self-attention
sources:
- author: Ashish Vaswani et al.
  license: cc-by
  source_title: Attention Is All You Need
  source_url: https://arxiv.org/abs/1706.03762
- author: Jay Alammar
  license: cc-by-sa
  source_title: The Illustrated Transformer
  source_url: https://jalammar.github.io/illustrated-transformer/
summary: The Transformer is a deep learning architecture introduced in 2017 that revolutionized
  natural language processing through its attention mechanism. It enables parallel
  processing of sequences and forms the foundation for modern language models like
  GPT, BERT, and T5 by allowing models to focus on relevant parts of input sequences
  regardless of their position.
tags:
- deep-learning
- nlp
- machine-learning
- fundamentals
- algorithms
title: Transformer
updated: '2025-01-15'
---

## Overview

The Transformer architecture, introduced in the groundbreaking 2017 paper "Attention Is All You Need,"
fundamentally changed how we approach sequence-to-sequence tasks in machine learning. Unlike previous
architectures that relied on recurrent or convolutional layers, Transformers use only attention mechanisms
to capture dependencies between input and output sequences.

## Key Innovation: Attention Mechanism

The core breakthrough of the Transformer is its **self-attention mechanism**, which allows the model to:

- **Focus on relevant information**: Determine which parts of the input are most important for each output
- **Capture long-range dependencies**: Connect distant elements in a sequence without the limitations of recurrent
  models
- **Process in parallel**: Unlike RNNs, attention operations can be computed simultaneously across all positions

### Self-Attention Formula

The attention mechanism computes attention weights using three vectors derived from the input:

```text
Attention(Q, K, V) = softmax(QK^T / √d_k)V

```text
Where:

- **Q (Query)**: What information we're looking for
- **K (Key)**: What information is available
- **V (Value)**: The actual information content
- **d_k**: Dimension of the key vectors (for scaling)

## Architecture Components

### 1. Multi-Head Attention

Instead of using a single attention function, Transformers use multiple "heads" that learn different types of
relationships:

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

```text
This allows the model to simultaneously attend to information from different representation subspaces.

### 2. Position Encoding

Since attention doesn't inherently understand sequence order, Transformers add positional information:

- **Sinusoidal encoding**: Uses sine and cosine functions of different frequencies
- **Learned embeddings**: Position vectors learned during training
- **Relative positional encoding**: Encodes relative distances between positions

### 3. Layer Structure

Each Transformer layer consists of:

- **Multi-head self-attention**: Processes relationships within the sequence
- **Feed-forward network**: Applies point-wise transformations
- **Residual connections**: Enable deep network training
- **Layer normalization**: Stabilizes training

## Encoder-Decoder Architecture

### Encoder Stack

- Processes the input sequence
- Each layer has self-attention and feed-forward sublayers
- Typically 6-12 layers in practice
- Generates rich representations of the input

### Decoder Stack

- Generates the output sequence autoregressively
- Has self-attention, encoder-decoder attention, and feed-forward layers
- Uses **masked self-attention** to prevent looking at future tokens during training

## Training Process

### Teacher Forcing

During training, the model sees the entire target sequence:

- Input: "The cat sat on the mat"
- Target: "Le chat s'est assis sur le tapis"
- The model learns to predict each target token given the source and previous target tokens

### Inference

At test time, generation is autoregressive:

1. Start with a special beginning token
2. Generate one token at a time
3. Feed each generated token back as input for the next step
4. Continue until an end token is produced

## Impact and Applications

### Language Models

- **BERT**: Bidirectional encoder for understanding tasks
- **GPT**: Autoregressive decoder for generation tasks
- **T5**: Text-to-text unified framework

### Beyond NLP

Transformers have been successfully adapted for:

- **Computer Vision**: Vision Transformer (ViT)
- **Protein Folding**: AlphaFold's attention mechanisms
- **Reinforcement Learning**: Decision Transformer
- **Multimodal Tasks**: CLIP, DALL-E

## Advantages

### Parallelization

- Unlike RNNs, all positions can be processed simultaneously
- Enables efficient training on modern hardware

### Long-Range Dependencies

- Direct connections between any two positions
- No degradation over long sequences

### Interpretability

- Attention weights provide insights into model decisions
- Can visualize what the model focuses on

## Limitations

### Computational Complexity

- Attention scales quadratically with sequence length: O(n²)
- Memory and compute intensive for very long sequences

### Inductive Biases

- Lacks built-in understanding of sequence order (requires position encoding)
- May need more data to learn patterns that RNNs capture naturally

## Variants and Improvements

### Efficient Transformers

- **Linformer**: Linear attention complexity
- **Performer**: Uses random features for attention approximation
- **Longformer**: Sliding window attention for long documents

### Architectural Innovations

- **GPT**: Decoder-only architecture
- **BERT**: Encoder-only with bidirectional attention
- **T5**: Unified text-to-text framework

## Programming Example

Here's a simplified attention mechanism:

```python
import torch
import torch.nn as nn
import math

class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.q_linear = nn.Linear(d_model, d_k)
        self.k_linear = nn.Linear(d_model, d_k)
        self.v_linear = nn.Linear(d_model, d_k)
    
    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

```text
The Transformer architecture's elegance lies in its simplicity and effectiveness. By showing that
"attention is all you need," it opened the door to the modern era of large language models and continues
to drive innovations in AI across multiple domains.
