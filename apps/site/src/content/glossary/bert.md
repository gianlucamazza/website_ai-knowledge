---
aliases:
- BERT
- bidirectional encoder
- BERT model
- masked language model
category: nlp
difficulty: intermediate
related:
- transformer
- llm
- gpt
- attention-mechanism
- fine-tuning
sources:
- author: Jacob Devlin et al.
  license: cc-by
  source_title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding'
  source_url: https://arxiv.org/abs/1810.04805
- author: Google AI
  license: proprietary
  source_title: 'Open Sourcing BERT: State-of-the-art Pre-training for Natural Language
    Processing'
  source_url: https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
summary: BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based
  language model developed by Google that revolutionized natural language understanding
  by using bidirectional context. Unlike autoregressive models like GPT, BERT can
  attend to both past and future tokens simultaneously, making it exceptionally effective
  for understanding tasks like question answering, sentiment analysis, and text classification.
tags:
- nlp
- deep-learning
- transformer
- machine-learning
- fundamentals
title: BERT (Bidirectional Encoder Representations from Transformers)
updated: '2025-01-15'
---

## Overview

BERT (Bidirectional Encoder Representations from Transformers) represents a fundamental shift in how language models
process text. Introduced by Google in 2018, BERT's key innovation is its ability to understand context from both
directions simultaneously—looking at words that come before AND after a target word. This bidirectional approach makes
BERT exceptionally powerful for language understanding tasks, though it cannot generate text like GPT models.

## Core Innovation: Bidirectional Context

### Traditional vs. Bidirectional Processing

### Traditional Left-to-Right Processing (like GPT)

```text
"The cat sat on the [MASK]"
Context available: "The cat sat on the"
Missing context: What comes after [MASK]

```text
### BERT's Bidirectional Processing

```text
"The cat sat on the [MASK] under the tree"
Context available: "The cat sat on the" + "under the tree"  
Full context: Both directions help predict [MASK] = "mat/ground/grass"

```text
This bidirectional understanding allows BERT to make more informed predictions by considering the complete context.

### Architectural Foundation

BERT uses a **encoder-only** transformer architecture:

- **Multi-Head Self-Attention**: Can attend to any position in the input sequence
- **No Causal Masking**: Unlike GPT, all tokens can see all other tokens
- **Deep Bidirectional**: Every layer processes full bidirectional context
- **Position Embeddings**: Learned positional encodings for sequence understanding

## Pre-training Approach

### Masked Language Modeling (MLM)

BERT's primary training objective involves randomly masking tokens and predicting them:

```python

# Example MLM training data

original_text = "The quick brown fox jumps over the lazy dog"
masked_text = "The [MASK] brown fox [MASK] over the lazy dog"
targets = ["quick", "jumps"]

## BERT learns to predict masked tokens using bidirectional context

```text
### Masking Strategy

- 15% of tokens are selected for masking
- Of these selected tokens:
  - 80% replaced with [MASK] token
  - 10% replaced with random token  
  - 10% left unchanged (but still predicted)

This strategy prevents overfitting to the [MASK] token and improves robustness.

### Next Sentence Prediction (NSP)

BERT also learns to understand relationships between sentences:

```python

## Training examples for NSP

positive_example = {
    "sentence_a": "The cat sat on the mat.",
    "sentence_b": "It was a comfortable spot to rest.",
    "is_next": True
}

negative_example = {
    "sentence_a": "The cat sat on the mat.",  
    "sentence_b": "The stock market opened higher today.",
    "is_next": False
}

```text
This objective helps BERT understand discourse and paragraph-level coherence.

## Model Architecture

### Input Representation

BERT combines three types of embeddings:

```python

## BERT input embedding

total_embedding = (
    token_embeddings +      # WordPiece vocabulary
    segment_embeddings +    # Sentence A vs Sentence B
    position_embeddings     # Absolute position in sequence
)

```text

#### Token Embeddings

- Uses WordPiece tokenization with 30,000 vocabulary
- Handles out-of-vocabulary words by breaking into subwords
- Special tokens: [CLS], [SEP], [MASK], [PAD]

#### Segment Embeddings  

- Distinguishes between sentence A and sentence B
- Enables processing of sentence pairs for tasks like question answering

#### Position Embeddings

- Learned embeddings for each position (up to 512 tokens)
- Different from transformer's sinusoidal positional encoding

### Model Variants

#### BERT-Base

```python
bert_base_config = {
    "hidden_size": 768,        # Embedding dimension
    "num_layers": 12,          # Transformer layers  
    "num_attention_heads": 12, # Multi-head attention
    "intermediate_size": 3072, # Feed-forward hidden size
    "vocab_size": 30522,       # WordPiece vocabulary
    "max_position_embeddings": 512,  # Max sequence length
    "parameters": "110M"       # Total parameters
}

```text

#### BERT-Large  

```python
bert_large_config = {
    "hidden_size": 1024,       # Larger embeddings
    "num_layers": 24,          # Deeper network
    "num_attention_heads": 16, # More attention heads
    "intermediate_size": 4096, # Larger feed-forward
    "parameters": "340M"       # Total parameters
}

```text

## Training Process

### Pre-training Data

BERT was trained on a massive text corpus:

- **BookCorpus**: 800M words from 11,038 books
- **English Wikipedia**: 2,500M words from article text
- **Total**: ~3.3 billion words of high-quality text
- **Preprocessing**: Sentence segmentation and tokenization

### Training Specifications

#### BERT-Base Training

- **Hardware**: 4 Cloud TPUs (16 TPU chips total)
- **Duration**: 4 days
- **Batch Size**: 256 sequences
- **Learning Rate**: 1e-4 with linear decay
- **Optimization**: Adam with weight decay

#### BERT-Large Training  

- **Hardware**: 16 Cloud TPUs (64 TPU chips total)
- **Duration**: 4 days
- **Batch Size**: 256 sequences
- **Cost**: Estimated $3,000-7,000 in compute

### Training Objectives

The loss function combines both pre-training tasks:

```python
def bert_loss(masked_tokens, nsp_labels, predictions):
    # Masked Language Model loss
    mlm_loss = cross_entropy(
        predictions.mlm_logits[masked_positions],
        masked_tokens
    )
    
    # Next Sentence Prediction loss  
    nsp_loss = cross_entropy(
        predictions.nsp_logits,
        nsp_labels
    )
    
    # Combined loss
    total_loss = mlm_loss + nsp_loss
    return total_loss

```text

## Fine-tuning for Downstream Tasks

### Task-Specific Adaptations

BERT can be fine-tuned for various NLP tasks with minimal architecture changes:

#### Text Classification

```python
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

```text

#### Question Answering

```python
class BERTQuestionAnswering(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.qa_outputs = nn.Linear(bert_model.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Predict start and end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

```text

#### Named Entity Recognition

```python
class BERTTokenClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Classify each token
        return self.classifier(sequence_output)

```text

### Fine-tuning Process

```python

## Fine-tuning example for sentiment analysis

from transformers import BertForSequenceClassification, BertTokenizer

## Load pre-trained model

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)

## Fine-tune on task-specific data

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):  # Typically 2-4 epochs
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()

```text

## Key Applications and Performance

### GLUE Benchmark Results

BERT achieved state-of-the-art results on the General Language Understanding Evaluation (GLUE) benchmark:

| Task | BERT-Base | BERT-Large | Previous Best |
|------|-----------|------------|---------------|
| MNLI | 84.6/83.4 | 86.7/85.9  | 80.5/80.1     |
| QQP  | 89.2      | 89.3       | 84.6          |
| QNLI | 90.5      | 92.7       | 82.3          |
| SST-2| 93.5      | 94.9       | 93.2          |
| CoLA | 52.1      | 60.5       | 35.0          |
| STS-B| 85.8      | 87.1       | 81.8          |
| MRPC | 88.9      | 89.3       | 86.8          |
| RTE  | 66.4      | 70.1       | 61.7          |

### Reading Comprehension

**SQuAD 1.1** (Stanford Question Answering Dataset):

- BERT-Large: 93.2 F1 score
- Previous best: 91.8 F1 score
- Human performance: 91.2 F1 score

**SQuAD 2.0** (with unanswerable questions):

- BERT-Large: 83.1 F1 score  
- Previous best: 76.3 F1 score
- Human performance: 89.5 F1 score

### Named Entity Recognition

**CoNLL-2003 NER**:

- BERT-Large: 96.4 F1 score
- Previous best: 95.7 F1 score

## BERT Variants and Improvements

### RoBERTa (Robustly Optimized BERT)

Facebook's improvements to BERT training:

- **Removed NSP**: Next Sentence Prediction found to be unhelpful
- **Dynamic Masking**: Different masking patterns for each epoch
- **Larger Batches**: 8K sequences vs BERT's 256
- **More Data**: 160GB of text vs BERT's 16GB
- **Longer Training**: 500K steps vs 1M steps

Results: Significant improvements across multiple benchmarks.

### ALBERT (A Lite BERT)

Parameter-efficient version of BERT:

- **Parameter Sharing**: Share parameters across layers
- **Factorized Embeddings**: Separate vocabulary and hidden size
- **Sentence Order Prediction**: Replace NSP with SOP task

Results: Better performance with 10x fewer parameters.

### DistilBERT

Knowledge distillation for efficient deployment:

- **60% smaller** than BERT-Base
- **60% faster** inference
- **97% of BERT's performance** retained

### DeBERTa (Decoding-enhanced BERT)

Microsoft's enhanced attention mechanism:

- **Disentangled Attention**: Separate content and position representations  
- **Enhanced Mask Decoder**: Better handling of relative positions
- **Virtual Adversarial Training**: Improved robustness

## Practical Implementation

### Using BERT with Hugging Face

```python
from transformers import BertTokenizer, BertModel
import torch

## Initialize tokenizer and model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

## Tokenize input

text = "BERT is a powerful language model."
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

## Get BERT embeddings

with torch.no_grad():
    outputs = model(**inputs)
    
## Access different outputs

last_hidden_states = outputs.last_hidden_state  # All token representations
pooler_output = outputs.pooler_output           # [CLS] token representation

```text

### Feature Extraction

```python
def extract_bert_features(texts, model, tokenizer, layer=-1):
    """Extract BERT features from a specific layer."""
    
    features = []
    model.eval()
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt',
                          padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract from specific layer
            layer_output = outputs.hidden_states[layer]
            
            # Use [CLS] token or mean pooling
            cls_embedding = layer_output[:, 0, :]  # [CLS] token
            mean_embedding = layer_output.mean(dim=1)  # Mean pooling
            
            features.append(cls_embedding.numpy())
    
    return np.array(features)

```text

### Fine-tuning Example

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

## Load model for classification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3  # Multi-class classification
)

## Training arguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    learning_rate=2e-5
)

## Create trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

## Fine-tune

trainer.train()

```text

## Limitations and Considerations

### Computational Requirements

**Memory Usage**:

- BERT-Base: ~1.3GB GPU memory for inference
- BERT-Large: ~5GB GPU memory for inference
- Fine-tuning requires 2-3x more memory

**Inference Speed**:

- Slower than simpler models due to transformer complexity
- Full attention computation scales quadratically with sequence length

### Sequence Length Constraints

```text
Maximum sequence length: 512 tokens

- Longer texts need truncation or segmentation
- Important information might be lost at boundaries

```text

### Generation Limitations

BERT cannot generate text naturally:

- Designed for understanding, not generation
- Bidirectional context makes autoregressive generation impossible
- Need specialized approaches for text generation tasks

## Comparison with Other Models

### BERT vs GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal (unidirectional) |
| Primary Use | Understanding | Generation |
| Training | MLM + NSP | Next token prediction |
| Context | Full sequence | Left-to-right only |

### BERT vs T5

| Aspect | BERT | T5 |
|--------|------|-----|
| Architecture | Encoder-only | Encoder-decoder |
| Task Format | Task-specific heads | Text-to-text |
| Training | MLM + NSP | Span corruption |
| Versatility | Fine-tuning required | Unified format |

## Impact and Legacy

### Research Impact

BERT's publication led to significant advances:

- **Bidirectional Pre-training**: Became standard practice
- **Transfer Learning**: Demonstrated effectiveness in NLP
- **Benchmark Improvements**: Dramatic performance gains across tasks
- **Model Architecture**: Influenced numerous subsequent models

### Industry Adoption

**Google Search**: BERT improved search query understanding
**Question Answering**: Enabled more accurate QA systems  
**Chatbots**: Better intent recognition and response relevance
**Document Processing**: Enhanced information extraction

### Subsequent Developments

BERT's success spawned numerous improvements:

- **Efficiency**: DistilBERT, MobileBERT for deployment
- **Scale**: RoBERTa, DeBERTa for better performance  
- **Specialization**: BioBERT, FinBERT for domain adaptation
- **Multilingual**: mBERT, XLM-R for cross-lingual understanding

## Future Directions

### Efficiency Improvements

**Model Compression**: Pruning, quantization, knowledge distillation
**Architecture Changes**: More efficient attention mechanisms
**Hardware Optimization**: TPU/GPU-specific optimizations

### Enhanced Understanding

**Longer Context**: Handling documents beyond 512 tokens
**Better Reasoning**: Improved logical and mathematical understanding  
**Multimodal**: Combining text with images, audio, and other modalities

### Domain Adaptation

**Specialized Models**: Domain-specific pre-training
**Few-Shot Learning**: Better performance with limited task data
**Continual Learning**: Updating knowledge without full retraining

BERT fundamentally transformed natural language processing by demonstrating that bidirectional context understanding
could dramatically improve performance on language tasks. Its encoder-only architecture and innovative training approach
established new standards for language model pre-training and fine-tuning, influencing virtually all subsequent
developments in NLP while remaining one of the most widely used models for language understanding applications.
