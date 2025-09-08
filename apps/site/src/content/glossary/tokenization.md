---
title: Tokenization
aliases: ["tokenization", "tokenizer", "text tokenization", "subword tokenization"]
summary: Tokenization is the process of breaking down text into smaller units called tokens (words, subwords, or characters) that can be processed by machine learning models. Modern tokenization methods like Byte-Pair Encoding (BPE) and SentencePiece enable language models to handle diverse vocabularies efficiently while managing out-of-vocabulary words and supporting multilingual text processing.
tags: ["nlp", "fundamentals", "data", "machine-learning"]
related: ["embedding", "transformer", "bert", "gpt", "llm"]
category: "nlp"
difficulty: "beginner"
updated: "2025-01-15"
sources:
  - source_url: "https://arxiv.org/abs/1508.07909"
    source_title: "Neural Machine Translation of Rare Words with Subword Units"
    license: "cc-by"
    author: "Rico Sennrich et al."
  - source_url: "https://arxiv.org/abs/1804.10959"
    source_title: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
    license: "cc-by"
    author: "Taku Kudo, John Richardson"
---

## Overview

Tokenization is the fundamental preprocessing step that converts raw text into a structured format that machine learning models can understand. By breaking text into discrete units (tokens), tokenization bridges the gap between human language and computational processing, enabling everything from simple word counting to complex language model training.

## Core Concepts

### What Are Tokens?

Tokens are the basic units of text processing that represent meaningful segments:

```python
# Different tokenization approaches for the same sentence
text = "The cat's running quickly!"

# Word tokenization
word_tokens = ["The", "cat's", "running", "quickly", "!"]

# Subword tokenization (BPE-style)
subword_tokens = ["The", "cat", "'s", "run", "ning", "quick", "ly", "!"]

# Character tokenization  
char_tokens = ["T", "h", "e", " ", "c", "a", "t", "'", "s", " ", "r", "u", "n", "n", "i", "n", "g", " ", "q", "u", "i", "c", "k", "l", "y", "!"]
```

### Token IDs and Vocabularies

Tokenizers map tokens to numerical IDs for model processing:

```python
# Example vocabulary mapping
vocabulary = {
    "<pad>": 0,    # Padding token
    "<unk>": 1,    # Unknown token  
    "<bos>": 2,    # Beginning of sequence
    "<eos>": 3,    # End of sequence
    "the": 4,
    "cat": 5,
    "run": 6,
    "ning": 7,
    # ... more tokens
}

# Text to IDs conversion
text = "the cat running"
tokens = ["the", "cat", "run", "ning"]
token_ids = [4, 5, 6, 7]
```

## Evolution of Tokenization Methods

### 1. Word-Level Tokenization

The simplest approach splits text by whitespace and punctuation:

```python
import re

def simple_word_tokenize(text):
    """Basic word tokenization using regex"""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

text = "Hello, world! How are you?"
tokens = simple_word_tokenize(text)
# Result: ["hello", "world", "how", "are", "you"]
```

**Advantages:**
- Simple and interpretable
- Preserves semantic word boundaries
- Works well for morphologically simple languages

**Disadvantages:**
- Large vocabulary sizes (millions of unique words)
- Out-of-vocabulary (OOV) problems
- Poor handling of morphologically rich languages
- Inconsistent treatment of variations (run/running/runs)

### 2. Character-Level Tokenization

Treats each character as a separate token:

```python
def char_tokenize(text):
    """Character-level tokenization"""
    return list(text)

text = "hello"
tokens = char_tokenize(text)
# Result: ["h", "e", "l", "l", "o"]
```

**Advantages:**
- Small, fixed vocabulary size
- No out-of-vocabulary issues
- Can handle any text in the character set

**Disadvantages:**
- Very long sequences
- Harder to capture semantic meaning
- Increased computational requirements

### 3. Subword Tokenization

Modern approach that balances vocabulary size and semantic preservation:

```python
# Example subword tokenization
text = "unhappiness"

# Possible subword breakdown:
subwords = ["un", "happi", "ness"]  # Meaningful morphological units

# Or BPE-style:
bpe_tokens = ["unha", "ppi", "ness"]  # Data-driven splits
```

## Modern Tokenization Algorithms

### Byte-Pair Encoding (BPE)

BPE iteratively merges the most frequent character pairs to build a vocabulary:

```python
def learn_bpe(corpus, num_merges):
    """Simplified BPE training algorithm"""
    
    # Initialize with character-level vocabulary
    vocab = set()
    word_freqs = {}
    
    # Count word frequencies and initialize vocab
    for word in corpus:
        word_freqs[word] = word_freqs.get(word, 0) + 1
        for char in word:
            vocab.add(char)
    
    # Learn merge rules
    merges = []
    for _ in range(num_merges):
        # Count all adjacent pairs
        pairs = {}
        for word, freq in word_freqs.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        
        # Find most frequent pair
        if not pairs:
            break
            
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        # Merge the best pair in all words
        new_word_freqs = {}
        for word in word_freqs:
            new_word = word.replace(
                best_pair[0] + best_pair[1], 
                best_pair[0] + best_pair[1]  # Merged token
            )
            new_word_freqs[new_word] = word_freqs[word]
        word_freqs = new_word_freqs
        
        # Add merged token to vocabulary
        vocab.add(best_pair[0] + best_pair[1])
    
    return vocab, merges

# Usage example
corpus = ["hello", "world", "help", "held", "world", "hello"]  
vocab, merges = learn_bpe(corpus, num_merges=10)
```

#### BPE in Practice

```python
# Using Hugging Face tokenizers library
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# Create BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train on corpus
trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<unk>"])
files = ["training_corpus.txt"]
tokenizer.train(files, trainer)

# Tokenize text
output = tokenizer.encode("Hello world!")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")
```

### WordPiece

Developed by Google, used in BERT and other models:

```python
# WordPiece differs from BPE in merge criteria
# Instead of most frequent pairs, it maximizes likelihood increase

def wordpiece_score(pair_freq, left_freq, right_freq):
    """WordPiece scoring function"""
    return pair_freq / (left_freq * right_freq)

# WordPiece also uses special prefix (##) for continuation tokens
# Example: "playing" → ["play", "##ing"]
```

Key differences from BPE:
- Uses likelihood-based merge criteria instead of frequency
- Uses ## prefix for subword continuation
- More linguistically motivated merges

### SentencePiece

Language-agnostic tokenization that works directly on raw text:

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=8000,
    model_type='bpe',  # or 'unigram'
    character_coverage=0.9995
)

# Load and use trained model
sp = smp.SentencePieceProcessor(model_file='tokenizer.model')

# Tokenize text
tokens = sp.encode('This is a test sentence.', out_type=str)
print(tokens)  # ['▁This', '▁is', '▁a', '▁test', '▁sent', 'ence', '.']

# Convert to IDs
ids = sp.encode('This is a test sentence.', out_type=int)
print(ids)  # [46, 25, 9, 688, 1370, 4005, 7]
```

Benefits of SentencePiece:
- Language independent (no need for word segmentation)
- Handles whitespace as regular characters
- Reversible tokenization
- Built-in handling of unknown characters

### Unigram Language Model

Alternative to BPE used in some SentencePiece models:

```python
# Unigram starts with large vocabulary and prunes iteratively
# Keeps subwords that minimize loss when removed

def unigram_tokenize(text, model):
    """Tokenize using unigram language model"""
    # Uses dynamic programming to find optimal segmentation
    # Maximizes: P(token_1) * P(token_2) * ... * P(token_n)
    
    # This is a simplified version
    best_segmentation = []
    # ... complex dynamic programming implementation
    return best_segmentation
```

## Practical Implementation

### Using Pre-trained Tokenizers

#### OpenAI GPT Tokenizer

```python
import tiktoken

# GPT-3.5/GPT-4 tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = "Hello, world! This is a test."
tokens = encoding.encode(text)
print(f"Tokens: {tokens}")
print(f"Decoded: {encoding.decode(tokens)}")

# Count tokens (important for API limits)
print(f"Token count: {len(tokens)}")
```

#### Hugging Face Transformers

```python
from transformers import AutoTokenizer

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

text = "The quick brown fox jumps over the lazy dog."

# Basic tokenization
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Convert to IDs
input_ids = tokenizer.encode(text, add_special_tokens=True)
print(f"Input IDs: {input_ids}")

# Full preprocessing for model input
inputs = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)
print(inputs)
```

### Custom Tokenizer Training

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

def train_custom_tokenizer(files, vocab_size=10000):
    """Train a custom BPE tokenizer"""
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Add normalization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    
    # Pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Decoder
    tokenizer.decoder = decoders.BPEDecoder()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]
    )
    
    # Train on files
    tokenizer.train(files, trainer)
    
    return tokenizer

# Usage
tokenizer = train_custom_tokenizer(["corpus.txt"], vocab_size=5000)
tokenizer.save("my_tokenizer.json")
```

## Handling Special Cases

### Multilingual Tokenization

```python
# Challenges with different scripts and languages
texts = [
    "Hello world",                    # English
    "Bonjour le monde",              # French
    "こんにちは世界",                   # Japanese  
    "مرحبا بالعالم",                  # Arabic
    "Здравствуй мир"                 # Russian
]

# SentencePiece handles multilingual text well
import sentencepiece as spm

# Train multilingual model
spm.SentencePieceTrainer.train(
    input='multilingual_corpus.txt',
    model_prefix='multilingual_tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,  # Important for multilingual
    model_type='unigram'
)
```

### Code Tokenization

```python
# Special considerations for programming languages
code_text = """
def hello_world():
    print("Hello, World!")
    return True
"""

# Code-specific tokenizers preserve meaningful units
# Example: keeping function names, operators, keywords intact
code_tokens = [
    "def", "hello_world", "(", ")", ":",
    "print", "(", '"Hello, World!"', ")",
    "return", "True"
]
```

### Out-of-Vocabulary Handling

```python
class RobustTokenizer:
    def __init__(self, tokenizer, unk_token="<unk>"):
        self.tokenizer = tokenizer
        self.unk_token = unk_token
        
    def tokenize_with_fallback(self, text):
        """Tokenize with graceful OOV handling"""
        try:
            return self.tokenizer.encode(text)
        except Exception:
            # Fallback to character-level for unknown text
            return self.character_fallback(text)
    
    def character_fallback(self, text):
        """Character-level tokenization for OOV text"""
        return [self.unk_token if c not in self.vocab else c for c in text]
```

## Performance Considerations

### Tokenization Speed

```python
import time
from typing import List

def benchmark_tokenizers(texts: List[str], tokenizers: dict):
    """Compare tokenization speed across different methods"""
    
    results = {}
    
    for name, tokenizer in tokenizers.items():
        start_time = time.time()
        
        for text in texts:
            tokens = tokenizer.encode(text)
            
        end_time = time.time()
        
        results[name] = {
            'time': end_time - start_time,
            'speed': len(texts) / (end_time - start_time)
        }
    
    return results

# Example benchmark
texts = ["Sample text"] * 10000
tokenizers = {
    'sentencepiece': sp_tokenizer,
    'bert_tokenizer': bert_tokenizer,
    'gpt_tokenizer': gpt_tokenizer
}

results = benchmark_tokenizers(texts, tokenizers)
```

### Memory Efficiency

```python
# Vocabulary size affects memory usage
def estimate_tokenizer_memory(vocab_size, embedding_dim):
    """Estimate memory usage for token embeddings"""
    
    # Each token needs an embedding vector
    embedding_memory = vocab_size * embedding_dim * 4  # 4 bytes per float32
    
    # Vocabulary lookup structures
    vocab_memory = vocab_size * 50  # Rough estimate for strings
    
    total_mb = (embedding_memory + vocab_memory) / (1024 * 1024)
    
    return {
        'embedding_memory_mb': embedding_memory / (1024 * 1024),
        'vocab_memory_mb': vocab_memory / (1024 * 1024),
        'total_mb': total_mb
    }

# Example: GPT-3 tokenizer
gpt3_memory = estimate_tokenizer_memory(vocab_size=50257, embedding_dim=12288)
print(f"GPT-3 tokenizer memory usage: {gpt3_memory['total_mb']:.2f} MB")
```

## Evaluation Metrics

### Compression Ratio

```python
def calculate_compression_ratio(text, tokenizer):
    """Measure how efficiently tokenizer compresses text"""
    
    # Original character count
    char_count = len(text)
    
    # Token count after tokenization
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    # Compression ratio (higher is better compression)
    ratio = char_count / token_count
    
    return {
        'char_count': char_count,
        'token_count': token_count,
        'compression_ratio': ratio
    }

# Compare different tokenizers
text = "The quick brown fox jumps over the lazy dog repeatedly."
ratios = {}

for name, tokenizer in tokenizers.items():
    ratios[name] = calculate_compression_ratio(text, tokenizer)
    print(f"{name}: {ratios[name]['compression_ratio']:.2f}")
```

### Fertility Score

```python
def calculate_fertility(word, tokenizer):
    """Calculate average number of tokens per word"""
    
    words = word.split()
    total_tokens = 0
    
    for word in words:
        tokens = tokenizer.encode(word)
        total_tokens += len(tokens)
    
    fertility = total_tokens / len(words)
    return fertility

# Lower fertility is generally better
fertility_scores = {}
test_text = "internationalization specialization"

for name, tokenizer in tokenizers.items():
    fertility_scores[name] = calculate_fertility(test_text, tokenizer)
```

## Common Pitfalls and Solutions

### 1. Inconsistent Preprocessing

```python
# Problem: Different preprocessing for training vs inference
def consistent_preprocessing(text):
    """Ensure consistent text preprocessing"""
    
    # Always apply same steps in same order
    text = text.lower()                    # Case normalization
    text = re.sub(r'\s+', ' ', text)      # Whitespace normalization  
    text = text.strip()                    # Remove leading/trailing spaces
    text = unicodedata.normalize('NFD', text)  # Unicode normalization
    
    return text

# Apply before tokenization
preprocessed_text = consistent_preprocessing(raw_text)
tokens = tokenizer.encode(preprocessed_text)
```

### 2. Special Token Handling

```python
class SafeTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_tokens = {
            'pad': tokenizer.pad_token,
            'unk': tokenizer.unk_token,
            'bos': tokenizer.bos_token,
            'eos': tokenizer.eos_token
        }
    
    def encode_safe(self, text, add_special_tokens=True, max_length=512):
        """Encode with proper special token handling"""
        
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        
        return tokens
    
    def decode_safe(self, token_ids, skip_special_tokens=True):
        """Decode with proper special token handling"""
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
```

### 3. Subword Boundary Issues

```python
def fix_subword_boundaries(tokens):
    """Reconstruct proper word boundaries from subword tokens"""
    
    words = []
    current_word = ""
    
    for token in tokens:
        if token.startswith("##"):  # WordPiece continuation
            current_word += token[2:]
        elif token.startswith("▁"):  # SentencePiece word start
            if current_word:
                words.append(current_word)
            current_word = token[1:]
        else:
            if current_word:
                words.append(current_word)
            current_word = token
    
    if current_word:
        words.append(current_word)
    
    return words

# Example usage
wordpiece_tokens = ["play", "##ing", "foot", "##ball"]
words = fix_subword_boundaries(wordpiece_tokens)
print(words)  # ["playing", "football"]
```

## Advanced Applications

### Adaptive Tokenization

```python
class AdaptiveTokenizer:
    """Tokenizer that adapts vocabulary based on domain"""
    
    def __init__(self, base_tokenizer, adaptation_threshold=0.1):
        self.base_tokenizer = base_tokenizer
        self.domain_vocab = {}
        self.adaptation_threshold = adaptation_threshold
    
    def adapt_to_domain(self, domain_corpus):
        """Learn domain-specific tokens"""
        
        # Find frequent domain-specific terms
        domain_tokens = {}
        for text in domain_corpus:
            tokens = self.base_tokenizer.encode(text)
            for token in tokens:
                domain_tokens[token] = domain_tokens.get(token, 0) + 1
        
        # Add high-frequency domain tokens
        total_tokens = sum(domain_tokens.values())
        for token, count in domain_tokens.items():
            if count / total_tokens > self.adaptation_threshold:
                self.domain_vocab[token] = count
    
    def tokenize_adapted(self, text):
        """Tokenize with domain adaptation"""
        base_tokens = self.base_tokenizer.encode(text)
        
        # Apply domain-specific merging rules
        adapted_tokens = self.apply_domain_merging(base_tokens)
        return adapted_tokens
```

### Privacy-Preserving Tokenization

```python
import hashlib

class PrivateTokenizer:
    """Tokenizer with privacy protection for sensitive terms"""
    
    def __init__(self, base_tokenizer, sensitive_patterns=None):
        self.base_tokenizer = base_tokenizer
        self.sensitive_patterns = sensitive_patterns or []
        self.hash_map = {}
    
    def hash_sensitive_tokens(self, token):
        """Hash sensitive tokens while preserving structure"""
        
        for pattern in self.sensitive_patterns:
            if re.match(pattern, token):
                if token not in self.hash_map:
                    # Create consistent hash
                    hash_value = hashlib.md5(token.encode()).hexdigest()[:8]
                    self.hash_map[token] = f"<HASH_{hash_value}>"
                return self.hash_map[token]
        
        return token
    
    def tokenize_private(self, text):
        """Tokenize with privacy protection"""
        
        tokens = self.base_tokenizer.encode(text)
        private_tokens = [self.hash_sensitive_tokens(t) for t in tokens]
        return private_tokens

# Example: Protect email addresses and phone numbers
sensitive_patterns = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    r'\b\d{3}-\d{3}-\d{4}\b'  # Phone number
]

private_tokenizer = PrivateTokenizer(base_tokenizer, sensitive_patterns)
```

## Future Trends

### Byte-Level Tokenization

```python
# GPT-2 style byte-level BPE
# Works directly on UTF-8 bytes instead of characters

class ByteLevelBPE:
    def __init__(self):
        # Map bytes to printable characters
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def bytes_to_unicode(self):
        """Map UTF-8 bytes to Unicode strings"""
        # Implementation details for mapping bytes to characters
        # that can be processed by standard text algorithms
        pass
    
    def encode(self, text):
        """Encode text using byte-level BPE"""
        # Convert to bytes, then to unicode chars, then apply BPE
        text_bytes = text.encode('utf-8')
        text_unicode = ''.join([self.byte_encoder[b] for b in text_bytes])
        # Apply standard BPE on the unicode representation
        return self.bpe_encode(text_unicode)
```

### Neural Tokenization

```python
# Learned tokenization using neural networks
# Tokenization as a differentiable operation

class NeuralTokenizer(nn.Module):
    def __init__(self, vocab_size, max_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Learnable segmentation network
        self.segment_network = nn.LSTM(256, 128, batch_first=True)
        self.boundary_predictor = nn.Linear(128, 1)
    
    def forward(self, char_embeddings):
        """Predict token boundaries"""
        lstm_out, _ = self.segment_network(char_embeddings)
        boundaries = torch.sigmoid(self.boundary_predictor(lstm_out))
        
        # Use boundaries to create differentiable tokenization
        return self.create_tokens(char_embeddings, boundaries)
```

### Dynamic Vocabulary

```python
class DynamicVocabularyTokenizer:
    """Tokenizer with vocabulary that adapts during inference"""
    
    def __init__(self, base_vocab_size=1000, expansion_rate=0.1):
        self.base_vocab_size = base_vocab_size
        self.expansion_rate = expansion_rate
        self.dynamic_vocab = {}
        self.token_frequencies = {}
    
    def update_vocabulary(self, new_tokens):
        """Update vocabulary based on new tokens encountered"""
        
        for token in new_tokens:
            self.token_frequencies[token] = self.token_frequencies.get(token, 0) + 1
            
            # Add to vocabulary if frequency exceeds threshold
            if (self.token_frequencies[token] > 10 and 
                token not in self.dynamic_vocab):
                self.dynamic_vocab[token] = len(self.dynamic_vocab)
    
    def tokenize_adaptive(self, text):
        """Tokenize with adaptive vocabulary updates"""
        
        # Standard tokenization
        tokens = self.base_tokenize(text)
        
        # Update vocabulary based on new tokens
        self.update_vocabulary(tokens)
        
        # Re-tokenize with updated vocabulary
        return self.tokenize_with_dynamic_vocab(text)
```

Tokenization remains a critical component in the NLP pipeline, and ongoing research continues to improve how we break down and represent human language for machine learning models. As language models become more sophisticated and multilingual, tokenization methods must evolve to handle the increasing complexity and diversity of textual data while maintaining efficiency and semantic preservation.