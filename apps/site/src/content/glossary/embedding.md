---
aliases:
- embeddings
- vector embedding
- word embedding
- neural embedding
category: fundamentals
difficulty: intermediate
related:
- transformer
- attention-mechanism
- bert
- gpt
- neural-network
sources:
- author: Tomas Mikolov et al.
  license: cc-by
  source_title: Efficient Estimation of Word Representations in Vector Space
  source_url: https://arxiv.org/abs/1301.3781
- author: Tomas Mikolov et al.
  license: cc-by
  source_title: Distributed Representations of Words and Phrases and their Compositionality
  source_url: https://arxiv.org/abs/1310.4546
summary: An embedding is a dense vector representation that captures semantic meaning
  of discrete objects like words, sentences, or images in a continuous numerical space.
  Embeddings enable machine learning models to process symbolic data by mapping similar
  concepts to nearby points in high-dimensional vector space, forming the foundation
  for modern NLP, recommendation systems, and similarity search applications.
tags:
- nlp
- machine-learning
- deep-learning
- fundamentals
- data
title: Embedding
updated: '2025-01-15'
---

## Overview

An embedding is a learned representation that maps discrete symbolic information (like words, sentences, users, or
products) into dense, continuous vector spaces where semantic similarity corresponds to geometric proximity. This
transformation allows machine learning models to work with symbolic data by converting it into numerical form that
preserves meaningful relationships and enables mathematical operations.

## Core Concepts

### From Symbolic to Numerical

Traditional approaches used sparse, one-hot representations:

```python

# One-hot encoding (sparse, high-dimensional)

vocabulary = ["cat", "dog", "bird", "fish"]

cat = [1, 0, 0, 0]      # 4-dimensional, mostly zeros
dog = [0, 1, 0, 0]      # No semantic relationship captured
bird = [0, 0, 1, 0]     # All words equally distant
fish = [0, 0, 0, 1]

```text
Embeddings use dense, low-dimensional vectors:

```python

## Dense embeddings (semantic relationships preserved)

cat = [0.2, 0.8, 0.3, 0.1]     # 4-dimensional, all values meaningful
dog = [0.3, 0.7, 0.4, 0.0]     # Similar to cat (both pets)
bird = [0.1, 0.2, 0.9, 0.3]    # Different from mammals
fish = [0.0, 0.1, 0.8, 0.6]    # Similar to bird (both non-mammals)

```text

### Semantic Space Properties

Well-trained embeddings exhibit remarkable properties:

#### Semantic Similarity

```python

## Words with similar meanings have similar vectors

cosine_similarity("king", "queen") = 0.72
cosine_similarity("cat", "dog") = 0.81
cosine_similarity("car", "bicycle") = 0.54
cosine_similarity("king", "apple") = 0.02

```text

#### Analogical Relationships

```python

## Vector arithmetic captures relationships

vector("king") - vector("man") ≈ vector("queen") - vector("woman")
vector("Paris") - vector("France") ≈ vector("Tokyo") - vector("Japan")
vector("walked") - vector("walk") ≈ vector("ran") - vector("run")

```text

#### Clustering

```python

## Related concepts cluster together

animals = {"cat", "dog", "bird", "fish"}
colors = {"red", "blue", "green", "yellow"}
countries = {"France", "Japan", "Brazil", "Canada"}

## Each group forms distinct clusters in embedding space

```text

## Types of Embeddings

### Word Embeddings

#### Word2Vec (2013)

Two training approaches for learning word representations:

**Skip-gram**: Predict context words from target word

```python

## Skip-gram training example

target_word = "cat"
context_window = ["the", "cat", "sat", "on", "mat"]

## Model learns: cat → {the, sat, on, mat}

## Objective: maximize P(context | target)

```text
**Continuous Bag of Words (CBOW)**: Predict target from context

```python  

## CBOW training example

context_words = ["the", "sat", "on", "mat"]
target_word = "cat"

## Model learns: {the, sat, on, mat} → cat  

## Objective: maximize P(target | context)

```text

#### GloVe (Global Vectors)

Combines matrix factorization with local context windows:

```python

## GloVe objective function

J = Σ f(X_ij) * (w_i^T * w_j + b_i + b_j - log(X_ij))²

## Where

## X_ij = co-occurrence count of word i with word j

## w_i, w_j = word vectors for words i and j

## b_i, b_j = bias terms

## f(x) = weighting function to handle frequent/rare words

```text

#### FastText

Extends Word2Vec to handle subword information:

```python

## FastText represents words as sum of subword vectors

word = "playing"
subwords = ["<pl", "pla", "lay", "ayi", "yin", "ing", "ng>"]

## Final embedding = sum of subword embeddings

embedding("playing") = Σ embedding(subword) for subword in subwords

```text
Benefits:

- Handles out-of-vocabulary words
- Captures morphological information
- Better for languages with rich morphology

### Contextual Embeddings

Traditional word embeddings assign fixed vectors to words, but contextual embeddings create different representations
based on context:

#### ELMo (Embeddings from Language Models)

Uses bidirectional LSTM to create context-aware embeddings:

```python

## ELMo creates different embeddings for same word

sentence1 = "I went to the bank to deposit money"
sentence2 = "I sat by the river bank"

## "bank" gets different embeddings based on context

bank_financial = elmo_embedding("bank", sentence1)  
bank_river = elmo_embedding("bank", sentence2)

cosine_similarity(bank_financial, bank_river) = 0.23  # Different meanings

```text

#### BERT Embeddings

Transformer-based contextual embeddings:

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_word_embedding(sentence, word, model, tokenizer):
    # Tokenize and get word position
    tokens = tokenizer.tokenize(sentence)
    word_idx = tokens.index(word)
    
    # Get BERT embeddings
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    
    # Extract embedding for specific word
    word_embedding = outputs.last_hidden_state[0, word_idx, :]
    return word_embedding

```text

### Sentence and Document Embeddings

#### Sentence-BERT (SBERT)

Creates meaningful sentence-level embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",  
    "Dogs are great pets",
    "Python is a programming language"
]

embeddings = model.encode(sentences)

## Similar sentences have similar embeddings  

similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])  # High
similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])  # Medium
similarity_1_4 = cosine_similarity(embeddings[0], embeddings[3])  # Low

```text

#### Doc2Vec

Extends Word2Vec to document-level embeddings:

```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

## Prepare documents

documents = [
    TaggedDocument(["neural", "networks", "deep", "learning"], ["doc1"]),
    TaggedDocument(["machine", "learning", "algorithms"], ["doc2"]),
    TaggedDocument(["cooking", "recipes", "food"], ["doc3"])
]

## Train Doc2Vec model

model = Doc2Vec(documents, vector_size=100, epochs=40)

## Get document embeddings

doc1_vector = model.docvecs["doc1"]
doc2_vector = model.docvecs["doc2"]
doc3_vector = model.docvecs["doc3"]

```text

### Multimodal Embeddings

#### CLIP (Contrastive Language-Image Pre-training)

Creates shared embedding space for text and images:

```python
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

## Process image and text

image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a cat", "a dog", "a car"]).to(device)

## Get embeddings in shared space

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Calculate similarities
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    print(f"Image-text similarities: {similarities}")

```text

## Training Methods

### Contrastive Learning

Many modern embeddings use contrastive learning objectives:

```python

## Contrastive loss function

def contrastive_loss(anchor, positive, negative, margin=1.0):
    """
    anchor: reference embedding
    positive: similar item embedding  
    negative: dissimilar item embedding
    """
    pos_distance = torch.norm(anchor - positive)
    neg_distance = torch.norm(anchor - negative)
    
    loss = torch.max(
        torch.tensor(0.0),
        pos_distance - neg_distance + margin
    )
    return loss

```text

### Triplet Loss

Used for learning embedding spaces with relative similarity:

```python
def triplet_loss(anchor, positive, negative, margin=1.0):
    """Learn embeddings where positive is closer than negative to anchor"""
    pos_dist = torch.norm(anchor - positive, p=2)
    neg_dist = torch.norm(anchor - negative, p=2)
    
    loss = torch.max(
        torch.tensor(0.0),
        pos_dist - neg_dist + margin
    )
    return loss

```text

### Self-Supervised Learning

Modern approaches use self-supervised objectives:

```python

## Masked language modeling (BERT-style)

def masked_lm_loss(embeddings, masked_tokens, predictions):
    """Predict masked tokens from context embeddings"""
    loss = cross_entropy(predictions[masked_positions], masked_tokens)
    return loss

## Next sentence prediction

def next_sentence_loss(sentence_a_emb, sentence_b_emb, is_next):
    """Predict if sentence B follows sentence A"""
    combined = torch.cat([sentence_a_emb, sentence_b_emb], dim=-1)
    prediction = classifier(combined)
    loss = cross_entropy(prediction, is_next)
    return loss

```text

## Applications

### Similarity Search and Retrieval

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingSearch:
    def __init__(self, embeddings, items):
        self.embeddings = np.array(embeddings)
        self.items = items
        
    def search(self, query_embedding, top_k=5):
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.items[i], similarities[i])
            for i in top_indices
        ]
        return results

## Usage example

search_engine = EmbeddingSearch(product_embeddings, product_names)
query = get_embedding("smartphone with good camera")
similar_products = search_engine.search(query, top_k=10)

```text

### Recommendation Systems

```python
class EmbeddingRecommender:
    def __init__(self, user_embeddings, item_embeddings):
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        
    def recommend(self, user_id, exclude_seen=None, top_k=10):
        user_emb = self.user_embeddings[user_id]
        
        # Calculate user-item similarities
        similarities = cosine_similarity(
            user_emb.reshape(1, -1),
            self.item_embeddings
        )[0]
        
        # Exclude already seen items
        if exclude_seen:
            similarities[exclude_seen] = -1
            
        # Get top recommendations
        top_items = np.argsort(similarities)[::-1][:top_k]
        return top_items

```text

### Clustering and Classification

```python
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

## Clustering using embeddings

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(word_embeddings)

## Classification using embeddings as features

classifier = LogisticRegression()
classifier.fit(sentence_embeddings, labels)
predictions = classifier.predict(new_sentence_embeddings)

```text

### Question Answering

```python
class EmbeddingQA:
    def __init__(self, passages, passage_embeddings):
        self.passages = passages
        self.passage_embeddings = passage_embeddings
        
    def answer_question(self, question, top_k=3):
        # Get question embedding
        question_emb = get_embedding(question)
        
        # Find most relevant passages
        similarities = cosine_similarity(
            question_emb.reshape(1, -1),
            self.passage_embeddings
        )[0]
        
        top_passages = np.argsort(similarities)[::-1][:top_k]
        
        # Return relevant passages for further processing
        return [self.passages[i] for i in top_passages]

```text

## Quality Assessment

### Intrinsic Evaluation

#### Word Similarity Tasks

```python

## Evaluate on human-annotated similarity datasets

def evaluate_similarity(embeddings, similarity_dataset):
    human_scores = []
    model_scores = []
    
    for word1, word2, human_score in similarity_dataset:
        emb1 = embeddings[word1]
        emb2 = embeddings[word2]
        model_score = cosine_similarity(emb1, emb2)
        
        human_scores.append(human_score)
        model_scores.append(model_score)
    
    # Calculate correlation
    correlation = scipy.stats.spearmanr(human_scores, model_scores)
    return correlation

```text

#### Analogy Tasks

```python
def evaluate_analogies(embeddings, analogy_dataset):
    correct = 0
    total = 0
    
    for a, b, c, expected_d in analogy_dataset:
        # a : b :: c : ?
        # vector(d) ≈ vector(b) - vector(a) + vector(c)
        target_vector = embeddings[b] - embeddings[a] + embeddings[c]
        
        # Find closest word to target_vector
        similarities = {}
        for word in embeddings:
            if word not in [a, b, c]:  # Exclude input words
                sim = cosine_similarity(target_vector, embeddings[word])
                similarities[word] = sim
        
        predicted_d = max(similarities, key=similarities.get)
        
        if predicted_d == expected_d:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

```text

### Extrinsic Evaluation

#### Downstream Task Performance

```python

## Evaluate embeddings on classification task

def evaluate_on_task(embeddings, X_text, y_labels):
    # Convert text to embeddings
    X_embeddings = np.array([embeddings[text] for text in X_text])
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y_labels, test_size=0.2
    )
    
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    # Evaluate
    accuracy = classifier.score(X_test, y_test)
    return accuracy

```text

## Technical Considerations

### Dimensionality Selection

```python

## Common embedding dimensions

embedding_dims = {
    "word2vec": [100, 200, 300],      # Typical range
    "glove": [50, 100, 200, 300],     # Pre-trained available
    "bert": [768, 1024],              # Fixed architecture
    "sentence_transformers": [384, 768], # Model-dependent
    "openai_ada": [1536],             # API-based embedding
}

## Trade-offs

## Smaller dimensions: Faster computation, less memory, potential information loss

## Larger dimensions: More expressive, slower computation, higher memory usage

```text

### Normalization

```python
def normalize_embeddings(embeddings):
    """L2 normalize embeddings for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)  # Avoid division by zero
    return normalized

## Benefits of normalization

## 1. Cosine similarity becomes dot product (faster computation)

## 2. All embeddings have unit length (consistent scale)

## 3. Focuses on direction rather than magnitude

```text

### Handling Out-of-Vocabulary (OOV) Words

```python
class EmbeddingHandler:
    def __init__(self, embeddings, unknown_token="<UNK>"):
        self.embeddings = embeddings
        self.unknown_token = unknown_token
        self.unk_vector = embeddings.get(unknown_token, self._create_unk_vector())
    
    def _create_unk_vector(self):
        # Create random vector or average of all embeddings
        if len(self.embeddings) > 0:
            all_vectors = np.array(list(self.embeddings.values()))
            return np.mean(all_vectors, axis=0)
        else:
            return np.random.normal(0, 0.1, size=300)  # Random initialization
    
    def get_embedding(self, word):
        return self.embeddings.get(word, self.unk_vector)

```text

## Modern Developments

### Large-Scale Embeddings

#### OpenAI Ada Embeddings

```python
import openai

## High-quality embeddings via API

client = openai.OpenAI()

def get_openai_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

## Properties

## - 1536 dimensions

## - Trained on diverse, high-quality data

## - Strong performance across tasks

## - Cost: ~$0.0001 per 1K tokens

```text

#### Cohere Embeddings

```python
import cohere

co = cohere.Client("your-api-key")

def get_cohere_embedding(texts):
    response = co.embed(
        texts=texts,
        model='embed-english-v2.0'
    )
    return response.embeddings

```text

### Specialized Embeddings

#### Code Embeddings

```python

## Microsoft CodeBERT for code similarity

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def get_code_embedding(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

```text

#### Scientific Paper Embeddings

```python

## SciBERT for scientific text

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert-scivocab-uncased")
model = AutoModel.from_pretrained("allenai/scibert-scivocab-uncased")

```text

### Multilingual Embeddings

```python

## Multilingual Universal Sentence Encoder

import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

texts = [
    "Hello world",           # English
    "Bonjour le monde",      # French  
    "Hola mundo",            # Spanish
    "こんにちは世界"           # Japanese
]

embeddings = embed(texts)

## All texts mapped to same semantic space regardless of language

```text

## Limitations and Challenges

### Bias and Fairness

Embeddings can perpetuate societal biases:

```python

## Example of gender bias in word embeddings

def check_bias(embeddings):
    # Problematic associations often found:
    programmer_vec = embeddings["programmer"]
    nurse_vec = embeddings["nurse"]
    
    male_vec = embeddings["he"] - embeddings["she"]
    
    # Programmer might be closer to "male" direction
    programmer_bias = cosine_similarity(programmer_vec, male_vec)
    nurse_bias = cosine_similarity(nurse_vec, male_vec)
    
    print(f"Programmer-male bias: {programmer_bias}")
    print(f"Nurse-male bias: {nurse_bias}")

```text

### Computational Scalability

```python

## Challenges with large-scale similarity search

def naive_similarity_search(query_emb, database_embs):
    """O(n) complexity - doesn't scale"""
    similarities = []
    for emb in database_embs:  # This becomes slow for millions of embeddings
        sim = cosine_similarity(query_emb, emb)
        similarities.append(sim)
    return similarities

## Solutions: Approximate nearest neighbor (ANN) libraries

## - Faiss (Facebook AI Similarity Search)

## - Annoy (Spotify)

## - Nmslib

## - Hnswlib

```text

### Context Length Limitations

```python

## Most embedding models have fixed context windows

model_limits = {
    "sentence-transformers": 512,      # tokens
    "openai-ada-002": 8191,           # tokens  
    "cohere-embed": 512,              # tokens
    "text-embedding-3-large": 8191    # tokens
}

## Long documents need chunking strategies

def chunk_document(text, max_length=512, overlap=50):
    """Split long text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
    
    return chunks

```text

## Future Directions

### Improved Training Methods

**Self-Supervised Learning**: Better pre-training objectives
**Contrastive Learning**: More effective positive/negative sampling
**Multi-Task Learning**: Training on multiple objectives simultaneously

### Architectural Innovations

**Sparse Embeddings**: Reducing computational and memory requirements
**Hierarchical Embeddings**: Capturing concepts at multiple levels
**Dynamic Embeddings**: Adapting representations based on context

### Cross-Modal Understanding

**Vision-Language**: Better alignment between visual and textual concepts
**Audio-Text**: Connecting spoken and written language
**Multimodal Fusion**: Combining multiple input types effectively

### Efficiency Improvements

**Quantization**: Reducing precision while maintaining quality
**Distillation**: Creating smaller models that match larger ones
**Hardware Optimization**: Specialized chips for embedding computation

Embeddings represent one of the most fundamental advances in machine learning, enabling models to work with symbolic
data in meaningful ways. As the field continues to evolve, embeddings are becoming more sophisticated, efficient, and
capable of capturing nuanced semantic relationships across diverse types of data and modalities.
