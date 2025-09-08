---
aliases:
- fine-tuning
- model fine-tuning
- transfer learning
- supervised fine-tuning
category: machine-learning
difficulty: intermediate
related:
- llm
- transformer
- supervised-learning
- prompt-engineering
- rag
sources:
- author: Jeremy Howard, Sebastian Ruder
  license: cc-by
  source_title: Universal Language Model Fine-tuning for Text Classification
  source_url: https://arxiv.org/abs/1801.06146
- author: Long Ouyang et al.
  license: cc-by
  source_title: Training language models to follow instructions with human feedback
  source_url: https://arxiv.org/abs/2203.02155
summary: Fine-tuning is the process of adapting a pre-trained machine learning model
  to specific tasks or domains by continuing training on task-specific data. This
  transfer learning approach leverages the general knowledge learned during pre-training
  while specializing the model for particular applications, achieving better performance
  with less data and computational resources than training from scratch.
tags:
- machine-learning
- deep-learning
- llm
- training
- ai-engineering
title: Fine-tuning
updated: '2025-01-15'
---

## Overview

Fine-tuning is a critical technique in modern machine learning that enables the adaptation of pre-trained models to
specific tasks, domains, or use cases. Rather than training a model from scratch, fine-tuning continues the training
process on a pre-trained foundation model using task-specific data. This approach has become essential in the era of
large language models, where pre-trained models like BERT, GPT, and T5 serve as starting points for countless
specialized applications.

## Core Concepts

### Transfer Learning Foundation

Fine-tuning is fundamentally a transfer learning approach:

```python

# Conceptual flow of fine-tuning

def fine_tuning_process():
    """The general fine-tuning workflow"""
    
    # 1. Start with pre-trained model
    base_model = load_pretrained_model("bert-base-uncased")
    # Model already knows language patterns, syntax, semantics
    
    # 2. Add task-specific components
    classifier_head = nn.Linear(base_model.hidden_size, num_classes)
    task_model = TaskSpecificModel(base_model, classifier_head)
    
    # 3. Fine-tune on task data
    fine_tuned_model = train(
        model=task_model,
        data=task_specific_dataset,
        learning_rate=2e-5,  # Much lower than pre-training
        epochs=3             # Much fewer than pre-training
    )
    
    return fine_tuned_model

```text

### Why Fine-tuning Works

Pre-trained models learn hierarchical representations:

```text
Pre-trained Model Knowledge Hierarchy:

Layer 1 (Low-level): Character patterns, tokenization
Layer 2-4 (Syntax): Grammar, sentence structure, POS tags
Layer 5-8 (Semantics): Word meanings, relationships, context
Layer 9-12 (High-level): Complex reasoning, task-specific patterns

Fine-tuning adapts higher layers while preserving lower-level knowledge

```text

## Types of Fine-tuning

### 1. Full Model Fine-tuning

Updates all model parameters:

```python
class FullFineTuning:
    def __init__(self, pretrained_model, num_classes):
        self.base_model = pretrained_model
        self.classifier = nn.Linear(pretrained_model.hidden_size, num_classes)
        
    def setup_training(self):
        """Setup full model fine-tuning"""
        
        # All parameters are trainable
        for param in self.base_model.parameters():
            param.requires_grad = True
            
        # Add task-specific head
        self.model = nn.Sequential(
            self.base_model,
            self.classifier
        )
        
        # Use smaller learning rate for pre-trained layers
        optimizer = torch.optim.AdamW([
            {'params': self.base_model.parameters(), 'lr': 2e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-4}
        ])
        
        return optimizer

## Example: BERT for sentiment classification

class BERTSentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

```text

### 2. Parameter-Efficient Fine-tuning (PEFT)

Updates only a small subset of parameters:

#### LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    
    def __init__(self, in_features, out_features, rank=4, alpha=32):
        super().__init__()
        
        # Original layer remains frozen
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank decomposition: A and B matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
    def forward(self, x, original_output):
        """Add LoRA adaptation to original layer output"""
        
        # Low-rank adaptation: x @ A^T @ B^T
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        
        # Combine with original output
        return original_output + lora_output * self.scaling

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(self, original_layer, rank=4, alpha=32):
        super().__init__()
        
        # Freeze original layer
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # Add LoRA adaptation
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha
        )
        
    def forward(self, x):
        original_output = self.original_layer(x)
        return self.lora(x, original_output)

## Apply LoRA to transformer model

def apply_lora_to_model(model, target_modules=["query", "key", "value"]):
    """Apply LoRA to specific modules in the model"""
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_module = model
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)
                    
                setattr(
                    parent_module,
                    name.split('.')[-1],
                    LoRALinear(module, rank=4, alpha=32)
                )
    
    return model

```text

#### Adapter Layers

```python
class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning"""
    
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        
        self.adapter_down = nn.Linear(hidden_size, adapter_size)
        self.adapter_up = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states):
        """Apply adapter transformation"""
        
        # Down-project
        adapter_input = self.adapter_down(hidden_states)
        adapter_input = self.activation(adapter_input)
        adapter_input = self.dropout(adapter_input)
        
        # Up-project
        adapter_output = self.adapter_up(adapter_input)
        
        # Residual connection
        return hidden_states + adapter_output

class TransformerWithAdapters(nn.Module):
    """Transformer layer with adapter modules"""
    
    def __init__(self, transformer_layer, adapter_size=64):
        super().__init__()
        
        # Freeze original transformer
        self.transformer = transformer_layer
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Add adapters
        self.adapter_attn = AdapterLayer(
            self.transformer.config.hidden_size,
            adapter_size
        )
        self.adapter_ffn = AdapterLayer(
            self.transformer.config.hidden_size,
            adapter_size
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # Original transformer forward pass
        attn_output = self.transformer.attention(hidden_states, attention_mask)
        attn_output = self.adapter_attn(attn_output)  # Apply adapter
        
        ffn_output = self.transformer.feed_forward(attn_output)
        ffn_output = self.adapter_ffn(ffn_output)     # Apply adapter
        
        return ffn_output

```text

### 3. Layer-wise Fine-tuning

Gradually unfreezes layers during training:

```python
class LayerwiseFineTuner:
    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers
        self.current_unfrozen = 0
        
    def setup_initial_training(self):
        """Start by freezing all layers except the head"""
        
        # Freeze all transformer layers
        for i in range(self.num_layers):
            layer = getattr(self.model, f'layer_{i}')
            for param in layer.parameters():
                param.requires_grad = False
        
        # Keep head trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_next_layer(self):
        """Unfreeze the next layer from the top"""
        
        if self.current_unfrozen < self.num_layers:
            layer_idx = self.num_layers - 1 - self.current_unfrozen
            layer = getattr(self.model, f'layer_{layer_idx}')
            
            for param in layer.parameters():
                param.requires_grad = True
                
            self.current_unfrozen += 1
            
    def get_optimizer(self):
        """Create optimizer for currently trainable parameters"""
        
        trainable_params = []
        
        # Add trainable parameters with different learning rates
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    trainable_params.append({'params': param, 'lr': 1e-4})
                else:
                    trainable_params.append({'params': param, 'lr': 2e-5})
        
        return torch.optim.AdamW(trainable_params)

## Training loop with gradual unfreezing

def train_with_layerwise_unfreezing(model, train_data, val_data, epochs_per_unfreeze=2):
    
    fine_tuner = LayerwiseFineTuner(model, num_layers=12)
    fine_tuner.setup_initial_training()
    
    for unfreeze_step in range(model.config.num_hidden_layers + 1):
        
        # Get current optimizer
        optimizer = fine_tuner.get_optimizer()
        
        # Train for a few epochs
        for epoch in range(epochs_per_unfreeze):
            train_epoch(model, train_data, optimizer)
            validate_model(model, val_data)
        
        # Unfreeze next layer
        fine_tuner.unfreeze_next_layer()

```text

## Task-Specific Fine-tuning

### Text Classification

```python
class TextClassificationFineTuner:
    def __init__(self, model_name, num_classes):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        
    def prepare_data(self, texts, labels):
        """Prepare text data for fine-tuning"""
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        return dataset
    
    def fine_tune(self, train_texts, train_labels, val_texts, val_labels):
        """Fine-tune model for classification"""
        
        # Prepare datasets
        train_dataset = self.prepare_data(train_texts, train_labels)
        val_dataset = self.prepare_data(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            learning_rate=2e-5
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Fine-tune
        trainer.train()
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted')
        }

```text

### Question Answering

```python
class QAFineTuner:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
    def prepare_qa_data(self, contexts, questions, answers):
        """Prepare QA data for fine-tuning"""
        
        input_ids = []
        attention_masks = []
        start_positions = []
        end_positions = []
        
        for context, question, answer in zip(contexts, questions, answers):
            # Encode question and context
            encoded = self.tokenizer(
                question,
                context,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Find answer positions in tokenized text
            start_pos, end_pos = self.find_answer_positions(
                context, answer, encoded
            )
            
            input_ids.append(encoded['input_ids'].squeeze())
            attention_masks.append(encoded['attention_mask'].squeeze())
            start_positions.append(start_pos)
            end_positions.append(end_pos)
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'start_positions': torch.tensor(start_positions),
            'end_positions': torch.tensor(end_positions)
        }
    
    def find_answer_positions(self, context, answer, encoded_input):
        """Find start and end positions of answer in tokenized context"""
        
        # Convert tokens back to text to find answer span
        tokens = self.tokenizer.convert_ids_to_tokens(
            encoded_input['input_ids'].squeeze()
        )
        
        # Find answer span in tokens
        answer_tokens = self.tokenizer.tokenize(answer)
        
        for i in range(len(tokens) - len(answer_tokens) + 1):
            if tokens[i:i + len(answer_tokens)] == answer_tokens:
                return i, i + len(answer_tokens) - 1
        
        # If exact match not found, return default positions
        return 0, 0
    
    def train_qa_model(self, train_data, val_data):
        """Train QA model"""
        
        training_args = TrainingArguments(
            output_dir='./qa_model',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        return trainer

```text

### Named Entity Recognition

```python
class NERFineTuner:
    def __init__(self, model_name, label_list):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
    def prepare_ner_data(self, texts, labels):
        """Prepare NER data for fine-tuning"""
        
        tokenized_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt',
            is_split_into_words=True  # For word-level labels
        )
        
        # Align labels with tokens
        aligned_labels = []
        for i, label_seq in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_label = self.align_labels_with_tokens(label_seq, word_ids)
            aligned_labels.append(aligned_label)
        
        tokenized_inputs['labels'] = torch.tensor(aligned_labels)
        
        return tokenized_inputs
    
    def align_labels_with_tokens(self, labels, word_ids):
        """Align word-level labels with subword tokens"""
        
        aligned_labels = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of a word gets the label
                aligned_labels.append(self.label2id[labels[word_id]])
            else:
                # Other subwords get -100 (ignored) or continuation label
                aligned_labels.append(-100)
                
            previous_word_id = word_id
        
        return aligned_labels
    
    def compute_ner_metrics(self, eval_pred):
        """Compute NER evaluation metrics"""
        
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [self.id2label[l] for l in label if l != -100]
            for label in labels
        ]
        
        # Compute metrics using seqeval
        return {
            'precision': precision_score(true_labels, true_predictions),
            'recall': recall_score(true_labels, true_predictions),
            'f1': f1_score(true_labels, true_predictions),
            'accuracy': accuracy_score(true_labels, true_predictions)
        }

```text

## Advanced Fine-tuning Techniques

### Multi-task Fine-tuning

```python
class MultiTaskFineTuner:
    def __init__(self, model_name, task_configs):
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.task_heads = nn.ModuleDict()
        self.task_configs = task_configs
        
        # Create task-specific heads
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                self.task_heads[task_name] = nn.Linear(
                    self.shared_encoder.config.hidden_size,
                    config['num_classes']
                )
            elif config['type'] == 'token_classification':
                self.task_heads[task_name] = nn.Linear(
                    self.shared_encoder.config.hidden_size,
                    config['num_labels']
                )
    
    def forward(self, input_ids, attention_mask, task_name):
        """Forward pass for specific task"""
        
        # Shared encoding
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Task-specific head
        if self.task_configs[task_name]['type'] == 'classification':
            logits = self.task_heads[task_name](outputs.pooler_output)
        else:
            logits = self.task_heads[task_name](outputs.last_hidden_state)
        
        return logits
    
    def train_multitask(self, task_datasets, num_epochs=3):
        """Train on multiple tasks simultaneously"""
        
        # Create combined dataloader
        combined_loader = self.create_multitask_dataloader(task_datasets)
        
        # Setup optimizer
        optimizer = AdamW(self.parameters(), lr=2e-5)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in combined_loader:
                optimizer.zero_grad()
                
                # Get batch task and data
                task_name = batch['task']
                
                # Forward pass
                logits = self.forward(
                    batch['input_ids'],
                    batch['attention_mask'],
                    task_name
                )
                
                # Compute task-specific loss
                loss = self.compute_task_loss(logits, batch['labels'], task_name)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(combined_loader)}")
    
    def create_multitask_dataloader(self, task_datasets):
        """Create a dataloader that samples from multiple tasks"""
        
        # Implementation depends on specific requirements
        # Could use round-robin, weighted sampling, or other strategies
        pass

```text

### Continual Fine-tuning

```python
class ContinualFineTuner:
    def __init__(self, model, regularization_strength=0.1):
        self.model = model
        self.regularization_strength = regularization_strength
        self.previous_task_params = None
        
    def compute_ewc_loss(self, current_params):
        """Compute Elastic Weight Consolidation loss"""
        
        if self.previous_task_params is None:
            return 0
        
        ewc_loss = 0
        for name, param in current_params:
            if name in self.previous_task_params:
                # Fisher information weighting (simplified)
                fisher_info = self.previous_task_params[name]['fisher']
                prev_param = self.previous_task_params[name]['param']
                
                ewc_loss += (fisher_info * (param - prev_param) ** 2).sum()
        
        return self.regularization_strength * ewc_loss
    
    def fine_tune_new_task(self, new_task_data, preserve_old_knowledge=True):
        """Fine-tune on new task while preserving old knowledge"""
        
        # Store current parameters before training
        if preserve_old_knowledge:
            self.store_task_parameters()
        
        # Regular fine-tuning loop with EWC regularization
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(3):
            for batch in new_task_data:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                task_loss = outputs.loss
                
                # Add regularization loss
                if preserve_old_knowledge:
                    ewc_loss = self.compute_ewc_loss(self.model.named_parameters())
                    total_loss = task_loss + ewc_loss
                else:
                    total_loss = task_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
    
    def store_task_parameters(self):
        """Store current parameters and compute Fisher information"""
        
        self.previous_task_params = {}
        
        for name, param in self.model.named_parameters():
            # Store parameter values
            param_copy = param.detach().clone()
            
            # Compute Fisher information (simplified diagonal approximation)
            fisher_info = param.grad ** 2 if param.grad is not None else torch.zeros_like(param)
            
            self.previous_task_params[name] = {
                'param': param_copy,
                'fisher': fisher_info
            }

```text

## Instruction Tuning and RLHF

### Supervised Fine-tuning (SFT)

```python
class InstructionTuner:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_instruction_data(self, instructions, inputs, outputs):
        """Prepare instruction-following dataset"""
        
        formatted_examples = []
        
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            # Format as instruction-following example
            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
            else:
                prompt = f"Instruction: {instruction}\nOutput: {output_text}"
            
            formatted_examples.append(prompt)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_examples,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    def instruction_tune(self, instruction_data, num_epochs=3):
        """Fine-tune model on instruction-following data"""
        
        training_args = TrainingArguments(
            output_dir='./instruction_tuned_model',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,  # Lower learning rate for instruction tuning
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy='steps',
            eval_steps=500
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=instruction_data,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        return trainer

```text

### Reinforcement Learning from Human Feedback (RLHF)

```python
class RLHFTrainer:
    def __init__(self, sft_model, reward_model):
        self.sft_model = sft_model
        self.reward_model = reward_model
        self.reference_model = copy.deepcopy(sft_model)  # Frozen reference
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
    
    def compute_ppo_loss(self, queries, responses, rewards, old_log_probs):
        """Compute PPO loss for RLHF"""
        
        # Get current model log probabilities
        current_log_probs = self.get_log_probabilities(queries, responses)
        
        # Get reference model log probabilities
        with torch.no_grad():
            reference_log_probs = self.get_reference_log_probabilities(queries, responses)
        
        # Compute ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # KL penalty to prevent drift from reference model
        kl_penalty = self.compute_kl_penalty(current_log_probs, reference_log_probs)
        
        # Adjust rewards with KL penalty
        adjusted_rewards = rewards - 0.2 * kl_penalty  # Beta = 0.2
        
        # PPO clipping
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)  # Epsilon = 0.2
        
        # Policy loss
        policy_loss1 = adjusted_rewards * ratio
        policy_loss2 = adjusted_rewards * clipped_ratio
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        # Value loss (if using value function)
        value_loss = 0  # Simplified
        
        total_loss = policy_loss + value_loss
        
        return total_loss
    
    def train_with_ppo(self, prompts, num_iterations=1000):
        """Train model using PPO for RLHF"""
        
        optimizer = AdamW(self.sft_model.parameters(), lr=1e-6)
        
        for iteration in range(num_iterations):
            # Generate responses
            responses = self.generate_responses(prompts)
            
            # Get rewards from reward model
            rewards = self.get_rewards(prompts, responses)
            
            # Get old log probabilities
            with torch.no_grad():
                old_log_probs = self.get_log_probabilities(prompts, responses)
            
            # PPO update
            optimizer.zero_grad()
            loss = self.compute_ppo_loss(prompts, responses, rewards, old_log_probs)
            loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
    
    def get_rewards(self, prompts, responses):
        """Get rewards from trained reward model"""
        
        with torch.no_grad():
            # Combine prompts and responses
            combined = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses)]
            
            # Tokenize
            inputs = self.tokenizer(
                combined,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            
            # Get reward scores
            rewards = self.reward_model(**inputs).logits.squeeze(-1)
            
        return rewards

```text

## Optimization Strategies

### Learning Rate Scheduling

```python
class FineTuningOptimizer:
    def __init__(self, model, base_lr=2e-5):
        self.model = model
        self.base_lr = base_lr
        
    def setup_layerwise_lr(self, num_layers, decay_factor=0.95):
        """Setup different learning rates for different layers"""
        
        optimizer_params = []
        
        # Embedding layer - lowest learning rate
        optimizer_params.append({
            'params': self.model.embeddings.parameters(),
            'lr': self.base_lr * (decay_factor ** num_layers)
        })
        
        # Transformer layers - decreasing learning rates for earlier layers
        for i in range(num_layers):
            layer = getattr(self.model, f'layer_{i}')
            lr = self.base_lr * (decay_factor ** (num_layers - i - 1))
            
            optimizer_params.append({
                'params': layer.parameters(),
                'lr': lr
            })
        
        # Classification head - highest learning rate
        optimizer_params.append({
            'params': self.model.classifier.parameters(),
            'lr': self.base_lr * 2
        })
        
        return AdamW(optimizer_params)
    
    def setup_warmup_schedule(self, optimizer, num_warmup_steps, num_training_steps):
        """Setup learning rate schedule with warmup"""
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                # Linear decay
                return max(
                    0.0,
                    float(num_training_steps - current_step) /
                    float(max(1, num_training_steps - num_warmup_steps))
                )
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler

```text

### Gradient Clipping and Accumulation

```python
class GradientManager:
    def __init__(self, max_grad_norm=1.0, accumulation_steps=4):
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.accumulated_steps = 0
        
    def accumulate_gradients(self, loss):
        """Accumulate gradients over multiple steps"""
        
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_steps += 1
        
        # Return whether to perform optimizer step
        return self.accumulated_steps % self.accumulation_steps == 0
    
    def step_with_clipping(self, optimizer, model):
        """Perform optimizer step with gradient clipping"""
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Reset accumulation counter
        self.accumulated_steps = 0

## Usage in training loop

gradient_manager = GradientManager(max_grad_norm=1.0, accumulation_steps=4)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    
    should_step = gradient_manager.accumulate_gradients(loss)
    
    if should_step:
        gradient_manager.step_with_clipping(optimizer, model)

```text

## Evaluation and Monitoring

### Fine-tuning Metrics

```python
class FineTuningMonitor:
    def __init__(self, model, validation_data):
        self.model = model
        self.validation_data = validation_data
        self.metrics_history = []
        
    def compute_perplexity(self, texts):
        """Compute perplexity on validation set"""
        
        total_log_likelihood = 0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt')
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                
                # Negative log likelihood
                nll = outputs.loss * inputs['input_ids'].size(1)
                total_log_likelihood += nll.item()
                total_tokens += inputs['input_ids'].size(1)
        
        # Perplexity = exp(average negative log likelihood)
        perplexity = torch.exp(torch.tensor(total_log_likelihood / total_tokens))
        return perplexity.item()
    
    def evaluate_task_performance(self, task_data):
        """Evaluate on downstream task"""
        
        predictions = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in task_data:
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        return {'accuracy': accuracy, 'f1': f1}
    
    def check_catastrophic_forgetting(self, original_tasks):
        """Check if model forgot previous tasks"""
        
        forgetting_metrics = {}
        
        for task_name, task_data in original_tasks.items():
            current_performance = self.evaluate_task_performance(task_data)
            
            # Compare with baseline (assuming we stored it)
            if hasattr(self, f'{task_name}_baseline'):
                baseline = getattr(self, f'{task_name}_baseline')
                forgetting = baseline['accuracy'] - current_performance['accuracy']
                forgetting_metrics[task_name] = forgetting
        
        return forgetting_metrics
    
    def monitor_training_progress(self, epoch, train_loss, val_loss):
        """Monitor training progress and detect issues"""
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'perplexity': self.compute_perplexity(self.validation_data),
            'task_performance': self.evaluate_task_performance(self.validation_data)
        }
        
        self.metrics_history.append(metrics)
        
        # Detect potential issues
        if len(self.metrics_history) > 1:
            prev_metrics = self.metrics_history[-2]
            
            # Check for overfitting
            if val_loss > prev_metrics['val_loss'] and train_loss < prev_metrics['train_loss']:
                print(f"Warning: Potential overfitting at epoch {epoch}")
            
            # Check for performance degradation
            current_f1 = metrics['task_performance']['f1']
            prev_f1 = prev_metrics['task_performance']['f1']
            
            if current_f1 < prev_f1 - 0.05:
                print(f"Warning: Performance drop at epoch {epoch}")
        
        return metrics

```text

## Best Practices and Common Pitfalls

### Best Practices

```python
class FineTuningBestPractices:
    """Collection of fine-tuning best practices"""
    
    @staticmethod
    def prepare_high_quality_data(raw_data):
        """Prepare high-quality training data"""
        
        best_practices = {
            'data_quality': [
                "Remove duplicates and near-duplicates",
                "Fix obvious errors and inconsistencies",
                "Ensure label quality and consistency",
                "Balance class distributions when possible"
            ],
            
            'data_size': [
                "Start with 100-1000 examples for initial experiments",
                "Gradually increase to 1000-10000 for production",
                "More data usually helps, but quality > quantity"
            ],
            
            'data_format': [
                "Use consistent formatting across examples",
                "Include diverse examples covering edge cases",
                "Separate train/validation/test sets properly",
                "Use stratified sampling for small datasets"
            ]
        }
        
        return best_practices
    
    @staticmethod
    def choose_hyperparameters():
        """Guidelines for hyperparameter selection"""
        
        recommendations = {
            'learning_rate': {
                'classification': '2e-5 to 5e-5',
                'generation': '1e-5 to 3e-5',
                'instruction_tuning': '5e-6 to 1e-5'
            },
            
            'batch_size': {
                'small_model': '16-32',
                'large_model': '4-8 (with gradient accumulation)',
                'memory_constraint': 'Use gradient accumulation'
            },
            
            'epochs': {
                'small_dataset': '5-10 epochs',
                'large_dataset': '2-3 epochs',
                'monitor': 'Use early stopping'
            }
        }
        
        return recommendations

def avoid_common_pitfalls():
    """Common pitfalls and how to avoid them"""
    
    pitfalls = {
        'overfitting': {
            'problem': 'Model memorizes training data, poor generalization',
            'solutions': [
                'Use validation set for early stopping',
                'Apply dropout and weight decay',
                'Reduce learning rate or epochs',
                'Increase dataset size or diversity'
            ]
        },
        
        'catastrophic_forgetting': {
            'problem': 'Model forgets previous knowledge',
            'solutions': [
                'Use lower learning rates',
                'Apply regularization (EWC, L2)',
                'Mix original and new task data',
                'Use parameter-efficient methods (LoRA)'
            ]
        },
        
        'poor_convergence': {
            'problem': 'Training loss not decreasing',
            'solutions': [
                'Check learning rate (too high/low)',
                'Verify data preprocessing',
                'Increase warmup steps',
                'Check gradient clipping'
            ]
        },
        
        'inconsistent_results': {
            'problem': 'Results vary significantly between runs',
            'solutions': [
                'Set random seeds properly',
                'Use multiple runs and average results',
                'Check for data leakage',
                'Ensure reproducible preprocessing'
            ]
        }
    }
    
    return pitfalls

```text
Fine-tuning represents one of the most practical and effective approaches to adapting powerful pre-trained models for
specific tasks and domains. As foundation models continue to grow in capability and size, fine-tuning techniques—from
full parameter updates to parameter-efficient methods like LoRA—enable practitioners to harness these capabilities for
specialized applications. The key to successful fine-tuning lies in understanding the trade-offs between different
approaches, carefully preparing high-quality training data, and monitoring the training process to prevent common
pitfalls like overfitting and catastrophic forgetting.
