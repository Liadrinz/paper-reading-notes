# BERT

## BERT: Bidirectional Encoder Representations from Transformers

### Two strategies for applying pre-trained language representations

- feature-based: regard the pre-trained representations as additional features (ELMo)
- fine-tuning: simply fine-tune all pre-trained parameters (BERT)
- The two approaches share the same objective function during pre-training (the pre-training methods is independent from the strategies for applying pre-trained model)

### The Architecture

a multi-layer bidirectional Transformer encoder

- multi-layer: multiple Transformer modules
- bidirectional: the leftward attentions are not masked out
- encoder: excluding the decoder part

Hyperparameters

- $\bold{BERT}_{\bold{BASE}}(\text{L=12,H=768,A=12,Total Parameters=110M})$
- $\bold{BERT}_{\bold{LARGE}}(\text{L=24,H=1024,A=16,Total Parameters=340M})$

### I/O Representations

- Able to unambiguously represent both a single sentence and a pair of sentences
  - "sentence": can be arbitrary span of contiguous text, rather than an actual linguistic sentence
  - "sequence": may be a single sentence or two sentences packed together

- Input
  - construction: Original Token + Segment (Which sentence) + Position Embedding
  - The first token of every sequence is always a special classification token [CLS], the final hidden state of which is used as the aggregate sequence representation for classification tasks.

- Output (the final hidden vector)
  - for [CLS]: $C\in\mathbb{R}^H$
  - for the $i$-th input token: $T_i\in\mathbb{R}^H$

### Pre-training BERT

- Tasks
  - Masked LM (Cloze Task)
  - Next Sentence Prediction
- Transfering
  - Prior work: only the embeddings are transferred
  - BERT: all parameters are transferred
- Data: use document-level corpus instead of shuffled sentence-level corpus

### Fine-tuning BERT

- Handling Text Pairs
  - Common: encode each sentence separately, then does bidirectional cross attention
  - BERT: uses the sel-attention mechanism to unify these two stages
- Fine-tuning
  - Plug in the task-specific inputs and outputs into BERT
  - Add an output layer after the pre-trained BERT
