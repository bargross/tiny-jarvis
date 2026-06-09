# Tiny-Jarvis: Building a Tiny LLM from Scratch in C# – Project Documentation

This document provides a comprehensive overview of the **Tiny-Jarvis** project – a from‑scratch implementation of a small transformer‑based language model (LLM) written entirely in C#. The goal was to understand the fundamental components of LLMs, debug common training issues, and create a working miniature conversational model. The project includes custom autograd (`Value`), transformer blocks, attention, feed‑forward networks, character‑level tokenization, and training loops. Below is a detailed account of the architecture, hyperparameters, encountered bugs, fixes, performance optimisations, and future plans.

---

## 1. Core Architecture

The model is implemented in `TinyJarvisModel` (originally named `TinyJarvisModel`). It replicates the GPT‑style decoder‑only transformer.

### 1.1 Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Token Embeddings (`wte`)** | Maps token IDs to dense vectors | Matrix `[vocabSize, embeddingSize]`, learned |
| **Position Embeddings (`wpe`)** | Injects position information | Matrix `[maxSequenceLength, embeddingSize]`, learned (or fixed sinusoids; we use learned) |
| **Transformer Block** | Applies self‑attention + feed‑forward | Repeated `layerCount` times |
| **Multi‑Head Self‑Attention** | Allows tokens to attend to previous tokens | Split into `headCount` heads, each with Q, K, V projections |
| **Feed‑Forward Network (MLP)** | Processes each token independently | Two linear layers with ReLU, expansion factor 4 |
| **Output Head (`lm_head`)** | Projects final hidden state to vocabulary logits | Linear layer `[vocabSize, embeddingSize]` |
| **RMSNorm** | Stabilises activations (pre‑norm) | Root‑mean‑square normalisation |
| **Residual Connections** | Adds input to output of each sub‑layer | Enables deeper training |

### 1.2 Autograd System: The `Value` Class

Every operation (addition, multiplication, ReLU, softmax, etc.) creates a `Value` object that records its inputs and local gradients. This builds a computation graph. During backpropagation (`loss.Backward()`), gradients are computed using reverse‑mode automatic differentiation. This approach is educational but memory‑ and CPU‑intensive.

### 1.3 Hyperparameters Used in Final Training

| Parameter | Value | Reason |
|-----------|-------|--------|
| `vocabSize` | 73 (character‑level) | Derived from training corpus; enough for letters, digits, basic punctuation |
| `embeddingSize` | 32 | Balances capacity and speed; divisible by `headCount` |
| `headCount` | 4 | Each head gets 8 dimensions (32/4) |
| `layerCount` | 4 | Sufficient depth for character patterns |
| `maxSequenceLength` | 32 | Short context to keep training fast; later could increase |
| `totalNumberOfSteps` | 10,000 | Early experiments showed convergence within 10k steps |
| `learningRate` | 0.001 | Adam default; used linear decay to 0 over steps |
| `maxGradNorm` | 1.0 | Gradient clipping to prevent explosions |
| `MomentumSmoothing` (β₁) | 0.9 | Standard Adam |
| `SquaredGradSmoothing` (β₂) | 0.999 | Standard Adam |
| `Epsilon` | 1e-8 | Numerical stability |

### 1.4 Tokenization

Initially we attempted a custom WordPiece trainer but faced performance and correctness issues. We fell back to a **character‑level tokenizer** that maps each character to a unique ID. Special tokens added: `[UNK]` (ID 0), `[BOS]` (ID 1), `[EOS]` (ID 2). This simplified debugging and is sufficient for a tiny LLM.

---

## 2. Training Process

### 2.1 Data Preparation

Training corpus: a mix of synthetic conversational pairs and short English sentences. Each training example is a line containing `user: query assistant: response`. The sequence is tokenized, then wrapped with `[BOS]` at the start and `[EOS]` at the end.

Example training sequence:  
`[BOS] user: hello assistant: hi [EOS]`

### 2.2 Training Loop (Simplified)

```csharp
for (int step = start; step < end; step++)
{
    // Get a document (line)
    string doc = docs[step % docs.Count];
    // Tokenize
    List<int> tokens = new List<int> { tokenizer.Bos };
    tokens.AddRange(tokenizer.Encode(doc));
    tokens.Add(tokenizer.Eos);

    int tokenCount = Math.Min(tokens.Count - 1, maxSequenceLength - 1);
    var keys = model.CreateKvCache();
    var values = model.CreateKvCache();
    Value loss = new Value(0);

    for (int pos = 0; pos < tokenCount; pos++)
    {
        var logits = model.Forward(tokens[pos], pos, keys, values);
        loss += Helpers.CrossEntropyLoss(logits, tokens[pos + 1]);
    }
    loss *= 1.0 / tokenCount;

    // Backward and optimiser step
    optimiser.ZeroGrad();
    loss.Backward(topo, visited, backwardStack);
    optimiser.Step(step);
}
```

### 2.3 Critical Bug and Fix
Bug: Initially, the loss used the current token as the target instead of the next token:

```csharp
loss += CrossEntropyLoss(logits, tokens[pos]);   // ❌ wrong
```

This trained the model to predict the same token, causing it to collapse to always output the most frequent character ('e').

**Fix**: Use the next token as target:
```csharp
loss += CrossEntropyLoss(logits, tokens[pos + 1]); // ✅ correct
```

### 3. Training Logs and Performance
The training loss evolved as follows (moving average):

| Step |	Moving Average Loss | Generation Output (temp=0.8) |
|------|------------------------|------------------------------|
| 1    |	4.40	| (random) |
| 500	 | 1.24	    | asist a |
| 1000 |	1.02	| assistan |
| 2000 |	0.86	| assistan |
| 2500 |	0.80	| assist |

The model learned to spell “assistant” with small errors, a huge success for such a tiny character‑level model. The loss continued to decrease, indicating ongoing learning.

Observation: The generation output sometimes regressed due to temperature‑based sampling; lowering temperature to 0.5 gave more deterministic (but still imperfect) completions.


### 4. Optimisation and Refactoring Efforts
## 4.1 Performance Bottlenecks
The Value autograd system creates a new object for every arithmetic operation, leading to memory spikes (5‑10 GB) and high CPU usage. Reducing embeddingSize from 64 to 32 dropped memory to 1 GB and improved stability.

## 4.2 Hybrid Computation
We introduced CrossEntropyLoss that computes softmax using double[] instead of Value nodes, returning only a single Value. This reduced graph size dramatically. Similar optimisations can be applied to attention and MLP.

## 4.3 Weight Categorisation (Planned)
Currently weights are stored in a dictionary _stateDict with string keys (e.g., "layer0.attn_wq"). To improve cache locality and code clarity, we plan to refactor into typed fields:

- `Value[][] _tokenEmbeddings`

- `Value[][] _positionEmbeddings`

- `List<LayerWeights> _layers` (each containing Q, K, V, O, fc1, fc2)

- `Value[][] _outputHead`

This will also simplify saving/loading.

## 4.4 GPU Integration via ILGPU (Planned)
The project currently runs on CPU only. We own an NVIDIA GTX 1080 Ti (11 GB VRAM). To leverage it while keeping the custom autograd, we plan to integrate ILGPU to write explicit CUDA kernels for matrix multiplication, softmax, and attention. This will be a learning exercise in low‑level GPU programming. Steps:

Replace Helpers.Linear with a GPU kernel that performs matrix‑vector multiplication on the device.

Keep the Value graph on the host; only the heavy linear algebra runs on GPU.

Manage memory buffers (copy weights to GPU once, then reuse).

This approach retains full control over the autograd system while accelerating the bottleneck operations.

### 5. Saving and Resuming Training
Currently training starts from scratch each time. We will implement:

- Checkpointing: After each milestone (e.g., every 1000 steps), save:

    - Model weights (all Value.Data arrays) in a binary file.

    - Optimiser state (Adam momentums and squared averages).

    - Current step count and moving average loss.

- Resume: Load the saved state and continue training from that step.

A simple binary format: write integers for hyperparameters, then the raw double values of each parameter.

### 6. Logging and Monitoring
We already added periodic generation tests. We will also add:

- Console + file logging using a TeeWriter that duplicates all console output to a timestamped log file.

- Loss history to CSV for later analysis.

- Parameter norm tracking to detect gradient explosions early.

### 7. Future: Multi‑lingual Training
The model is character‑based, so it can theoretically learn any language written in a Latin script (or even Unicode characters if we expand the vocabulary). Training on parallel text (e.g., English‑French sentence pairs) could enable simple translation. However, the tiny context window (32 characters) severely limits translation quality – longer sequences are needed. A practical approach:

- Increase maxSequenceLength to 128 (requires retraining with larger embeddings).

- Use a subword tokenizer (BPE) to reduce sequence length per language.

- Train on short parallel phrases: user: Hello assistant: Bonjour.

Given the model’s size, it will never be a good general translator, but it could learn a few simple equivalences – a fun demo.

### 8. Lessons Learned

- Autograd from scratch is invaluable for learning, but performance requires careful graph reduction.

- The target token bug is a classic mistake; always verify that the loss uses the next token.

- Hyperparameters matter – too small embeddingSize (16) limits learning; too large (64) slows CPU training unacceptably. 32 was a good balance.

- Character‑level models can learn to spell short words, but struggle with long‑range dependencies.

- Testing generation during training gives early feedback and is easy to implement.

### 9. Next Steps (Immediate)

1. Implement checkpointing (save/load) to avoid retraining from zero.

2. Integrate ILGPU for one operation (e.g., Linear) as a proof of concept.

3. Add dual logging (console + file) to preserve training history.

4. Experiment with multilingual data – e.g., 50 English‑French phrase pairs.

5. Refactor weight categorisation for cleaner code.

The project is now at a stage where the model trains stably and produces plausible outputs. It serves as both an educational tool and a foundation for more ambitious experiments.