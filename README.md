# Tiny-Jarvis: Building a Tiny LLM from Scratch in C# – Project Documentation

This document provides a comprehensive overview of **Tiny-Jarvis** – a from‑scratch implementation of a small transformer‑based language model (LLM) written entirely in C#. The goal was to understand the fundamental components of LLMs, debug common training issues, and create a working miniature conversational model. The project includes custom autograd (`Value`), transformer blocks, attention, feed‑forward networks, character‑level tokenization, and training loops. Below is a detailed account of the architecture, hyperparameters, encountered bugs, fixes, performance optimisations, and future plans.

---

## 1. Core Architecture

The model replicates the GPT‑style decoder‑only transformer.

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

The project contains 4 tokenizers:

- A simple `Character` tokenizer
- A `Unigram` tokenizer
- A `Word Piece` tokenizer
- A `Byte Pair Encoding` tokenizer

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
- **Inference parameters (`temperature`, `top‑k`, `top‑p`) have a huge impact on output quality** – they can be tuned manually, but a genetic algorithm can automate and improve this tuning.

### 9. Next Steps (Immediate)

1. Implement checkpointing (save/load) to avoid retraining from zero.
2. Integrate ILGPU for one operation (e.g., Linear) as a proof of concept.
3. Add dual logging (console + file) to preserve training history.
4. Experiment with multilingual data – e.g., 50 English‑French phrase pairs.
5. Refactor weight categorisation for cleaner code.
6. **Implement a genetic algorithm (GA) for self‑evolution** – specifically to optimise inference parameters (`temperature`, `top‑k`, `top‑p`) dynamically. The GA will maintain a population of parameter sets, evaluate their fitness (e.g., response coherence, diversity, or task‑specific score), and evolve them across generations. This will allow the model to adapt its generation behaviour without retraining, effectively “learning to sample better” through evolutionary search.

The project is now at a stage where the model trains stably and produces plausible outputs. It serves as both an educational tool and a foundation for more ambitious experiments.

## 9.1 Detailed Design: Genetic Algorithm for Parameter Optimisation (Theoretical - not implemented yet)

The genetic algorithm will operate entirely on the **inference hyperparameters** – it does not modify the model’s weights. This is a form of **meta‑optimisation** that runs after training (or interleaved with generation).

### Representation (Genome)

Each individual is a tuple of three values:

| Parameter | Data type | Range | Encoding |
|-----------|-----------|-------|----------|
| `temperature` | `double` | 0.3 – 1.5 | Real‑valued (no encoding) |
| `topK` | `int` | 0 – 100 (0 = disabled) | Integer |
| `topP` | `double` | 0.5 – 1.0 | Real‑valued |

Example individual: `(0.85, 50, 0.92)`

### Population

- Population size: 20–50 individuals.
- Initialisation: random uniform within the allowed ranges, plus a few hand‑picked defaults (e.g., `(0.7, 40, 0.9)`, `(1.0, 0, 1.0)`).

### Fitness Function

The fitness function evaluates a parameter set by running the model on a **validation set of prompts** (e.g., 5–10 fixed conversational starters) and scoring the responses. The score is a weighted combination of:

| Metric | Formula | Weight | Rationale |
|--------|---------|--------|-----------|
| **Length** | `min(response_length, 30) / 30` | 0.3 | Encourages responses longer than 1 token |
| **Uniqueness** | `distinct_tokens(response) / response_length` | 0.2 | Penalises repetitive outputs (e.g., all `'e'`) |
| **Log‑likelihood** | Average per‑token log‑probability from the model (using the same parameters) | 0.3 | Favours confident predictions |
| **Format** | `1.0` if response does **not** start with `[EOS]` or `[BOS]`; else `0.0` | 0.2 | Ensures the model actually produces content |

Final fitness = weighted sum (clamped to [0,1]).

### Genetic Operators

- **Selection**: Tournament selection (size 3) – picks parents based on fitness.
- **Crossover**: Uniform crossover – each gene (temperature, topK, topP) is inherited randomly from either parent with probability 0.5.
- **Mutation**: Gaussian mutation for real‑valued genes (temperature, topP) with standard deviation 0.05; integer mutation for topK (add/subtract up to 5, then clamp). Mutation rate per gene: 0.2.

### Evolutionary Loop

1. **Initialise** random population.
2. **Evaluate** fitness of each individual (run model on validation prompts).
3. **Select** parents via tournament.
4. **Crossover** to create offspring population (same size as parent population).
5. **Mutate** offspring.
6. **Replace** the old population with the new offspring (generational replacement).
7. Repeat for **10–20 generations** (or until convergence).
8. **Output** the best‑performing parameter set found.

### Integration with the Chat Loop

- After the GA finishes, the optimal parameters are stored in a configuration file (e.g., `best_params.json`). <-- an example as this is theoretical for now.
- The chat program loads these parameters at startup and uses them for generation.
- Optionally, the GA can run in the background periodically, re‑evaluating and adapting the parameters as the conversation context changes (though that is more advanced).

### Expected Benefits

- **Automated tuning** – no more manual guessing of temperature/top‑k.
- **Adaptation to the specific model and task** – the GA will find parameters that maximise the fitness function for the tiny LLM (TinyJarvis).
- **Exploration of extreme values** – the GA may discover that a low temperature (e.g., 0.4) with high top‑p produces the most coherent answers, or that high temperature (1.2) yields more creative (but less coherent) responses.

This evolutionary approach directly implements the “self‑evolution” concept – the model learns to sample better without any weight updates, purely through meta‑optimisation of its inference parameters.
