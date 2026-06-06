using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.Util;
using Tiny.Jarvis.Models;

namespace Tiny.Jarvis.MLModels;

public class TinyJarvisModel
{
    // The state dict keys follow PyTorch / GPT-2 convention (wte = weight token embedding,
    // wpe = weight position embedding, etc.) so this code can map directly to PyTorch
    // checkpoints if you ever want to load real GPT-2 weights. The aliased properties
    // below give us readable C# names to use inside Forward without losing that bridge.
    private readonly Dictionary<string, Value[][]> _stateDict;
    private readonly int _embeddingSize;
    private readonly int _headCount;
    private readonly int _layerCount;
    private readonly int _headDimension;
    private readonly int _vocabularySize;
    private readonly int _bosToken;

    private Value[][] TokenEmbeddings => _stateDict["wte"];
    private Value[][] PositionEmbeddings => _stateDict["wpe"];
    private Value[][] OutputProjection => _stateDict["lm_head"];

    /// <summary>All trainable parameters, flattened into a single list for the optimiser.</summary>
    public List<Value> Parameters { get; }
    public int MaxSequenceLength { get; }

    public TinyJarvisModel(
        int vocabSize,
        int embeddingSize,
        int headCount,
        int layerCount,
        int maxSequenceLength,
        int bosToken,
        Random random
    )
    {
        _embeddingSize = embeddingSize;
        _headCount = headCount;
        _layerCount = layerCount;
        _headDimension = embeddingSize / headCount;
        _vocabularySize = vocabSize;
        _bosToken = bosToken;

        _stateDict = new Dictionary<string, Value[][]>
        {
            ["wte"] = Helpers.CreateMatrix(random, vocabSize, embeddingSize),
            ["wpe"] = Helpers.CreateMatrix(random, maxSequenceLength, embeddingSize),
            ["lm_head"] = Helpers.CreateMatrix(random, vocabSize, embeddingSize),
        };

        for (int i = 0; i < layerCount; i++)
        {
            _stateDict[$"layer{i}.attn_wq"] = Helpers.CreateMatrix(random, embeddingSize, embeddingSize);
            _stateDict[$"layer{i}.attn_wk"] = Helpers.CreateMatrix(random, embeddingSize, embeddingSize);
            _stateDict[$"layer{i}.attn_wv"] = Helpers.CreateMatrix(random, embeddingSize, embeddingSize);
            _stateDict[$"layer{i}.attn_wo"] = Helpers.CreateMatrix(random, embeddingSize, embeddingSize);
            _stateDict[$"layer{i}.mlp_fc1"] = Helpers.CreateMatrix(
                random,
                4 * embeddingSize,
                embeddingSize
            );  
            _stateDict[$"layer{i}.mlp_fc2"] = Helpers.CreateMatrix(
                random,
                embeddingSize,
                4 * embeddingSize
            );
        }

        MaxSequenceLength = maxSequenceLength;

        // Dictionary<TKey,TValue> enumeration order is not guaranteed by the spec.
        // In .NET Core+ it preserves insertion order in practice, so Adam's momentum[]/squaredGradAvg[]
        // line up across runs - but if that implementation detail ever changes, switch
        // to a List<KeyValuePair<string, ...>> to make the order explicit.
        Parameters = _stateDict.Values.SelectMany(mat => mat).SelectMany(row => row).ToList();
    }

    public List<Value> Forward(
        int tokenId,
        int posId,
        List<List<Value>>[] keys,
        List<List<Value>>[] values
    )
    {
        // validate ids
        if (tokenId < 0 || tokenId >= TokenEmbeddings.Length)
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"tokenId {tokenId} is out of bounds for vocab size {TokenEmbeddings.Length}");

        if (posId < 0 || posId >= PositionEmbeddings.Length)
            throw new ArgumentOutOfRangeException(nameof(posId), $"posId {posId} is out of bounds for position embedding size {PositionEmbeddings.Length}");

        // use ids directly (remove the -1 adjustment)
        var tokenEmbedding = TokenEmbeddings.GetRow(tokenId);
        var positionEmbedding = PositionEmbeddings.GetRow(posId);

        var probabilities = new List<Value>();
        for (int i = 0; i < _embeddingSize; i++)
        {
            probabilities.Add(tokenEmbedding[i] + positionEmbedding[i]);
        }

        // Initial RmsNorm: stabilises the embeddings before entering the first block.
        // This isn't standard in all transformer implementations, but gives the
        // residual stream a stable starting magnitude.
        probabilities = Helpers.RmsNorm(probabilities);

        for (int layerIndex = 0; layerIndex < _layerCount; layerIndex++)
        {
            probabilities = AttentionBlock(probabilities, layerIndex, keys, values);
            probabilities = MlpBlock(probabilities, layerIndex);
        }

        // Note: production transformers typically apply a final RmsNorm here
        // before the output projection. We omit it for simplicity.
        return Helpers.Linear(probabilities, OutputProjection);
    }

    // Attention wrapped with pre-norm and a residual connection.
    // Mutates keys[layerIndex] and values[layerIndex] by appending the current position's K and V.
    private List<Value> AttentionBlock(
        List<Value> probabilities,
        int layerIndex,
        List<List<Value>>[] keys,
        List<List<Value>>[] values
    )
    {
        var xResidual = new List<Value>(probabilities);
        probabilities = Helpers.RmsNorm(probabilities);

        List<Value> query = Helpers.Linear(probabilities, _stateDict[$"layer{layerIndex}.attn_wq"]);
        List<Value> key = Helpers.Linear(probabilities, _stateDict[$"layer{layerIndex}.attn_wk"]);
        List<Value> value = Helpers.Linear(probabilities, _stateDict[$"layer{layerIndex}.attn_wv"]);

        keys[layerIndex].Add(key);
        values[layerIndex].Add(value);

        var concatenatedHeads = new List<Value>();
        for (int count = 0; count < _headCount; count++)
        {
            int headStart = count * _headDimension;
            List<Value> queryForHead = query.GetRange(headStart, _headDimension);

            var attentionLogits = new List<Value>();
            int cachedCount = keys[layerIndex].Count;
            for (int t = 0; t < cachedCount; t++)
            {
                List<Value> keyForHead = keys[layerIndex][t].GetRange(headStart, _headDimension);
                var dot = new Value(0);
                for (int j = 0; j < _headDimension; j++)
                {
                    dot += queryForHead[j] * keyForHead[j];
                }

                attentionLogits.Add(dot / Math.Sqrt(_headDimension));
            }

            List<Value> attentionWeights = Helpers.Softmax(attentionLogits);

            var headOutput = new List<Value>();
            for (int j = 0; j < _headDimension; j++)
            {
                headOutput.Add(new Value(0));
            }

            for (int dimension = 0; dimension < cachedCount; dimension++)
            {
                List<Value> valueForHead = values[layerIndex][dimension]
                    .GetRange(headStart, _headDimension);

                Value weight = attentionWeights[dimension];
                for (int idx = 0; idx < _headDimension; idx++)
                {
                    headOutput[idx] += weight * valueForHead[idx];
                }
            }

            concatenatedHeads.AddRange(headOutput);
        }

        probabilities = Helpers.Linear(concatenatedHeads, _stateDict[$"layer{layerIndex}.attn_wo"]);
        for (int i = 0; i < _embeddingSize; i++)
        {
            probabilities[i] += xResidual[i];
        }

        return probabilities;
    }

    // Two-layer feed-forward with ReLU, wrapped with pre-norm and a residual connection.
    private List<Value> MlpBlock(List<Value> probabilities, int layerIndex)
    {
        var xResidual = new List<Value>(probabilities);

        probabilities = Helpers.RmsNorm(probabilities);
        probabilities = Helpers.Linear(probabilities, _stateDict[$"layer{layerIndex}.mlp_fc1"]);

        probabilities = probabilities.Select(xi => xi.Relu()).ToList();

        probabilities = Helpers.Linear(probabilities, _stateDict[$"layer{layerIndex}.mlp_fc2"]);
        
        for (int embeddingIndex = 0; embeddingIndex < _embeddingSize; embeddingIndex++)
            probabilities[embeddingIndex] += xResidual[embeddingIndex];
        

        return probabilities;
    }

    /// <summary>
    /// Generates new token IDs autoregressively, given a starting prompt.
    /// </summary>
    /// <param name="inputIds">Token IDs of the prompt (from tokenizer.Encode).</param>
    /// <param name="maxNewTokens">Maximum number of tokens to generate.</param>
    /// <param name="temperature">>1 = more random, <1 = more deterministic.</param>
    /// <param name="topK">If >0, only sample from the K most likely tokens.</param>
    /// <param name="topP">Nucleus sampling: keep smallest set of tokens whose cumulative prob >= topP.</param>
    /// <param name="endTokenId">If provided, stop generation when this token is produced.</param>
    /// <returns>List of newly generated token IDs (excluding the original prompt).</returns>
    public IReadOnlyList<int> Generate(
        IReadOnlyList<int> tokens,
        int maxNewTokens,
        double temperature = 1.0,
        int topK = 0,
        double topP = 1.0,
        int? endTokenId = null,
        bool prependBos = true)
    {
        // Create fresh KV caches: one list per layer, each will store keys/values for every position
        var keys = CreateKvCache();   // List<List<Value>>[layerCount]
        var values = CreateKvCache();

        // promptTokens will hold the full sequence (prompt + generated)
        var promptTokens = new List<int>();

        // This will store the logits returned by the last Forward call.
        // After processing the final prompt token, these logits represent predictions for the first new token.
        List<Value>? lastLogits = null;

        if (prependBos && (tokens.Count == 0 || tokens[0] != _bosToken))
        {
            tokens = new List<int> {  }.Concat(tokens).ToList();
        }

        // Any name longer than maxSequenceLength - 1 is silently truncated here.
        int tokenCount = Math.Min(MaxSequenceLength - 1, tokens.Count);

        // ----- Feed the entire prompt (one token at a time) to fill the KV caches -----
        for (int posId = 0; posId < tokenCount; posId++)
        {
            int tokenId = tokens[posId];

            // Forward returns logits for the *next* token (position pos+1)
            lastLogits = Forward(tokenId, posId, keys, values);
            // We ignore the logits from intermediate steps; only the final last Logits matters.
        }

        // currentPos tracks the index of the token we will feed next.
        // After the prompt, the next position to fill is at index inputIds.Count.
        int currentPos = tokenCount;

        // ----- Generate new tokens -----
        for (int step = 0; step < maxNewTokens; step++)
        {
            // Sample the next token ID using the configured sampling strategy
            int nextToken = Helpers.SampleToken(lastLogits, temperature, topK, topP);

            // Append the new token to our full sequence
            promptTokens.Add(nextToken);

            // If we hit the end-of-sequence token (if provided), stop generation
            if (endTokenId.HasValue && nextToken == endTokenId.Value)
                break;

            // prevent posId from exceeding available position embeddings
            if (currentPos >= MaxSequenceLength)
                break;

            // Feed the newly generated token to get logits for the *following* token.
            // Position is currentPos because that's the index this token occupies.
            // The returned logits will predict the token at currentPos+1.
            lastLogits = Forward(nextToken, currentPos, keys, values);
            currentPos++;
        }

        // Return only the newly generated tokens (exclude the original prompt)
        return promptTokens.Skip(tokens.Count).ToList();
    }

    /// <summary>Creates a fresh KV cache for a new document/sample.</summary>
    public List<List<Value>>[] CreateKvCache()
    {
        var cache = new List<List<Value>>[_layerCount];
        for (int i = 0; i < _layerCount; i++)
        {
            cache[i] = [];
        }

        return cache;
    }
}