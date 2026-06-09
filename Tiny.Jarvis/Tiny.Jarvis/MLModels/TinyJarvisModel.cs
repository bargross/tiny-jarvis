using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.Util;
using Tiny.Jarvis.Training.Models;
using Tiny.Jarvis.Tokenization;

namespace Tiny.Jarvis.MLModels;

public class TinyJarvisModel
{
    // Embeddings
    private readonly Value[][] _tokenEmbeddings;
    private readonly Value[][] _positionEmbeddings;

    // Per‑layer weights
    private readonly List<LayerWeights> _layers;

    // Output head
    private readonly Value[][] _outputHead;

    // Still need a flat list for optimiser
    private List<Value> _allParameters;

    private readonly int _embeddingSize;
    private readonly int _headCount;
    private readonly int _layerCount;
    private readonly int _headDimension;

    private readonly ITokenizer _tokenizer;

    /// <summary>All trainable parameters, flattened into a single list for the optimiser.</summary>
    public int MaxSequenceLength { get; }
    //public int TotalTokenEmbeddings { get; }
    //public int TotalPositionEmbeddings { get; }
    public IReadOnlyList<Value> Parameters
    {
        get
        {
            if (_allParameters == null)
            {
                _allParameters = _tokenEmbeddings?.SelectMany(row => row).ToList();
                _allParameters.AddRange(_positionEmbeddings.SelectMany(row => row));

                foreach (var layer in _layers)
                {
                    _allParameters.AddRange(layer.Query.SelectMany(row => row));
                    _allParameters.AddRange(layer.Key.SelectMany(row => row));
                    _allParameters.AddRange(layer.Value.SelectMany(row => row));
                    _allParameters.AddRange(layer.Output.SelectMany(row => row));
                    _allParameters.AddRange(layer.FeedForwardOne.SelectMany(row => row));
                    _allParameters.AddRange(layer.FeedForwardTwo.SelectMany(row => row));
                }

                _allParameters.AddRange(_outputHead.SelectMany(row => row));
            }
            return _allParameters;
        }
    }

    public TinyJarvisModel(
        int embeddingSize,
        int headCount,
        int layerCount,
        int maxSequenceLength,
        Random random,
        ITokenizer tokenizer
    )
    {
        _embeddingSize = embeddingSize;
        _headCount = headCount;
        _layerCount = layerCount;
        _headDimension = embeddingSize / headCount;
        _tokenizer = tokenizer;

        _tokenEmbeddings = Helpers.CreateMatrix(random, _tokenizer.VocabSize, embeddingSize);
        _positionEmbeddings = Helpers.CreateMatrix(random, maxSequenceLength, embeddingSize);
        _outputHead = Helpers.CreateMatrix(random, _tokenizer.VocabSize, embeddingSize);

        _layers = new List<LayerWeights>();
        for (int i = 0; i < layerCount; i++)
        {
            _layers.Add(new LayerWeights
            {
                Query = Helpers.CreateMatrix(random, embeddingSize, embeddingSize),
                Key = Helpers.CreateMatrix(random, embeddingSize, embeddingSize),
                Value = Helpers.CreateMatrix(random, embeddingSize, embeddingSize),
                Output = Helpers.CreateMatrix(random, embeddingSize, embeddingSize),
                FeedForwardOne = Helpers.CreateMatrix(random, 4 * embeddingSize, embeddingSize),
                FeedForwardTwo = Helpers.CreateMatrix(random, embeddingSize, 4 * embeddingSize)
            });
        }

        MaxSequenceLength = maxSequenceLength;
    }

    public List<Value> Forward(
        int tokenId,
        int posId,
        List<List<Value>>[] keys,
        List<List<Value>>[] values
    ) {
        // validate ids
        if (tokenId < 0 || tokenId >= _tokenEmbeddings.Length)
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"tokenId {tokenId} is out of bounds for vocab size {_tokenEmbeddings.Length}");

        if (posId < 0 || posId >= _positionEmbeddings.Length)
            throw new ArgumentOutOfRangeException(nameof(posId), $"posId {posId} is out of bounds for position embedding size {_positionEmbeddings.Length}");

        // use ids directly (remove the -1 adjustment)
        var tokenEmbedding = _tokenEmbeddings.GetRow(tokenId);
        var positionEmbedding = _positionEmbeddings.GetRow(posId);

        var probabilities = new List<Value>();
        for (var i = 0; i < _embeddingSize; i++)
            probabilities.Add(tokenEmbedding[i] + positionEmbedding[i]);

        // Initial RmsNorm: stabilises the embeddings before entering the first block.
        // This isn't standard in all transformer implementations, but gives the
        // residual stream a stable starting magnitude.
        probabilities = Helpers.RmsNorm(probabilities);

        for (var layerIndex = 0; layerIndex < _layerCount; layerIndex++)
        {
            probabilities = AttentionBlock(probabilities, layerIndex, keys, values);
            probabilities = MlpBlock(probabilities, layerIndex);
        }

        // Note: production transformers typically apply a final RmsNorm here
        // before the output projection. We omit it for simplicity.
        return Helpers.Linear(probabilities, _outputHead);
    }

    // Attention wrapped with pre-norm and a residual connection.
    // Mutates keys[layerIndex] and values[layerIndex] by appending the current position's K and V.
    private List<Value> AttentionBlock(
        List<Value> hiddenState,
        int layerIndex,
        List<List<Value>>[] keysCache,
        List<List<Value>>[] valuesCache)
    {
        // Save input for residual connection later
        var residualConnection = new List<Value>(hiddenState);
        hiddenState = Helpers.RmsNorm(hiddenState);

        // Compute Query, Key, Value projections
        var queryProjection = Helpers.Linear(hiddenState, _layers[layerIndex].Query);
        var keyProjection = Helpers.Linear(hiddenState, _layers[layerIndex].Key);
        var valueProjection = Helpers.Linear(hiddenState, _layers[layerIndex].Value);

        // Store current Key and Value in caches (for autoregressive generation)
        keysCache[layerIndex].Add(keyProjection);
        valuesCache[layerIndex].Add(valueProjection);

        // Multi‑head attention: process each head independently
        var concatenatedHeadOutputs = new List<Value>();
        for (var headIndex = 0; headIndex < _headCount; headIndex++)
        {
            var headStartIndex = headIndex * _headDimension;
            var queryForHead = queryProjection.GetRange(headStartIndex, _headDimension);

            // Compute scaled dot‑product attention scores against all past keys
            var attentionLogits = new List<Value>();
            var cachedPositionsCount = keysCache[layerIndex].Count;
            for (var pastPosition = 0; pastPosition < cachedPositionsCount; pastPosition++)
            {
                var keyForHead = keysCache[layerIndex][pastPosition].GetRange(headStartIndex, _headDimension);
                var dotProduct = new Value(0);
                for (var dimension = 0; dimension < _headDimension; dimension++)
                    dotProduct += queryForHead[dimension] * keyForHead[dimension];

                attentionLogits.Add(dotProduct / Math.Sqrt(_headDimension));
            }

            // Convert logits to probabilities
            var attentionWeights = Helpers.Softmax(attentionLogits);

            // Weighted sum of values (this head's output)
            var headOutputValues = new List<Value>();
            for (var dimension = 0; dimension < _headDimension; dimension++)
                headOutputValues.Add(new Value(0));

            for (var pastPosition = 0; pastPosition < cachedPositionsCount; pastPosition++)
            {
                var valueForHead = valuesCache[layerIndex][pastPosition].GetRange(headStartIndex, _headDimension);
                var weight = attentionWeights[pastPosition];
                for (var dimension = 0; dimension < _headDimension; dimension++)
                    headOutputValues[dimension] += weight * valueForHead[dimension];
            }

            concatenatedHeadOutputs.AddRange(headOutputValues);
        }

        // Final linear projection and residual connection
        var attentionOutput = Helpers.Linear(concatenatedHeadOutputs, _layers[layerIndex].Output);
        for (var dimensionIndex = 0; dimensionIndex < _embeddingSize; dimensionIndex++)
            attentionOutput[dimensionIndex] += residualConnection[dimensionIndex];

        return attentionOutput;
    }

    // Two-layer feed-forward with ReLU, wrapped with pre-norm and a residual connection.
    private List<Value> MlpBlock(List<Value> probabilities, int layerIndex)
    {
        var xResidual = new List<Value>(probabilities);

        probabilities = Helpers.RmsNorm(probabilities);
        probabilities = Helpers.Linear(probabilities, _layers[layerIndex].FeedForwardOne);

        probabilities = probabilities.Select(xi => xi.Relu()).ToList();

        probabilities = Helpers.Linear(probabilities, _layers[layerIndex].FeedForwardTwo);
        
        for (var embeddingIndex = 0; embeddingIndex < _embeddingSize; embeddingIndex++)
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
    bool prependBos = true)
    {
        // Copy the prompt to a mutable list and optionally prepend BOS
        var allTokens = new List<int>(tokens);
        if (prependBos && (allTokens.Count == 0 || allTokens[0] != _tokenizer.BOS))
        {
            allTokens.Insert(0, _tokenizer.BOS);
        }

        // Reserve at least one slot for generation, but don't go over MaxSequenceLength
        var maxPromptTokens = MaxSequenceLength - 1; // leave room for at least one generated token
        int tokenCount = Math.Min(tokens.Count, maxPromptTokens);

        // If the prompt is too long, you might want to truncate from the front, but here we just take the first tokenCount tokens.
        if (tokenCount < allTokens.Count)
        {
            // Optional: log a warning that prompt was truncated
            allTokens = allTokens.Take(tokenCount).ToList();
        }

        var keys = CreateKvCache();
        var values = CreateKvCache();
        List<Value>? lastLogits = null;

        // Feed prompt tokens
        for (var pos = 0; pos < tokenCount; pos++)
            lastLogits = Forward(allTokens[pos], pos, keys, values);

        var currentPos = tokenCount;
        var generated = new List<int>();

        for (var step = 0; step < maxNewTokens; step++)
        {
            // Ensure we have logits (should never be null if tokenCount > 0)
            if (lastLogits == null) break;

            var nextToken = Helpers.SampleToken(lastLogits, temperature, topK, topP);

            generated.Add(nextToken);
            allTokens.Add(nextToken);

            if (nextToken == _tokenizer.EOS)
                break;

            // -1 because we need to leave room? Actually we can use up to MaxSequenceLength-1 for feeding the token itself.
            if (currentPos >= maxPromptTokens) 
            {
                break;
            }

            lastLogits = Forward(nextToken, currentPos, keys, values);
            currentPos++;
        }

        return generated; // or return allTokens.Skip(originalPromptLength)
    }

    /// <summary>Creates a fresh KV cache for a new document/sample.</summary>
    public List<List<Value>>[] CreateKvCache()
    {
        var cache = new List<List<Value>>[_layerCount];
        for (var i = 0; i < _layerCount; i++)
        {
            cache[i] = [];
        }

        return cache;
    }
}