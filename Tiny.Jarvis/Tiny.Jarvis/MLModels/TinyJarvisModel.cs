using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.Training.Models;
using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.MLModels;

public class TinyJarvisModel
{
    // Embeddings
    private readonly Scalar[][] _tokenEmbeddings;
    private readonly Scalar[][] _positionEmbeddings;

    // Per‑layer weights
    private readonly List<LayerWeights> _layers;

    // Output head
    private readonly Scalar[][] _outputHead;

    // Still need a flat list for optimiser
    private readonly List<Scalar> _allParameters;

    private readonly int _embeddingSize;
    private readonly int _headCount;
    private readonly int _layerCount;
    private readonly int _headDimension;

    private readonly int _bos;
    private readonly int _eos;

    public Scalar[][] TokenEmbeddings
    {
        get { return _tokenEmbeddings; }
    }
    public Scalar[][] PositionEmbeddings
    {
        get { return _positionEmbeddings; }
    }
    public List<LayerWeights> Layers
    {
        get { return _layers; }
    }

    public Scalar[][] OutputHead 
    {
        get { return _outputHead; } 
    }

    /// <summary>All trainable parameters, flattened into a single list for the optimiser.</summary>
    public int MaxSequenceLength { get; }

    public IReadOnlyList<Scalar> Parameters => BuildParameterList();

    public TinyJarvisModel(
        int embeddingSize,
        int headCount,
        int layerCount,
        int maxSequenceLength,
        Scalar[][] tokenEmbeddings,
        Scalar[][] positionEmbeddings,
        Scalar[][] outputHead,
        List<LayerWeights> layers,
        Random random,
        int bos,
        int eos
    )
    {
        _embeddingSize = embeddingSize;
        _headCount = headCount;
        _layerCount = layerCount;
        _headDimension = embeddingSize / headCount;
        _bos = bos;
        _eos = eos;

        _tokenEmbeddings = tokenEmbeddings;
        _positionEmbeddings = positionEmbeddings;
        _outputHead = outputHead;
        _layers = layers;
    }

    public TinyJarvisModel(
        int embeddingSize,
        int headCount,
        int layerCount,
        int maxSequenceLength,
        Random random,
        int bos,
        int eos,
        int vocabularySize
    ) {
        _embeddingSize = embeddingSize;
        _headCount = headCount;
        _layerCount = layerCount;
        _headDimension = embeddingSize / headCount;
        _bos = bos;
        _eos = eos;

        _tokenEmbeddings = Helpers.CreateMatrix(random, vocabularySize, embeddingSize);
        _positionEmbeddings = Helpers.CreateMatrix(random, maxSequenceLength, embeddingSize);
        _outputHead = Helpers.CreateMatrix(random, vocabularySize, embeddingSize);

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

    private IReadOnlyList<Scalar> BuildParameterList()
    {
        // updates parameters -> might be best to move this to the constructor
        var allParameters = new List<Scalar>();

        allParameters.AddRange((_tokenEmbeddings ?? []).SelectMany(row => row));

        allParameters?.AddRange((_positionEmbeddings ?? []).SelectMany(row => row));

        foreach (var layer in _layers)
        {
            allParameters.AddRange(layer.Query.SelectMany(row => row));
            allParameters.AddRange(layer.Key.SelectMany(row => row));
            allParameters.AddRange(layer.Value.SelectMany(row => row));
            allParameters.AddRange(layer.Output.SelectMany(row => row));
            allParameters.AddRange(layer.FeedForwardOne.SelectMany(row => row));
            allParameters.AddRange(layer.FeedForwardTwo.SelectMany(row => row));
        }

        allParameters.AddRange(_outputHead.SelectMany(row => row));

        return allParameters;
    }

    public List<Scalar> Forward(
        int tokenId,
        int posId,
        List<List<Scalar>>[] keys,
        List<List<Scalar>>[] values
    ) {
        // validate ids
        if (tokenId < 0 || tokenId >= _tokenEmbeddings.Length)
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"tokenId {tokenId} is out of bounds for vocab size {_tokenEmbeddings.Length}");

        if (posId < 0 || posId >= _positionEmbeddings.Length)
            throw new ArgumentOutOfRangeException(nameof(posId), $"posId {posId} is out of bounds for position embedding size {_positionEmbeddings.Length}");

        var tokenEmbedding = _tokenEmbeddings.GetRow(tokenId);
        var positionEmbedding = _positionEmbeddings.GetRow(posId);

        var probabilities = new List<Scalar>();
        for (var i = 0; i < _embeddingSize; i++)
            probabilities.Add(tokenEmbedding[i] + positionEmbedding[i]);

        // Initial RmsNorm: stabilises the embeddings before entering the first block.
        // This isn't standard in all transformer implementations, but gives the
        // residual stream a stable starting magnitude.
        probabilities = Calculate.RmsNorm(probabilities);

        for (var layerIndex = 0; layerIndex < _layerCount; layerIndex++)
        {
            probabilities = AttentionBlock(probabilities, layerIndex, keys, values);
            probabilities = MlpBlock(probabilities, layerIndex);
        }

        probabilities = Calculate.RmsNorm(probabilities);

        // Note: production transformers typically apply a final RmsNorm here
        // before the output projection. We omit it for simplicity.
        return Calculate.Linear(probabilities, _outputHead);
    }

    // Attention wrapped with pre-norm and a residual connection.
    // Mutates keys[layerIndex] and values[layerIndex] by appending the current position's K and V.
    // Attention wrapped with pre-norm and a residual connection.
    // Mutates keys[layerIndex] and values[layerIndex] by appending the current position's K and V.
    private List<Scalar> AttentionBlock(
        List<Scalar> hiddenState,
        int layerIndex,
        List<List<Scalar>>[] keysCache,
        List<List<Scalar>>[] valuesCache)
    {
        // Save input for residual connection later
        var residualConnection = new List<Scalar>(hiddenState);
        hiddenState = Calculate.RmsNorm(hiddenState);

        // Compute Query, Key, Value projections
        var queryProjection = Calculate.Linear(hiddenState, _layers[layerIndex].Query);
        var keyProjection = Calculate.Linear(hiddenState, _layers[layerIndex].Key);
        var valueProjection = Calculate.Linear(hiddenState, _layers[layerIndex].Value);

        // Store current Key and Value in caches (for autoregressive generation)
        keysCache[layerIndex].Add(keyProjection);
        valuesCache[layerIndex].Add(valueProjection);

        // Multi‑head attention: process each head independently
        var concatenatedHeadOutputs = new List<Scalar>();
        for (var headIndex = 0; headIndex < _headCount; headIndex++)
        {
            var headStartIndex = headIndex * _headDimension;
            var queryForHead = queryProjection.GetRange(headStartIndex, _headDimension);

            // Compute scaled dot‑product attention scores against all past keys
            var attentionLogits = new List<Scalar>();
            var cachedPositionsCount = keysCache[layerIndex].Count;
            for (var pastPosition = 0; pastPosition < cachedPositionsCount; pastPosition++)
            {
                var keyForHead = keysCache[layerIndex][pastPosition].GetRange(headStartIndex, _headDimension);
                var dotProduct = new Scalar(0);
                for (var dimension = 0; dimension < _headDimension; dimension++)
                    dotProduct += queryForHead[dimension] * keyForHead[dimension];

                attentionLogits.Add(dotProduct / Math.Sqrt(_headDimension));
            }

            // Convert logits to probabilities
            var attentionWeights = Calculate.Softmax(attentionLogits);

            // Weighted sum of values (this head's output)
            var headOutputValues = new List<Scalar>();
            for (var dimension = 0; dimension < _headDimension; dimension++)
                headOutputValues.Add(new Scalar(0));

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
        var attentionOutput = Calculate.Linear(concatenatedHeadOutputs, _layers[layerIndex].Output);
        for (var dimensionIndex = 0; dimensionIndex < _embeddingSize; dimensionIndex++)
            attentionOutput[dimensionIndex] += residualConnection[dimensionIndex];

        return attentionOutput;
    }

    // Two-layer feed-forward with ReLU, wrapped with pre-norm and a residual connection.
    private List<Scalar> MlpBlock(List<Scalar> probabilities, int layerIndex)
    {
        var xResidual = new List<Scalar>(probabilities);

        probabilities = Calculate.RmsNorm(probabilities);
        probabilities = Calculate.Linear(probabilities, _layers[layerIndex].FeedForwardOne);

        probabilities = probabilities.Select(xi => xi.SiLU()).ToList();

        probabilities = Calculate.Linear(probabilities, _layers[layerIndex].FeedForwardTwo);
        
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
        if (prependBos && (allTokens.Count == 0 || allTokens[0] != _bos))
            allTokens.Insert(0, _bos);
        
        // Reserve at least one slot for generation, but don't go over MaxSequenceLength
        var maxPromptTokens = MaxSequenceLength - 1; // leave room for at least one generated token

        // If the prompt is too long, you might want to truncate from the front, but here we just take the first tokenCount tokens.
        if (allTokens.Count > maxPromptTokens)
            allTokens = allTokens.Take(maxPromptTokens).ToList();

        var tokenCount = allTokens.Count;

        var keys = CreateKvCache();
        var values = CreateKvCache();
        List<Scalar>? lastLogits = null;

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

            if (nextToken == _eos)
                break;

            generated.Add(nextToken);
            allTokens.Add(nextToken);

            // -1 because we need to leave room? Actually we can use up to MaxSequenceLength-1 for feeding the token itself.
            if (currentPos >= MaxSequenceLength) break;

            lastLogits = Forward(nextToken, currentPos, keys, values);
            currentPos++;
        }

        return generated; // or return allTokens.Skip(originalPromptLength)
    }

    /// <summary>Creates a fresh KV cache for a new document/sample.</summary>
    public List<List<Scalar>>[] CreateKvCache()
    {
        var cache = new List<List<Scalar>>[_layerCount];
        for (var i = 0; i < _layerCount; i++)
        {
            cache[i] = [];
        }

        return cache;
    }
}