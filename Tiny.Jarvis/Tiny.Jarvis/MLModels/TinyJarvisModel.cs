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

    private Value[][] TokenEmbeddings => _stateDict["wte"];
    private Value[][] PositionEmbeddings => _stateDict["wpe"];
    private Value[][] OutputProjection => _stateDict["lm_head"];

    /// <summary>All trainable parameters, flattened into a single list for the optimiser.</summary>
    public List<Value> Parameters { get; }

    public TinyJarvisModel(
        int vocabSize,
        int embeddingSize,
        int headCount,
        int layerCount,
        int maxSequenceLength,
        Random random
    )
    {
        _embeddingSize = embeddingSize;
        _headCount = headCount;
        _layerCount = layerCount;
        _headDimension = embeddingSize / headCount;

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

        // Dictionary<TKey,TValue> enumeration order is not guaranteed by the spec.
        // In .NET Core+ it preserves insertion order in practice, so Adam's momentum[]/squaredGradAvg[]
        // line up across runs - but if that implementation detail ever changes, switch
        // to a List<KeyValuePair<string, ...>> to make the order explicit.
        Parameters = [.. _stateDict.Values.SelectMany(mat => mat).SelectMany(row => row)];
    }

    public List<Value> Forward(
        int tokenId,
        int posId,
        List<List<Value>>[] keys,
        List<List<Value>>[] values
    )
    {
        var tokenEmbedding = TokenEmbeddings.GetRow(tokenId);
        var positionEmbedding = PositionEmbeddings.GetRow(posId);

        var x = new List<Value>();
        for (int i = 0; i < _embeddingSize; i++)
        {
            x.Add(tokenEmbedding[i] + positionEmbedding[i]);
        }

        // Initial RmsNorm: stabilises the embeddings before entering the first block.
        // This isn't standard in all transformer implementations, but gives the
        // residual stream a stable starting magnitude.
        x = Helpers.RmsNorm(x);

        for (int layerIndex = 0; layerIndex < _layerCount; layerIndex++)
        {
            x = AttentionBlock(x, layerIndex, keys, values);
            x = MlpBlock(x, layerIndex);
        }

        // Note: production transformers typically apply a final RmsNorm here
        // before the output projection. We omit it for simplicity.
        return Helpers.Linear(x, OutputProjection);
    }

    // Attention wrapped with pre-norm and a residual connection.
    // Mutates keys[layerIndex] and values[layerIndex] by appending the current position's K and V.
    private List<Value> AttentionBlock(
        List<Value> x,
        int layerIndex,
        List<List<Value>>[] keys,
        List<List<Value>>[] values
    )
    {
        var xResidual = new List<Value>(x);
        x = Helpers.RmsNorm(x);

        List<Value> query = Helpers.Linear(x, _stateDict[$"layer{layerIndex}.attn_wq"]);
        List<Value> key = Helpers.Linear(x, _stateDict[$"layer{layerIndex}.attn_wk"]);
        List<Value> value = Helpers.Linear(x, _stateDict[$"layer{layerIndex}.attn_wv"]);

        keys[layerIndex].Add(key);
        values[layerIndex].Add(value);

        var concatenatedHeads = new List<Value>();
        for (int h = 0; h < _headCount; h++)
        {
            int headStart = h * _headDimension;
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

            for (int t = 0; t < cachedCount; t++)
            {
                List<Value> valueForHead = values[layerIndex]
                    [t]
                    .GetRange(headStart, _headDimension);
                Value w = attentionWeights[t];
                for (int j = 0; j < _headDimension; j++)
                {
                    headOutput[j] += w * valueForHead[j];
                }
            }
            concatenatedHeads.AddRange(headOutput);
        }

        x = Helpers.Linear(concatenatedHeads, _stateDict[$"layer{layerIndex}.attn_wo"]);
        for (int i = 0; i < _embeddingSize; i++)
        {
            x[i] += xResidual[i];
        }

        return x;
    }

    // Two-layer feed-forward with ReLU, wrapped with pre-norm and a residual connection.
    private List<Value> MlpBlock(List<Value> x, int layerIndex)
    {
        var xResidual = new List<Value>(x);

        x = Helpers.RmsNorm(x);
        x = Helpers.Linear(x, _stateDict[$"layer{layerIndex}.mlp_fc1"]);

        x = [.. x.Select(xi => xi.Relu())];

        x = Helpers.Linear(x, _stateDict[$"layer{layerIndex}.mlp_fc2"]);
        for (int i = 0; i < _embeddingSize; i++)
        {
            x[i] += xResidual[i];
        }

        return x;
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