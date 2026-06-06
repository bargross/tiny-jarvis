using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.Models;

namespace Tiny.Jarvis.Util;

public static class Helpers
{
    /// <summary>
    /// Matrix-vector multiply. Each row of weights is multiplied element-by-element
    /// with input and summed into a single value.
    /// </summary>
    public static List<Value> Linear(List<Value> input, Value[][] weights) =>
        [.. weights.SelectRow(row => Value.Dot(row, input))];

    /// <summary>
    /// Converts raw scores (logits) into a probability distribution.
    /// </summary>
    public static List<Value> Softmax(List<Value> logits)
    {
        var maxVal = logits.Max(v => v.Data);
        for (int i = 0; i < logits.Count; i++)
            logits[i] = Math.Max(logits[i].Data, 1e-8);

        var exponentials = logits.Select(v => (v - maxVal).Exp()).ToList();
        var total = new Value(0);
        foreach (Value? e in exponentials)
        {
            total += e;
        }

        return [.. exponentials.Select(e => e / total)];
    }

    /// <summary>
    /// Generates a random number from a bell curve (Gaussian/normal distribution)
    /// centered on the mean, with most values falling within 'std' of it.
    /// Uses the Box-Muller transform - turns two uniform random numbers into
    /// one bell-curve random number.
    /// </summary>
    public static double RandomBellCurve(Random rng, double mean, double std)
    {
        double rand1 = 1.0 - rng.NextDouble();
        double rand2 = 1.0 - rng.NextDouble();

        return mean + std * Math.Sqrt(-2.0 * Math.Log(rand1)) * Math.Sin(2.0 * Math.PI * rand2);
    }

    /// <summary>
    /// Creates a matrix of Value objects initialized to small random numbers.
    /// </summary>
    public static Value[][] CreateMatrix(Random rng, int rows, int cols, double std = 0.08)
    {
        var matrix = new Value[rows][];
        for (int i = 0; i < rows; i++)
        {
            matrix[i] = new Value[cols];
            for (int j = 0; j < cols; j++)
            {
                matrix[i][j] = new Value(RandomBellCurve(rng, 0, std));
            }
        }

        return matrix;
    }

    /// <summary>
    /// Rescales a vector so its overall magnitude is close to 1, using the root mean
    /// square of its values. Keeps activations stable across deep networks.
    /// </summary>
    public static List<Value> RmsNorm(List<Value> probabilities)
    {
        var sumSq = new Value(0);
        foreach (Value xi in probabilities)
        {
            sumSq += xi * xi;
        }

        Value ms = sumSq / probabilities.Count;
        Value scale = (ms + 1e-5).Pow(-0.5);

        return probabilities.Select(xi => xi * scale).ToList();
    }

    public static void ApplyTemperature(List<Value> logits, double temperature)
    {
        if (temperature != 1.0)
        {
            for (int i = 0; i < logits.Count; i++)
                logits[i] /= temperature;
        }
    }

    public static int SampleToken(List<Value> logits, double temperature, int topK, double topP)
    {
        // Apply temperature
        ApplyTemperature(logits, temperature);

        // Softmax
        var probs = Softmax(logits);

        // Top‑k
        if (topK > 0 && topK < probs.Count)
        {
            var indices = probs.Select((p, idx) => (p, idx)).OrderByDescending(x => x.p.Data).Take(topK).Select(x => x.idx).ToHashSet();
            for (int i = 0; i < probs.Count; i++)
                if (!indices.Contains(i))
                    probs[i] = 0.0;

            var sum = probs.Sum(x => x.Data);
            for (int i = 0; i < probs.Count; i++)
                probs[i] /= sum;
        }

        // Top‑p
        if (topP < 1.0f && topP > 0)
        {
            var indexed = probs.Select((p, idx) => (p, idx)).OrderByDescending(x => x.p.Data).ToList();
            var cummulitative = 0.0;
            var keep = new HashSet<int>();
            foreach (var item in indexed)
            {
                cummulitative += item.p.Data;
                keep.Add(item.idx);

                if (cummulitative >= topP) break;
            }
            for (int i = 0; i < probs.Count; i++)
                if (!keep.Contains(i))
                    probs[i] = 0;

            var sum = probs.Select(probability => probability.Data).Sum();
            for (int i = 0; i < probs.Count; i++)
                probs[i] /= sum;
        }

        // Sample
        var rand = new Random().NextDouble();
        var cum = 0.0;
        for (int i = 0; i < probs.Count; i++)
        {
            cum += probs[i].Data;
            if (rand < cum)
                return i;
        }

        return probs.Count - 1;
    }

    public static Value CrossEntropyLoss(List<Value> logits, int targetTokenId)
    {
        // Find the maximum logit for numerical stability (prevents overflow in exp)
        var maxLogit = logits.Max(v => v.Data);

        // Compute exponentiated logits (shifted by maxLogit) and their sum
        var exponentiatedLogits = new double[logits.Count];
        var sumOfExponentiatedLogits = 0.0;
        for (int i = 0; i < logits.Count; i++)
        {
            exponentiatedLogits[i] = Math.Exp(logits[i].Data - maxLogit);
            sumOfExponentiatedLogits += exponentiatedLogits[i];
        }

        // Compute softmax probabilities
        var softmaxProbabilities = new double[logits.Count];
        for (int i = 0; i < logits.Count; i++)
            softmaxProbabilities[i] = exponentiatedLogits[i] / sumOfExponentiatedLogits;

        // Extract the probability of the target token and compute negative log loss
        var targetProbability = softmaxProbabilities[targetTokenId];
        var clampedProbability = targetProbability < 1e-8 ? 1e-8 : targetProbability;
        var lossValue = -Math.Log(clampedProbability);

        // Prepare for backpropagation: local gradient of loss w.r.t each logit
        //    d(loss)/d(logit_i) = softmaxProbability_i - (1 if i == target else 0)
        Value[] inputLogitsArray = logits.ToArray();
        var localGradients = new double[logits.Count];
        for (int i = 0; i < logits.Count; i++)
            localGradients[i] = softmaxProbabilities[i] - (i == targetTokenId ? 1.0 : 0.0);

        // Return a single Value node that encapsulates the loss and its local gradients
        return new Value(lossValue, inputLogitsArray, localGradients);
    }
}