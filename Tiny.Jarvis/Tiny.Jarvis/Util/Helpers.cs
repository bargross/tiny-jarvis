using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.Models;

namespace Tiny.Jarvis.Util;

public static class Helpers
{
    /// <summary>
    /// Matrix-vector multiply. Each row of weights is multiplied element‑wise with input and summed.
    /// </summary>
    public static List<Value> Linear(List<Value> input, Value[][] weights) =>
        [.. weights.SelectRow(row => Value.Dot(row, input))];

    /// <summary>
    /// Converts raw logits into a probability distribution using softmax.
    /// </summary>
    public static List<Value> Softmax(List<Value> logits)
    {
        if (logits == null || logits.Count == 0)
            return new List<Value>();

        var maxLogit = logits.Max(value => value.Data);
        var exponentials = logits.Select(value => (value - maxLogit).Exp()).ToList();
        var sumOfExponentials = new Value(0);

        foreach (Value exponential in exponentials)
            sumOfExponentials += exponential;

        return exponentials
            .Select(exponential => exponential / sumOfExponentials)
            .ToList();
    }

    /// <summary>
    /// Generates a normally distributed (Gaussian) random number using the Box‑Muller transform.
    /// </summary>
    public static double RandomBellCurve(Random randomGenerator, double mean, double standardDeviation)
    {
        var uniformA = 1.0 - randomGenerator.NextDouble();
        var uniformB = 1.0 - randomGenerator.NextDouble();

        var z = Math.Sqrt(-2.0 * Math.Log(uniformA)) * Math.Sin(2.0 * Math.PI * uniformB);

        return mean + standardDeviation * z;
    }

    /// <summary>
    /// Creates a matrix of Value objects initialized with small random numbers (normal distribution).
    /// </summary>
    public static Value[][] CreateMatrix(Random randomGenerator, int rowCount, int columnCount, double standardDeviation = 0.08)
    {
        var matrix = new Value[rowCount][];
        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++)
        {
            matrix[rowIndex] = new Value[columnCount];

            for (int columnIndex = 0; columnIndex < columnCount; columnIndex++)
                matrix[rowIndex][columnIndex] = new Value(RandomBellCurve(randomGenerator, 0, standardDeviation));
        }

        return matrix;
    }

    /// <summary>
    /// RMSNorm: rescales a vector so its root‑mean‑square is close to 1. Keeps activations stable.
    /// </summary>
    public static List<Value> RmsNorm(List<Value> activations)
    {
        var sumOfSquares = new Value(0);
        foreach (Value activation in activations)
            sumOfSquares += activation * activation;

        Value meanSquare = sumOfSquares / activations.Count;
        Value scale = (meanSquare + 1e-5).Pow(-0.5);

        return activations.Select(activation => activation * scale).ToList();
    }

    /// <summary>
    /// In‑place temperature scaling of logits.
    /// </summary>
    public static void ApplyTemperature(List<Value> logits, double temperature)
    {
        if (Math.Abs(temperature - 1.0) > 1e-8) 
            for (int index = 0; index < logits.Count; index++)
                logits[index] /= temperature;   // Divides Value by double – works if operator/ is defined
    }

    /// <summary>
    /// Samples a token ID from the logits using temperature, top‑k, and top‑p (nucleus) sampling.
    /// </summary>
    public static int SampleToken(List<Value> logits, double temperature, int topK, double topP)
    {
        ApplyTemperature(logits, temperature);

        var probabilities = Softmax(logits);

        // ---- Top‑k filtering ----
        if (topK > 0 && topK < probabilities.Count)
        {
            var topKIndices = probabilities
                .Select((probability, index) => (probability, index))
                .OrderByDescending(pair => pair.probability.Data)
                .Take(topK)
                .Select(pair => pair.index)
                .ToHashSet();

            for (int index = 0; index < probabilities.Count; index++)
                if (!topKIndices.Contains(index))
                    probabilities[index] = 0.0;   // Set to double (converted to Value implicitly)

            var sumOfProbs = probabilities.Sum(prob => prob.Data);
            for (int index = 0; index < probabilities.Count; index++)
                probabilities[index] /= sumOfProbs;
        }

        // ---- Top‑p (nucleus) filtering ----
        if (topP < 1.0 && topP > 0.0)
        {
            var sorted = probabilities
                .Select((prob, idx) => (prob, idx))
                .OrderByDescending(pair => pair.prob.Data)
                .ToList();

            var cumulativeProbability = 0.0;
            var indicesToKeep = new HashSet<int>();
            foreach (var item in sorted)
            {
                cumulativeProbability += item.prob.Data;
                indicesToKeep.Add(item.idx);

                if (cumulativeProbability >= topP)
                    break;
            }

            for (int index = 0; index < probabilities.Count; index++)
                if (!indicesToKeep.Contains(index))
                    probabilities[index] = 0.0;

            var totalKeptProbability = probabilities.Sum(prob => prob.Data);
            for (int index = 0; index < probabilities.Count; index++)
                probabilities[index] /= totalKeptProbability;
        }

        // ---- Sample from the final distribution ----
        var random = new Random();
        var randomValue = random.NextDouble();
        var accumulatedProbability = 0.0;

        for (int tokenId = 0; tokenId < probabilities.Count; tokenId++)
        {
            accumulatedProbability += probabilities[tokenId].Data;

            if (randomValue < accumulatedProbability)
                return tokenId;
        }

        return probabilities.Count - 1;   // Fallback (should never reach)
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
        var localGradients = new double[logits.Count];
        for (int i = 0; i < logits.Count; i++)
            localGradients[i] = softmaxProbabilities[i] - (i == targetTokenId ? 1.0 : 0.0);

        // Return a single Value node that encapsulates the loss and its local gradients
        return new Value(lossValue, logits.ToArray(), localGradients);
    }
}