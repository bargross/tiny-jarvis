using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Util;

public static class Helpers
{
    /// <summary>
    /// Creates a matrix of Value objects initialized with small random numbers (normal distribution).
    /// </summary>
    public static Value[][] CreateMatrix(Random randomGenerator, int rowCount, int columnCount, double standardDeviation = 0.08)
    {
        var matrix = new Value[rowCount][];
        for (var rowIndex = 0; rowIndex < rowCount; rowIndex++)
        {
            matrix[rowIndex] = new Value[columnCount];

            for (var columnIndex = 0; columnIndex < columnCount; columnIndex++)
                matrix[rowIndex][columnIndex] = new Value(Calculate.RandomBellCurve(randomGenerator, 0, standardDeviation));
        }

        return matrix;
    }

    /// <summary>
    /// Samples a token ID from the logits using temperature, top‑k, and top‑p (nucleus) sampling.
    /// </summary>
    public static int SampleToken(List<Value> logits, double temperature, int topK, double topP)
    {
        Calculate.ApplyTemperature(logits, temperature);

        var probabilities = Calculate.Softmax(logits);

        // ---- Top‑k filtering ----
        if (topK > 0 && topK < probabilities.Count)
        {
            var topKIndices = probabilities
                .Select((probability, index) => (probability, index))
                .OrderByDescending(pair => pair.probability.Data)
                .Take(topK)
                .Select(pair => pair.index)
                .ToHashSet();

            for (var index = 0; index < probabilities.Count; index++)
                if (!topKIndices.Contains(index))
                    probabilities[index] = 0.0;   // Set to double (converted to Value implicitly)

            var sumOfProbs = probabilities.Sum(prob => prob.Data);
            for (var index = 0; index < probabilities.Count; index++)
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

            for (var index = 0; index < probabilities.Count; index++)
                if (!indicesToKeep.Contains(index))
                    probabilities[index] = 0.0;

            var totalKeptProbability = probabilities.Sum(prob => prob.Data);
            for (var index = 0; index < probabilities.Count; index++)
                probabilities[index] /= totalKeptProbability;
        }

        // ---- Sample from the final distribution ----
        var random = new Random();
        var randomValue = random.NextDouble();
        var accumulatedProbability = 0.0;

        for (var tokenId = 0; tokenId < probabilities.Count; tokenId++)
        {
            accumulatedProbability += probabilities[tokenId].Data;

            if (randomValue < accumulatedProbability)
                return tokenId;
        }

        return probabilities.Count - 1;   // Fallback (should never reach)
    }
}