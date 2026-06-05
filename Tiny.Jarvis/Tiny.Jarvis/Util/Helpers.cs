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
        double maxVal = logits.Max(v => v.Data);
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
    public static List<Value> RmsNorm(List<Value> x)
    {
        var sumSq = new Value(0);
        foreach (Value xi in x)
        {
            sumSq += xi * xi;
        }

        Value ms = sumSq / x.Count;
        Value scale = (ms + 1e-5).Pow(-0.5);
        return [.. x.Select(xi => xi * scale)];
    }
}