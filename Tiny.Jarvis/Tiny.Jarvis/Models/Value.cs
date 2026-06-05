namespace Tiny.Jarvis.Models;

public class Value(double data, Value[]? inputs = null, double[]? localGrads = null)
{
    public double Data = data;
    public double Grad; // filled in during the backward pass (Chapter 2)

    private readonly Value[]? _inputs = inputs;
    private readonly double[]? _localGrads = localGrads;

    public Value[] Inputs => _inputs ?? Array.Empty<Value>();
    public double[] LocalGrads => _localGrads ?? Array.Empty<double>();

    // --- Arithmetic operations ---
    // Each operation records three things: the result, the inputs, and the local gradients.
    // The local gradients are explained in the "Verifying Local Gradients" section below.

    public static Value operator +(Value a, Value b) => new(a.Data + b.Data, [a, b], [1.0, 1.0]);

    public static Value operator *(Value a, Value b) =>
        new(a.Data * b.Data, [a, b], [b.Data, a.Data]);

    // NaN if Data is negative and n is fractional.
    public Value Pow(double n) => new(Math.Pow(Data, n), [this], [n * Math.Pow(Data, n - 1)]);

    // -Infinity if Data == 0, NaN if Data < 0. If you see NaN propagating through
    // training, a softmax probability collapsed to 0 and this is usually the entry point.
    public Value Log() => new(Math.Log(Data), [this], [1.0 / Data]);

    public Value Exp() => new(Math.Exp(Data), [this], [Math.Exp(Data)]);

    // ReLU: passes positive values through unchanged, blocks negatives entirely.
    public Value Relu() => new(Math.Max(0, Data), [this], [Data > 0 ? 1.0 : 0.0]);

    /// <summary>
    /// Dot product of two lists of Values. The result is a single Value, and the local gradients are
    /// computed with respect to each input Value.
    /// </summary>
    /// <param name="a">The first list of Values.</param>
    /// <param name="b">The second list of Values.</param>
    /// <returns>A single Value representing the dot product of the two lists.</returns>
    public static Value Dot(IEnumerable<Value> a, IEnumerable<Value> b)
    {
        var result = new Value(0);
        for (int i = 0; i < a.Count(); i++)
        {
            result += a.ElementAt(i) * b.ElementAt(i);
        }

        return result;
    }

    // --- Convenience overloads ---
    public static Value operator +(Value a, double b) => a + new Value(b);

    public static Value operator *(Value a, double b) => a * new Value(b);

    public static Value operator -(Value a) => a * -1;

    public static Value operator -(Value a, double b) => a + (-b);

    public static Value operator /(Value a, Value b) => a * b.Pow(-1);

    public static Value operator /(Value a, double b) => a * Math.Pow(b, -1);

    public static implicit operator Value(double d) => new(d, [], []);

    public override string ToString() => $"Value(data={Data})";
}