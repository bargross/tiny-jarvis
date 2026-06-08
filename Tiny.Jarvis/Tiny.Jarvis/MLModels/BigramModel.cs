using System.Text;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Orchestrators;

namespace Tiny.Jarvis.MLModels;

internal class BigramModel
{
    private readonly double[,] _nextTokenProbs;
    private ITokenizer Tokenizer;

    public BigramModel(List<string> docs, CharacterTokenizer tokenizer)
    {
        int vocabSize = tokenizer.VocabSize;

        // Count transitions: counts[i,j] = how often token j follows token i
        int[,] counts = CountTransitions(docs);

        // Convert counts to probabilities
        _nextTokenProbs = ConvertToProbabilities(counts);
    }

    internal int[,] CountTransitions(List<string> docs)
    {
        Tokenizer = TokenizerGenerator.GetTokenizer(Enums.TokenizerStrategy.Chars, docs);
        int vocabSize = Tokenizer.VocabSize;
        int[,] counts = new int[vocabSize, vocabSize];

        foreach (string doc in docs)
        {
            var tokens = Tokenizer.Encode(doc);
            for (var i = 0; i < tokens.Count - 1; i++)
            {
                counts[tokens[i], tokens[i + 1]]++;
            }
        }

        return counts;
    }
    
    internal double[,] ConvertToProbabilities(int[,] counts)
    {
        int vocabSize = Tokenizer.VocabSize;
        double[,] probs = new double[vocabSize, vocabSize];

        for (var i = 0; i < vocabSize; i++)
        {
            var rowSum = 0.0;
            for (var j = 0; j < vocabSize; j++)
            {
                rowSum += counts[i, j];
            }
            if (rowSum > 0)
            {
                for (var j = 0; j < vocabSize; j++)
                {
                    probs[i, j] = counts[i, j] / rowSum;
                }
            }
        }
        return probs;
    }

    internal string Generate(Random random, int maxLength = 20)
    {
        var token = 0;
        var wordBuilder = new StringBuilder();
        for (var step = 0; step < maxLength; step++)
        {
            var r = random.NextDouble();
            var cumulative = 0.0;
            var next = Tokenizer.VocabSize - 1;
            for (var j = 0; j < Tokenizer.VocabSize; j++)
            {
                cumulative += _nextTokenProbs[token, j];
                if (r <= cumulative)
                {
                    next = j;
                    break;
                }
            }

            if (next == token) break;

            wordBuilder.Append(Tokenizer.Decode(new List<int> { next }));

            token = next;
        }

        return wordBuilder.ToString();
    }

    public void PrintSamples(int count, Random random)
    {
        Console.WriteLine("\n--- bigram samples ---");
        for (var s = 0; s < count; s++)
        {
            Console.WriteLine($"sample {s + 1,2}: {Generate(random)}");
        }
    }

    /// <summary>
    /// Computes the average negative log probability across all documents.
    /// This is the loss baseline that our neural network should beat.
    /// </summary>
    public double ComputeLoss(List<string> docs)
    {
        var totalLoss = 0.0;
        var totalTokens = 0;
        foreach (string doc in docs)
        {
            var tokens = Tokenizer.Encode(doc);
            for (var i = 0; i < tokens.Count - 1; i++)
            {
                var pair = _nextTokenProbs[tokens[i], tokens[i + 1]];
                // A pair never seen during training has p == 0, which would give
                // -log(0) = +infinity. We skip the loss contribution (but still count
                // the token in the denominator), which slightly flatters the baseline.
                if (pair > 0)
                {
                    totalLoss += -Math.Log(pair);
                }

                totalTokens++;
            }
        }

        return totalLoss / totalTokens;
    }
}