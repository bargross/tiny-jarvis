using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Models;
using Tiny.Jarvis.Optimization;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Orchestrators;
using Tiny.Jarvis.Util;

namespace Tiny.Jarvis.Training.Trainers;

public static class TinyJarvisModelTrainer
{
    public static (TinyJarvisModel, ITokenizer) Train(IEnumerable<string> docs, TokenizerStrategy strategy, int maxSequenceLength = 32, int totalNumberOfSteps = 10000, int vocabularySize = 50)
    {
        // metrics
        var watch = System.Diagnostics.Stopwatch.StartNew();

        // ── Hyperparameters ──────────────────────────────────────

        var embeddingSize = 64;
        var layerCount = 4; // just one transformer block for speed - try layerCount=2 to see improvement
        var headCount = 4;
        var learningRate = 0.001;
        var startTime = DateTime.UtcNow;

        // ── Dataset and Tokenizer ────────────────────────────────
        var tokenizer = null as ITokenizer;
        switch (strategy)
        {
            case TokenizerStrategy.Chars:
                tokenizer = TokenizerGenerator.GetTokenizer(strategy, ["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\""], vocabularySize);
                break;

            default:
                tokenizer = TokenizerGenerator.GetTokenizer(strategy, docs, vocabularySize);
                break;
        }

        Console.WriteLine($"num docs: {docs.Count()}");
        Console.WriteLine($"vocab size: {tokenizer.VocabSize}");

        // ── Model ────────────────────────────────────────────────

        var model = new TinyJarvisModel(
            tokenizer.VocabSize,
            embeddingSize,
            headCount,
            layerCount,
            maxSequenceLength,
            tokenizer.BOS, // give it the bos token to use later when using Generate
            tokenizer.EOS, // eos tokent o signal end of function
            new Random(42)
        );

        Console.WriteLine($"num params: {model.Parameters.Count}");
        Console.WriteLine(Environment.NewLine);

        // ── Training Loop ────────────────────────────────────────

        var optimiser = new AdamOptimiser(model.Parameters, learningRate, totalNumberOfSteps);

        // Running average to smooth out the noisy per-step loss.
        var avgLoss = 0.0;

        // Milestone tracking so we can report the previous milestone's avg loss
        // alongside the current one every 1000 steps.
        var lastMilestoneLoss = 0.0;

        // Reusable buffers for Backward
        var topo = new List<Value>();
        var visited = new HashSet<Value>();
        var backwardStack = new Stack<(Value, int)>();

        for (var step = 0; step < totalNumberOfSteps; step++)
        {
            var doc = docs.ElementAt(step % docs.Count());
            var tokens = new List<int> { tokenizer.BOS };

            tokens.AddRange(tokenizer.Encode(doc));

            tokens.Add(tokenizer.EOS); // mark the end of the sequence

            // Any name longer than maxSequenceLength - 1 is silently truncated here.
            var maxInputPositions = maxSequenceLength - 1;   // reserve one slot for generation
            var tokenCount = Math.Min(tokens.Count - 1, maxInputPositions);

            var keys = model.CreateKvCache();
            var values = model.CreateKvCache();

            var loss = new Value(0);

            for (var posId = 0; posId < tokenCount; posId++)
            {
                var token = tokens[posId];
                var nextToken = tokens[posId + 1];

                var logits = model.Forward(token, posId, keys, values);

                // loss is now calculated by CrossEntropyLoss instead of manually calculating via Softmax
                loss += Helpers.CrossEntropyLoss(logits, nextToken);
            }

            loss *= 1.0 / tokenCount;

            // Track running average (exponential moving average with alpha = 0.01)
            avgLoss = step == 0 ? loss.Data : 0.99 * avgLoss + 0.01 * loss.Data;
            
            if (step == 0) lastMilestoneLoss = avgLoss;

            optimiser.ZeroGrad();

            topo.Clear();
            visited.Clear();
            backwardStack.Clear();

            loss.Modify((_, grad) => grad = grad == default ? 1.0 : grad);
            loss.Backward(topo, visited, backwardStack);

            optimiser.Step(step);
            var percentage = (step + 1) * 100.0 / totalNumberOfSteps;

            if (step == 0 || (step + 1) % 100 == 0)
            {
                Console.Write($"\rTraining: {percentage:F2}% complete  | ");
                Console.WriteLine(
                    $"step: {step + 1,5} / {totalNumberOfSteps,5} | loss {loss.Data:F4} | avg {avgLoss:F4}"
                );
            }

            if (step % 10 == 0)
                Console.WriteLine($"Training Current Step: {step} / {totalNumberOfSteps}");

                // Every 1000 steps, print a milestone showing overall progress.
            if ((step + 1) % 1000 == 0)
            {
                Console.WriteLine($"[milestone], avg. loss: {avgLoss:F4} (was {lastMilestoneLoss:F4})");
                Console.WriteLine(Environment.NewLine);

                lastMilestoneLoss = avgLoss;
            }

            if (avgLoss < 1e-5) break;

        }

        watch.Stop();

        var secondsDiff = watch.ElapsedMilliseconds / 1000;
        var minutesDiff = secondsDiff / 60;
        var hoursDiff = minutesDiff / 60;

        Console.WriteLine($"Start time: {startTime}");
        Console.WriteLine($"End time: {DateTime.UtcNow}");
        Console.WriteLine($"Training was completed in: {hoursDiff}H {minutesDiff}m {secondsDiff}s");

        return (model, tokenizer);
    }
}