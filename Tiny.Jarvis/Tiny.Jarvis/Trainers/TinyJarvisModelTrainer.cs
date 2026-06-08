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
        // ── Hyperparameters ──────────────────────────────────────

        int embeddingSize = 16;
        int layerCount = 2; // just one transformer block for speed - try layerCount=2 to see improvement
        int headCount = 4;
        var learningRate = 0.01;
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

        for (int step = 0; step < totalNumberOfSteps; step++)
        {
            var doc = docs.ElementAt(step % docs.Count());
            var tokens = new List<int> { tokenizer.BOS };

            tokens.AddRange(tokenizer.Encode(doc));

            tokens.Add(tokenizer.EOS); // mark the end of the sequence

            // Any name longer than maxSequenceLength - 1 is silently truncated here.
            int tokenCount = Math.Min(maxSequenceLength - 1, tokens.Count);

            List<List<Value>>[] keys = model.CreateKvCache();
            List<List<Value>>[] values = model.CreateKvCache();

            var loss = new Value(0);

            for (int posId = 0; posId < tokenCount; posId++)
            {
                var token = tokens[posId];

                List<Value> logits = model.Forward(token, posId, keys, values);
                List<Value> probabilities = Helpers.Softmax(logits);

                if (posId < 0) continue;

                loss += Helpers.CrossEntropyLoss(logits, token);
            }

            loss *= 1.0 / tokenCount;

            // Track running average (exponential moving average with alpha = 0.01)
            avgLoss = step == 0 ? loss.Data : 0.99 * avgLoss + 0.01 * loss.Data;
            
            if (step == 0) lastMilestoneLoss = avgLoss;

            optimiser.ZeroGrad();

            topo.Clear();
            visited.Clear();
            backwardStack.Clear();

            loss.SetDefaultGrad(1.0);
            loss.Backward(topo, visited, backwardStack);

            // Debug
            //if (step == 0 || step == 100)
            //{
            //    var someParam = model.Parameters[0];
            //    Console.WriteLine($"Step {step}, param Data: {someParam.Data:F6}, Grad: {someParam.Grad:F6}");
            //}

            optimiser.Step(step);
            var percentage = (step + 1) * 100.0 / totalNumberOfSteps;

            if (step == 0 || (step + 1) % 100 == 0)
            {
                Console.Write($"\rTraining: {percentage:F2}% complete  | ");
                Console.WriteLine(
                    $"step: {step + 1,5} / {totalNumberOfSteps,5} | loss {loss.Data:F4} | avg {avgLoss:F4}"
                );
            }

            // Every 1000 steps, print a milestone showing overall progress.
            if ((step + 1) % 1000 == 0)
            {
                Console.WriteLine($"[milestone], avg. loss: {avgLoss:F4} (was {lastMilestoneLoss:F4})");
                Console.WriteLine(Environment.NewLine);

                lastMilestoneLoss = avgLoss;
            }

        }

        var endTime = DateTime.UtcNow;
        var hoursDiff = endTime.Hour - startTime.Hour;
        var minutesDiff = endTime.Minute - startTime.Minute;
        var secondsDiff = endTime.Second - startTime.Second;

        Console.WriteLine($"Start time: {startTime}");
        Console.WriteLine($"End of training at: {endTime}");
        Console.WriteLine($"Training was completed in: {hoursDiff}H {minutesDiff}m {secondsDiff}s");

        return (model, tokenizer);
    }
}