using System.Collections.Concurrent;
using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Models;
using Tiny.Jarvis.Optimization;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Util;

namespace Tiny.Jarvis.Training.Trainers;

public static class TinyJarvisModelTrainer
{
    public static (TinyJarvisModel, ITokenizer) Train(IEnumerable<string> docs, TokenizerStrategy strategy)
    {        
        // ── Hyperparameters ──────────────────────────────────────

        int embeddingSize = 16;
        int layerCount = 2; // just one transformer block for speed - try layerCount=2 to see improvement
        int maxSequenceLength = 8;
        int totalNumSteps = 10000; // should increase to 100,000 for better results - but 10k is good enough to see learning happening
        int headCount = 4;
        var learningRate = 1e-2;
        var random = new Random(42);

        // ── Dataset and Tokenizer ────────────────────────────────

        var tokenizer = SmartTokenizerGenerator.GetTokenizer(strategy, docs);

        Console.WriteLine($"num docs: {docs.Count()}");
        Console.WriteLine($"vocab size: {tokenizer.VocabSize}");

        // ── Model ────────────────────────────────────────────────

        var model = new TinyJarvisModel(
            tokenizer.VocabSize,
            embeddingSize,
            headCount,
            layerCount,
            maxSequenceLength,
            random
        );

        Console.WriteLine($"num params: {model.Parameters.Count}");

        // ── Training Loop ────────────────────────────────────────

        var optimiser = new AdamOptimiser(model.Parameters, learningRate, totalNumSteps);

        // Running average to smooth out the noisy per-step loss.
        var avgLoss = 0.0;

        // Milestone tracking so we can report the previous milestone's avg loss
        // alongside the current one every 1000 steps.
        var lastMilestoneLoss = 0.0;

        var totalWorkerCount = Environment.ProcessorCount;
        var batchContainer = new List<Task>();
        var maxNumSteps = totalNumSteps / totalWorkerCount;

        // this computes the start and end step for each worker in advance so that they can be passed to the parallel loop without needing to capture the loop variable (which would cause issues with closures)
        int[] startNumStepsPerBatch = new int[totalWorkerCount];
        int[] endNumStepsPerBatch = new int[totalWorkerCount];
        for (int i = 0; i < totalWorkerCount; i++)
        {
            startNumStepsPerBatch[i] = i * maxNumSteps;
            endNumStepsPerBatch[i] = (i + 1) * maxNumSteps;
        }

        endNumStepsPerBatch[totalWorkerCount - 1] = totalNumSteps;

        Parallel.For(0, totalWorkerCount, batch =>
        {
            // Reusable buffers for Backward (see Chapter 2's convenience overload for the
            var topo = new List<Value>();
            var visited = new HashSet<Value>();
            var backwardStack = new ConcurrentStack<(Value, int)>();

            var startNumSteps = startNumStepsPerBatch[batch];
            var endNumSteps = endNumStepsPerBatch[batch];

            Console.WriteLine($"batch {batch + 1} has started.");

            TrainModel(batch, docs, startNumSteps, endNumSteps, tokenizer, maxSequenceLength, model, avgLoss, lastMilestoneLoss, optimiser, topo, visited, backwardStack);
        });

        return (model, tokenizer);
    }

    private static void TrainModel(int batch, IEnumerable<string> docs, int startNumStep, int maxNumSteps, ITokenizer tokenizer, int maxSequenceLength, TinyJarvisModel model, double avgLoss, double lastMilestoneLoss, AdamOptimiser optimiser, List<Value> topo, HashSet<Value> visited, ConcurrentStack<(Value, int)> backwardStack)
    {
        for (int step = startNumStep; step < maxNumSteps; step++)
        {
            var doc = docs.ElementAt(step % docs.Count());
            var tokens = new List<int> { tokenizer.Bos };

            tokens.AddRange(tokenizer.Encode(doc));

            tokens.Add(tokenizer.Bos); // mark the end of the sequence

            // Any name longer than maxSequenceLength - 1 is silently truncated here.
            int tokenCount = Math.Min(maxSequenceLength, tokens.Count - 1);

            List<List<Value>>[] keys = model.CreateKvCache();
            List<List<Value>>[] values = model.CreateKvCache();

            var losses = new List<Value>();
            for (int posId = 0; posId < tokenCount; posId++)
            {
                var token = tokens[posId];
                List<Value> logits = model.Forward(token, posId, keys, values);
                List<Value> probabilities = Helpers.Softmax(logits);

                var index = posId + 1;
                if (index < 0) continue;

                losses.Add(-probabilities[tokens[posId + 1]].Log());
            }

            var loss = new Value(0);
            foreach (Value l in losses)
            {
                loss += l;
            }

            loss *= 1.0 / tokenCount;

            // Track running average (exponential moving average with alpha = 0.01)
            avgLoss = step == 0 ? loss.Data : 0.99 * avgLoss + 0.01 * loss.Data;
            if (step == 0)
            {
                lastMilestoneLoss = avgLoss;
            }

            optimiser.ZeroGrad();

            topo.Clear();
            visited.Clear();
            backwardStack.Clear();

            loss.Backward(topo, visited, backwardStack);

            optimiser.Step(step);

            if (step == 0 || (step + 1) % 100 == 0)
            {
                Console.WriteLine(
                    $"Batch: {batch}, step: {step + 1,5} / {maxNumSteps,5} | loss {loss.Data:F4} | avg {avgLoss:F4}"
                );
            }

            // Every 1000 steps, print a milestone showing overall progress.
            if ((step + 1) % 1000 == 0)
            {
                Console.WriteLine(
                    $"Batch: {batch} [milestone], avg. loss: {avgLoss:F4} (was {lastMilestoneLoss:F4})"
                );

                lastMilestoneLoss = avgLoss;
            }
        }
    }
}