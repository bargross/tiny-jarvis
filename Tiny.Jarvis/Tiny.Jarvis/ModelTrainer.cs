using System.Collections.Concurrent;
using System.Text;
using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Models;
using Tiny.Jarvis.Optimization;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Util;

namespace Tiny.Jarvis;

internal class ModelTrainer
{
    public static void Run(string docRef, TokenType type)
    {
        // ── Hyperparameters ──────────────────────────────────────
        int embeddingSize = 16;
        int layerCount = 2; // just one transformer block for speed - try layerCount=2 to see improvement
        int maxSequenceLength = 8;
        int totalNumSteps = 10000; // should increase to 100,000 for better results - but 10k is good enough to see learning happening
        int headCount = 4;
        double learningRate = 1e-2;
        var random = new Random(42);
        const double Temperature = 0.5;

        // ── Dataset and Tokenizer ────────────────────────────────
        List<string> docs = Document.LoadDocs(docRef, random);
        var tokenizer = new SimpleTokenizer(docs);
        Console.WriteLine($"num docs: {docs.Count}");
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

        // Reusable buffers for Backward (see Chapter 2's convenience overload for the
        // simpler allocating version - here we hoist them out of the hot loop so 10,000
        // training steps don't allocate 10,000 fresh sets).
        var topo = new List<Value>();
        var visited = new HashSet<Value>();
        var backwardStack = new ConcurrentStack<(Value, int)>();

        // Running average to smooth out the noisy per-step loss.
        double avgLoss = 0.0;
        // Milestone tracking so we can report the previous milestone's avg loss
        // alongside the current one every 1000 steps.
        double lastMilestoneLoss = 0.0;

        var totalWorkerCount = Environment.ProcessorCount;
        var batchContainer = new List<Thread>();
        var startNumSteps = 0;
        var maxNumSteps = totalNumSteps / totalWorkerCount;
        //var backwardStackBatchContainer = new ConcurrentDictionary<int, Stack<(Value, int)>>();
        
        for (int batch = 0; batch < totalWorkerCount; batch++)
        {
            Console.WriteLine($"\n=== batch {batch + 1} ===");
            if (batch > 0)
            {
                startNumSteps = maxNumSteps;
                maxNumSteps += maxNumSteps;
            }

            //var backwardStack = new Stack<(Value, int)>();
            //backwardStackBatchContainer.AddOrUpdate(batch, backwardStack);
            var workerThread = new Thread(() =>
                Train(docs, startNumSteps, maxNumSteps, tokenizer, type, maxSequenceLength, model, avgLoss, lastMilestoneLoss, optimiser, topo, visited, backwardStack));

            batchContainer.Add(workerThread);

            workerThread.IsBackground = true;
            workerThread.Start();
            workerThread.Join(); // <--- stops main thread from continuing until all threads finish
        }

        Console.WriteLine("\n--- inference (new, hallucinated names) ---");
        for (int sampleIdx = 0; sampleIdx < 100; sampleIdx++)
        {
            List<List<Value>>[] keys = model.CreateKvCache();
            List<List<Value>>[] values = model.CreateKvCache();

            int tokenId = tokenizer.Bos;
            var sample = new StringBuilder();

            for (int posId = 0; posId < maxSequenceLength; posId++)
            {
                List<Value> logits = model.Forward(tokenId, posId, keys, values);

                var scaledLogits = logits.Select(l => l / Temperature).ToList();
                List<Value> probabilities = Helpers.Softmax(scaledLogits);

                var r = random.NextDouble();
                var sum = 0.0;
                var nextToken = -1;
                var probabilityValues = probabilities.Select(p => p.Data).ToList();

                // Softmax probabilities can sum to slightly less/more than 1 due to floating point.
                // Rescale r into the actual total so we never fall off the end of the loop.
                var totalProb = probabilityValues.Sum();
                r *= totalProb;

                for (int i = 0; i < probabilityValues.Count; i++)
                {
                    sum += probabilityValues[i];
                    if (r <= sum)
                    {
                        nextToken = i;
                        break;
                    }
                }
                if (nextToken == -1)
                {
                    nextToken = probabilityValues.Count - 1;
                }

                tokenId = nextToken;
                if (tokenId == tokenizer.Bos)
                {
                    break;
                }

                sample.Append(tokenizer.Decode(new List<int> { tokenId }));
            }

            Console.WriteLine($"sample {sampleIdx + 1,2}: {sample}");
        }
    }

    private static void Train(List<string> docs, int startNumStep, int maxNumSteps, SimpleTokenizer tokenizer, TokenType type, int maxSequenceLength, TinyJarvisModel model, double avgLoss, double lastMilestoneLoss, AdamOptimiser optimiser, List<Value> topo, HashSet<Value> visited, ConcurrentStack<(Value, int)> backwardStack)
    {
        for (int step = startNumStep; step < maxNumSteps; step++)
        {
            string doc = docs[step % docs.Count];
            var tokens = new List<int> { tokenizer.Bos };

            tokens.AddRange(tokenizer.Encode(doc));

            tokens.Add(tokenizer.Bos);
            // Any name longer than maxSequenceLength - 1 is silently truncated here.
            int tokenCount = Math.Min(maxSequenceLength, tokens.Count - 1);

            List<List<Value>>[] keys = model.CreateKvCache();
            List<List<Value>>[] values = model.CreateKvCache();

            var losses = new List<Value>();
            for (int posId = 0; posId < tokenCount; posId++)
            {
                List<Value> logits = model.Forward(tokens[posId], posId, keys, values);
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
                    $"step {step + 1,5} / {maxNumSteps,5} | loss {loss.Data:F4} | avg {avgLoss:F4}"
                );
            }

            // Every 1000 steps, print a milestone showing overall progress.
            if ((step + 1) % 1000 == 0)
            {
                Console.WriteLine(
                    $"  [milestone] avg loss: {avgLoss:F4} (was {lastMilestoneLoss:F4})"
                );

                lastMilestoneLoss = avgLoss;
            }
        }
    }
}