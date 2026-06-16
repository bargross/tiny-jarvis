using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.ControlFlow;
using Tiny.Jarvis.Training.Generators;
using Tiny.Jarvis.Training.Models;
using Tiny.Jarvis.Training.Optimization;
using Tiny.Jarvis.Training.Orchestrators;
using Tiny.Jarvis.Training.Serializers;
using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.Training.Trainers;

public static class TinyJarvisModelTrainer
{
    public static (TinyJarvisModel, Either<ITokenizer<byte[]>, ITokenizer<string>>) Train(IEnumerable<string> docs, TinyJarvisHyperParameters hyperParams)
    {
        // metrics
        var watch = System.Diagnostics.Stopwatch.StartNew();

        // ── Hyperparameters ──────────────────────────────────────

        var embeddingSize = hyperParams.EmbeddingSize;
        var layerCount = hyperParams.LayerCount; // just one transformer block for speed - try layerCount=2 to see improvement
        var headCount = hyperParams.HeadCount;
        var learningRate = hyperParams.LearningRate;
        var tokenizerStrategy = hyperParams.TokenizerStrategy;
        var optimizerStrategy = hyperParams.OptimizerStrategy;
        var vocabularySize = hyperParams.VocabularySize;
        var numOfMerges = hyperParams.NumOfMerges;
        var maxSequenceLength = hyperParams.MaxSequenceLength;
        var totalNumberOfSteps = hyperParams.MaxNumberOfSteps;
        var maxGradNorm = hyperParams.MaxGradNorm;
        var startTime = DateTime.UtcNow;
        var docList = docs.ToList();

        // ── Dataset and Tokenizer ────────────────────────────────
        Either<ITokenizer<byte[]>, ITokenizer<string>> tokenizerContainer;


        if (hyperParams.LoadTokenizerFile != null)
        {
            if (tokenizerStrategy == TokenizerStrategy.ByteLevelBPE)
                tokenizerContainer = new Either<ITokenizer<byte[]>, ITokenizer<string>>(TokenizerSerializer.Load<byte[]>(hyperParams.LoadTokenizerFile, tokenizerStrategy));

            else tokenizerContainer = new Either<ITokenizer<byte[]>, ITokenizer<string>>(TokenizerSerializer.Load<string>(hyperParams.LoadTokenizerFile, tokenizerStrategy));
        }

        else
        {
            if (tokenizerStrategy == TokenizerStrategy.ByteLevelBPE)
                tokenizerContainer = new Either<ITokenizer<byte[]>, ITokenizer<string>>(TokenizerGenerator.GetTokenizer<byte[]>(tokenizerStrategy, docs, vocabularySize, numOfMerges));

            else switch (tokenizerStrategy)
            {
                case TokenizerStrategy.Chars:
                    tokenizerContainer = new Either<ITokenizer<byte[]>, ITokenizer<string>>(TokenizerGenerator.GetTokenizer<string>(tokenizerStrategy, ["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,|!?-'\""]));
                    break;

                default:
                    tokenizerContainer = new Either<ITokenizer<byte[]>, ITokenizer<string>>(TokenizerGenerator.GetTokenizer<string>(tokenizerStrategy, docList, vocabularySize, numOfMerges));
                    break;
            }
        }

        var existingVocabularySize = tokenizerContainer.IsLeft ? tokenizerContainer.Left.VocabSize : tokenizerContainer.Right.VocabSize;
        Console.WriteLine($"num docs: {docList.Count}");
        Console.WriteLine($"vocab size: {existingVocabularySize}");

        Console.WriteLine($"Training Start Time: {startTime}");

        // ── Model ────────────────────────────────────────────────
        var random = new Random(42);
        var bos = tokenizerContainer.IsLeft ? tokenizerContainer.Left.BOS : tokenizerContainer.Right.BOS;
        var eos = tokenizerContainer.IsLeft ? tokenizerContainer.Left.EOS : tokenizerContainer.Right.EOS;
        var model = null as TinyJarvisModel;
        
        if (hyperParams.LoadModelFile != null)
        {
            if (tokenizerContainer.IsLeft) model = ModelSerializer.Load<byte[]>(hyperParams.LoadModelFile, bos, eos, random);
            else model = ModelSerializer.Load<string>(hyperParams.LoadModelFile, bos, eos, random);
        }
        else model = new TinyJarvisModel(
            embeddingSize,
            headCount,
            layerCount,
            maxSequenceLength,
            random,
            bos,
            eos,
            existingVocabularySize
        );

        Console.WriteLine($"num params: {model.Parameters.Count}");
        Console.WriteLine(Environment.NewLine);

        // ── Optimizer ────────────────────────────────────────

        var momentum = 0.9;
        var weightDecay = 0.0;
        var optimizer = null as IOptimizer;
        if (hyperParams.LoadTokenizerFile != null)
        {
            optimizer.SetParameters(model.Parameters.ToList());

            optimizer = OptimizerSerializer.Load(hyperParams.LoadTokenizerFile, optimizerStrategy, learningRate, totalNumberOfSteps, maxGradNorm, momentum, weightDecay);
        }
        else optimizer = OptimizerGenerator.GetOptimizer(optimizerStrategy, model.Parameters, learningRate, totalNumberOfSteps, momentum, weightDecay, maxGradNorm);
        
        // ── Training Loop ────────────────────────────────────────


        // Running average to smooth out the noisy per-step loss.
        var avgLoss = 0.0;

        // Milestone tracking so we can report the previous milestone's avg loss
        // alongside the current one every 1000 steps.
        var lastMilestoneLoss = 0.0;

        // Reusable buffers for Backward
        var topo = new List<Value>();
        var visited = new HashSet<Value>();
        var backwardStack = new Stack<(Value, int)>();

        // the optimizer keeps track of the steps so we can carry on from that step in the loop
        for (var step = optimizer.CurrentStep; step < totalNumberOfSteps; step++)
        {
            var doc = docList[(step % docList.Count)];

            // the LLM will know that all sequences start with BOS and end with EOS after training, or should.
            var tokens = new List<int> { bos }; // add bos token at the beginning of the sequence to mark the start

            tokens.AddRange(tokenizerContainer.IsLeft ? tokenizerContainer.Left.Encode(doc) : tokenizerContainer.Right.Encode(doc));

            tokens.Add(eos); // mark the end of the sequence

            // Any sequence (word, sentence, etc...) longer than maxSequenceLength - 1 is silently truncated here.
            var maxInputPositions = maxSequenceLength - 1;   // reserve one slot for generation - EOS token
            var tokenCount = Math.Min(tokens.Count - 1, maxInputPositions);

            // ── Forward ──────────────────────────────────────────
            var keys = model.CreateKvCache();
            var values = model.CreateKvCache();

            var loss = new Value(0);

            for (var posId = 0; posId < tokenCount; posId++)
            {
                var currentToken = tokens[posId];
                var nextToken = tokens[posId + 1];

                var logits = model.Forward(currentToken, posId, keys, values);

                // loss is now calculated by CrossEntropyLoss instead of manually calculating via Softmax
                loss += Calculate.CrossEntropyLoss(logits, nextToken);
            }

            loss *= 1.0 / tokenCount;

            // Track running average (exponential moving average with alpha = 0.01)
            avgLoss = step == 0 ? loss.Data : 0.99 * avgLoss + 0.01 * loss.Data;
            
            if (step == 0) lastMilestoneLoss = avgLoss;

            optimizer.ZeroGrad();

            // ── Clear graph buffers ───────────────────────────────
            topo.Clear();
            visited.Clear();
            backwardStack.Clear();

            // ── Backward ──────────────────────────────────────────
            loss.Grad = loss.Grad == default ? maxGradNorm : loss.Grad;
            loss.Backward(topo, visited, backwardStack);

            // ── Update weights ────────────────────────────────────
            optimizer.Step(step);

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

            // save the configuration every 5000 steps
            if ((step + 1) % 5000 == 0)
            {
                ModelSerializer.Save(model, hyperParams);

                if (tokenizerContainer.IsLeft) TokenizerSerializer.Save(tokenizerContainer.Left, hyperParams.SaveTokenizerFile);
                else TokenizerSerializer.Save(tokenizerContainer.Right, hyperParams.SaveTokenizerFile);

                OptimizerSerializer.Save(optimizer, hyperParams.SaveOptimizerFile);
            }

            // For debug during Training, to ensure the model is generating more coherent sentences, so basically to know its learning.
            if ((step + 1) % 500 == 0)
            {
                Console.WriteLine(Environment.NewLine);
                Console.WriteLine("\n--- Testing generation ---");

                var testPrompt = "user: hello assistant:";
                var encodedPromptIds = (tokenizerContainer.IsLeft ? tokenizerContainer.Left.Encode(testPrompt) : tokenizerContainer.Right.Encode(testPrompt)).ToList();

                encodedPromptIds.Insert(0, bos);
                var tokenIds = model.Generate(encodedPromptIds, maxNewTokens: 20, temperature: 0.8);

                var response = tokenizerContainer.IsLeft ? tokenizerContainer.Left.Decode(tokenIds) : tokenizerContainer.Right.Decode(tokenIds);

                Console.WriteLine($"Prompt: {testPrompt}");
                Console.WriteLine($"Response: {response}");
                Console.WriteLine("--- End test ---\n");
            }

            if (avgLoss < 0.1) break; // if avg loss is 0.1, then break out as the model is not learning anything.
        }

        watch.Stop();

        ModelSerializer.Save(model, hyperParams);

        if (tokenizerContainer.IsLeft) TokenizerSerializer.Save(tokenizerContainer.Left, hyperParams.SaveTokenizerFile);
        else TokenizerSerializer.Save(tokenizerContainer.Right, hyperParams.SaveTokenizerFile);

        OptimizerSerializer.Save(optimizer, hyperParams.SaveOptimizerFile);

        var timespan = TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds);
        var secondsDiff = timespan.Seconds;
        var minutesDiff = timespan.Minutes;
        var hoursDiff = timespan.Hours;

        Console.WriteLine($"Start time: {startTime}");
        Console.WriteLine($"End time: {DateTime.UtcNow}");
        Console.WriteLine($"Training was completed in: {hoursDiff}H {minutesDiff}m {secondsDiff}s");

        return (model, tokenizerContainer);
    }
}