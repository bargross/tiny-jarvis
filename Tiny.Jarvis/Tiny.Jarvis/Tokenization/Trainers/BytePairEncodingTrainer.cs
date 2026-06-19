using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.Tokenization.Trainers
{
    internal class BytePairEncodingTrainer
    {
        public (Dictionary<string, int> IdentifierToToken, List<(string Left, string Right)> MergeRules) Train(
            IEnumerable<string> trainingCorpus,
            int numberOfMerges)
        {
            Console.WriteLine($"Starting Byte‑Pair Encoding (BPE) training with {numberOfMerges} merges...");

            // Step 1: Pre‑tokenize and count word frequencies
            var wordFrequencies = new Dictionary<string, int>();
            foreach (var doc in trainingCorpus)
            {
                var tokens = Helpers.PreTokenize(doc);
                foreach (var token in tokens)
                {
                    wordFrequencies.TryGetValue(token, out var count);
                    wordFrequencies[token] = count + 1;
                }
            }

            Console.WriteLine($"Pre‑tokenised {wordFrequencies.Count} unique words.");

            // Remove any empty tokens (just in case)
            wordFrequencies.Remove(string.Empty);

            // Step 2: Initial tokens = all characters of each word
            var currentTokenization = wordFrequencies
                .ToDictionary(
                    pair => pair.Key,
                    pair => pair.Key.Select(character => character.ToString()).ToList()
                );

            var mergeRules = new List<(string Left, string Right)>();

            var reportInterval = Math.Max(1, numberOfMerges / 20); // report ~5% increments

            for (var merge = 0; merge < numberOfMerges; merge++)
            {
                // Count all adjacent pairs across all words
                var pairFrequencies = from wordWithTokens in currentTokenization
                                      from index in Enumerable.Range(0, wordWithTokens.Value.Count - 1)
                                      let left = wordWithTokens.Value[index]
                                      let right = wordWithTokens.Value[index + 1]
                                      let frequency = wordFrequencies[wordWithTokens.Key]
                                      group frequency by (Left: left, Right: right) into pairGroup
                                      select new { Pair = pairGroup.Key, Frequency = pairGroup.Sum() };

                var bestPair = pairFrequencies
                    .OrderByDescending(pair => pair.Frequency)
                    .FirstOrDefault()?.Pair;

                if (bestPair == null)
                {
                    Console.WriteLine($"No more pairs to merge. Stopping early at merge {merge}.");
                    break;
                }

                mergeRules.Add(bestPair.Value);

                // Apply the merge to all word tokenizations
                foreach (var word in currentTokenization.Keys.ToList())
                {
                    var mergedTokens = new List<string>();
                    var tokens = currentTokenization[word];
                    int i = 0;
                    while (i < tokens.Count)
                    {
                        if (i < tokens.Count - 1 && tokens[i] == bestPair.Value.Left && tokens[i + 1] == bestPair.Value.Right)
                        {
                            mergedTokens.Add(bestPair.Value.Left + bestPair.Value.Right);
                            i += 2;
                        }
                        else
                        {
                            mergedTokens.Add(tokens[i]);
                            i++;
                        }
                    }
                    currentTokenization[word] = mergedTokens;
                }

                // Progress reporting
                var currentVocabSize = currentTokenization.Values
                    .SelectMany(tokens => tokens)
                    .Distinct()
                    .Count();

                if ((merge + 1) % reportInterval == 0 || merge == numberOfMerges - 1)
                {
                    var percent = (int)((merge + 1) * 100.0 / numberOfMerges);
                    Console.WriteLine($"Progress: {percent}% | Merge {merge + 1}/{numberOfMerges} | Current vocab size: {currentVocabSize}");
                }
            }

            // Build final vocabulary from all unique tokens appearing after merges
            var allTokens = currentTokenization.Values
                .SelectMany(tokens => tokens)
                .Distinct()
                .OrderBy(token => token)
                .ToList();

            var tokenToIdentifier = allTokens
                .Select((token, index) => new { token, index })
                .ToDictionary(pair => pair.token, pair => pair.index);

            Console.WriteLine($"Training complete. Final vocabulary size: {tokenToIdentifier.Count}");
            Console.WriteLine($"Total merges performed: {mergeRules.Count}");

            return (tokenToIdentifier, mergeRules);
        }
    }
}
