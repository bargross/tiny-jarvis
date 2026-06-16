using System.Text.RegularExpressions;
using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.Tokenization.Trainers
{
    public class UnigramTrainer
    {
        private const int MaxSeedVocabSize = 5000;
        private const int MaxSubstringLength = 5;

        /// <summary>
        /// Trains a Unigram subword vocabulary using EM and frequency‑based pruning.
        /// Returns a dictionary of token -> log probability.
        /// </summary>
        public Dictionary<string, double> Train(IEnumerable<string> trainingCorpus, int targetVocabularySize)
        {
            Console.WriteLine("Beginning Unigram tokenizer training...");

            // ── Pre‑tokenize: split into words and punctuation using the same logic as WordPiece ──
            var wordFreq = new Dictionary<string, int>();
            foreach (var sentence in trainingCorpus)
            {
                foreach (var token in Helpers.PreTokenize(sentence))  // punctuation separated
                {
                    wordFreq.TryGetValue(token, out int cnt);
                    wordFreq[token] = cnt + 1;
                }
            }

            // Build seed vocabulary from character n‑grams (frequent substrings) ──
            var ngramCounts = new Dictionary<string, int>();
            foreach (var word in wordFreq.Keys)
            {
                for (var len = 1; len <= Math.Min(MaxSubstringLength, word.Length); len++)
                {
                    for (var start = 0; start <= word.Length - len; start++)
                    {
                        var sub = word.Substring(start, len);

                        ngramCounts.TryGetValue(sub, out int cnt);
                        ngramCounts[sub] = cnt + wordFreq[word];
                    }
                }
            }

            var seedVocab = ngramCounts.OrderByDescending(kv => kv.Value)
                                       .Take(MaxSeedVocabSize)
                                       .Select(kv => kv.Key)
                                       .ToHashSet();

            // Ensure we include all single characters (they are already there from n‑grams)
            foreach (var character in wordFreq.Keys.SelectMany(w => w).Distinct())
                seedVocab.Add(character.ToString());

            Console.WriteLine($"Seed vocabulary size: {seedVocab.Count}");

            // Estimate initial probabilities via EM (run a few iterations) ──
            var tokenProbs = seedVocab.ToDictionary(t => t, t => 1.0 / seedVocab.Count);
            for (var emIter = 0; emIter < 5; emIter++)
            {
                var newTokenCounts = new Dictionary<string, int>();
                foreach (var (word, freq) in wordFreq)
                {
                    var segmentation = FindBestSegmentation(word, seedVocab, tokenProbs);
                    if (segmentation == null) continue; // should not happen

                    foreach (var token in segmentation)
                    {
                        newTokenCounts.TryGetValue(token, out int cnt);
                        newTokenCounts[token] = cnt + freq;
                    }
                }

                var total = newTokenCounts.Values.Sum();
                if (total == 0) break;
                tokenProbs = newTokenCounts.ToDictionary(kv => kv.Key, kv => (double)kv.Value / total);
            }

            // Prune to target size by removing the least frequent tokens ──
            var finalVocab = tokenProbs.OrderByDescending(kv => kv.Value)
                                       .Take(targetVocabularySize)
                                       .Select(kv => kv.Key)
                                       .ToHashSet();

            // Re‑estimate final probabilities on the pruned vocabulary (one more EM iteration)
            var finalProbs = new Dictionary<string, double>();
            var finalCounts = new Dictionary<string, int>();
            foreach (var (word, freq) in wordFreq)
            {
                var segmentation = FindBestSegmentation(word, finalVocab, tokenProbs);
                if (segmentation == null) continue;

                foreach (var token in segmentation)
                {
                    finalCounts.TryGetValue(token, out int cnt);
                    finalCounts[token] = cnt + freq;
                }
            }

            var finalTotal = finalCounts.Values.Sum();
            finalProbs = finalCounts.ToDictionary(kv => kv.Key, kv => Math.Log((double)kv.Value / finalTotal));

            Console.WriteLine($"Training complete. Final vocabulary size: {finalVocab.Count}");
            return finalProbs;
        }

        /// <summary>
        /// Viterbi segmentation that returns the most probable token sequence,
        /// using log probabilities to avoid underflow.
        /// </summary>
        private List<string> FindBestSegmentation(
            string word,
            HashSet<string> vocabulary,
            Dictionary<string, double> probs)   // probs are raw probabilities (not logs)
        {
            // Convert to log probabilities once for efficiency
            var logProbs = new Dictionary<string, double>();
            foreach (var kv in probs)
                logProbs[kv.Key] = Math.Log(kv.Value);

            var n = word.Length;
            var dp = new double[n + 1];
            var bestPrev = new int[n + 1];

            for (var i = 1; i <= n; i++)
                dp[i] = double.NegativeInfinity;

            dp[0] = 0;
            bestPrev[0] = -1;

            for (var end = 1; end <= n; end++)
            {
                for (var start = 0; start < end; start++)
                {
                    var token = word.Substring(start, end - start);
                    if (vocabulary.Contains(token))
                    {
                        var logProb = logProbs.GetValueOrDefault(token, -20.0); // penalty for unknown
                        var candidate = dp[start] + logProb;
                        if (candidate > dp[end])
                        {
                            dp[end] = candidate;
                            bestPrev[end] = start;
                        }
                    }
                }

                // Fallback: if no segmentation, treat as single character
                if (double.IsNegativeInfinity(dp[end]))
                {
                    var ch = word[end - 1].ToString();
                    var logProb = logProbs.GetValueOrDefault(ch, -20.0);
                    dp[end] = dp[end - 1] + logProb;
                    bestPrev[end] = end - 1;
                }
            }

            // Reconstruct segmentation
            var tokens = new List<string>();
            var pos = n;
            while (pos > 0)
            {
                var prev = bestPrev[pos];
                tokens.Insert(0, word.Substring(prev, pos - prev));
                pos = prev;
            }

            return tokens;
        }
    }
}