namespace Tiny.Jarvis.Tokenization.Trainers
{
    internal class UnigramTrainer
    {
        public Dictionary<string, double> Train(IEnumerable<string> trainingCorpus, int targetVocabularySize)
        {
            // Step 1: Create seed vocabulary from all possible substrings (up to some length limit)
            var seedVocabulary = new HashSet<string>();
            foreach (var sentence in trainingCorpus)
            {
                foreach (var word in sentence.Split(' '))
                {
                    for (int start = 0; start < word.Length; start++)
                    {
                        for (int end = start + 1; end <= Math.Min(start + 10, word.Length); end++) // limit substring length
                        {
                            seedVocabulary.Add(word.Substring(start, end - start));
                        }
                    }
                }
            }

            // Step 2: Estimate initial probabilities (via EM algorithm – simplified here)
            var tokenProbabilities = EstimateProbabilities(trainingCorpus, seedVocabulary);

            // Step 3: Prune vocabulary until target size reached
            var currentVocabulary = seedVocabulary.ToHashSet();
            while (currentVocabulary.Count > targetVocabularySize)
            {
                // Compute loss increase if each token is removed
                var lossIncrease = currentVocabulary
                    .AsParallel()
                    .Select(token => new { Token = token, Loss = ComputeLossIncrease(trainingCorpus, currentVocabulary, token, tokenProbabilities) })
                    .OrderBy(item => item.Loss)
                    .ToList();

                // Remove the token with smallest loss increase (prune)
                var tokenToRemove = lossIncrease.First().Token;
                currentVocabulary.Remove(tokenToRemove);

                // Re-estimate probabilities on reduced vocabulary
                tokenProbabilities = EstimateProbabilities(trainingCorpus, currentVocabulary);
            }

            return tokenProbabilities;
        }

        private Dictionary<string, double> EstimateProbabilities(IEnumerable<string> corpus, HashSet<string> vocabulary)
        {
            // Simplified: count token occurrences using best segmentation (Viterbi)
            var tokenCounts = new Dictionary<string, int>();

            foreach (var sentence in corpus)
            {
                foreach (var word in sentence.Split(' '))
                {
                    var segmentation = FindBestSegmentation(word, vocabulary, new Dictionary<string, double>()); // initial uniform
                    foreach (var token in segmentation)
                    {
                        if (!tokenCounts.ContainsKey(token)) tokenCounts[token] = 0;
                        tokenCounts[token]++;
                    }
                }
            }

            int totalTokens = tokenCounts.Values.Sum();
            return tokenCounts.ToDictionary(
                pair => pair.Key,
                pair => (double)pair.Value / totalTokens
            );
        }

        private double ComputeLossIncrease(
            IEnumerable<string> corpus,
            HashSet<string> vocabulary,
            string candidateToRemove,
            Dictionary<string, double> currentProbabilities)
        {
            var reducedVocabulary = new HashSet<string>(vocabulary);
            reducedVocabulary.Remove(candidateToRemove);

            double originalLoss = ComputeTotalLoss(corpus, vocabulary, currentProbabilities);
            double newLoss = ComputeTotalLoss(corpus, reducedVocabulary, currentProbabilities);

            return newLoss - originalLoss;
        }

        private double ComputeTotalLoss(
            IEnumerable<string> corpus,
            HashSet<string> vocabulary,
            Dictionary<string, double> probabilities)
        {
            double totalNegativeLogLikelihood = 0.0;
            foreach (var sentence in corpus)
            {
                foreach (var word in sentence.Split(' '))
                {
                    var segmentation = FindBestSegmentation(word, vocabulary, probabilities);
                    double wordProbability = segmentation
                        .Select(token => probabilities.GetValueOrDefault(token, 1e-10))
                        .Aggregate(1.0, (product, prob) => product * prob);
                    totalNegativeLogLikelihood += -Math.Log(wordProbability);
                }
            }
            return totalNegativeLogLikelihood;
        }

        private List<string> FindBestSegmentation(
            string word,
            HashSet<string> vocabulary,
            Dictionary<string, double> probabilities)
        {
            // Viterbi (same as earlier but uses provided probabilities)
            var best = new Dictionary<int, (double LogProb, List<string> Tokens)> { [0] = (0.0, new List<string>()) };
            for (int end = 1; end <= word.Length; end++)
            {
                var candidates = from start in Enumerable.Range(0, end)
                                 let token = word.Substring(start, end - start)
                                 where vocabulary.Contains(token)
                                 let logProb = best[start].LogProb + Math.Log(probabilities.GetValueOrDefault(token, 1e-10))
                                 select new { LogProb = logProb, Tokens = best[start].Tokens.Concat(new[] { token }).ToList() };
                var bestCandidate = candidates.OrderByDescending(c => c.LogProb).FirstOrDefault();
                if (bestCandidate != null)
                    best[end] = (bestCandidate.LogProb, bestCandidate.Tokens);
                else
                    best[end] = (double.NegativeInfinity, new List<string>()); // fallback
            }
            return best[word.Length].Tokens;
        }
    }
}
