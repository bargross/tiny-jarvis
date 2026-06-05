namespace Tiny.Jarvis.Tokenization.Trainers
{
    internal class BytePairEncodingTrainer
    {
        public (Dictionary<string, int> TokenToIdentifier, List<(string Left, string Right)> MergeRules) Train(
            string trainingCorpus,
            int numberOfMerges)
        {
            // Step 1: Split into words and count frequencies
            var wordFrequencies = trainingCorpus
                .Split(' ')
                .GroupBy(word => word)
                .ToDictionary(group => group.Key, group => group.Count());

            // Step 2: Initial tokens = all characters
            var currentTokenization = wordFrequencies
                .ToDictionary(
                    pair => pair.Key,
                    pair => pair.Key.Select(character => character.ToString()).ToList()
                );

            var mergeRules = new List<(string Left, string Right)>();

            for (int merge = 0; merge < numberOfMerges; merge++)
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

                if (bestPair == null) break;

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
            }

            // Build vocabulary from all unique tokens appearing after merges
            var allTokens = currentTokenization.Values
                .SelectMany(tokens => tokens)
                .Distinct()
                .OrderBy(token => token)
                .ToList();

            var tokenToIdentifier = allTokens
                .Select((token, index) => new { token, index })
                .ToDictionary(pair => pair.token, pair => pair.index);

            return (tokenToIdentifier, mergeRules);
        }
    }
}
