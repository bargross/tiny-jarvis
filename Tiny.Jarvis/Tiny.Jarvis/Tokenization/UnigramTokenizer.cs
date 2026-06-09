using Tiny.Jarvis.Tokenization.Trainers;

namespace Tiny.Jarvis.Tokenization
{
    public class UnigramTokenizer: ITokenizer
    {
        private readonly Dictionary<string, double> _tokenLogProbabilities;
        private readonly Dictionary<int, string> _tokenToIdentifier;
        private readonly Dictionary<string, int> _identifierToToken;

        private readonly int _unknownTokenIdentifier;
        private const string _unknownToken = "[UNK]";
        private const string _bosToken = "[BOS]";
        private const string _endOfSequenceToken = "[EOS]";
        private const double _unknownTokenLogProbability = -100.0;
        private readonly int _vocabularySize;

        public int VocabSize => _vocabularySize;
        public int BOS { get; } // Beginning of Sequence token ID
        public int EOS { get; } // End of Sequence token ID

        public UnigramTokenizer(IEnumerable<string> docs, int targetVocabularySize = 20)
        {
            // Train Unigram model to get token → log probability dictionary
            var tokenLogProbabilities = new UnigramTrainer().Train(docs, targetVocabularySize);

            // Add special tokens to the probability map (assign small log probs)
            const double defaultLogProb = -15.0;  // low probability

            if (!tokenLogProbabilities.ContainsKey(_unknownToken))
                tokenLogProbabilities[_unknownToken] = defaultLogProb;

            if (!tokenLogProbabilities.ContainsKey(_bosToken))
                tokenLogProbabilities[_bosToken] = defaultLogProb;

            if (!tokenLogProbabilities.ContainsKey(_endOfSequenceToken))
                tokenLogProbabilities[_endOfSequenceToken] = defaultLogProb;

            // Build deterministic list of all tokens (special tokens first, then sorted)
            var allTokens = new List<string> { _unknownToken, _bosToken, _endOfSequenceToken };
            allTokens.AddRange(tokenLogProbabilities.Keys
                .Where(t => t != _unknownToken && t != _bosToken && t != _endOfSequenceToken)
                .OrderBy(t => t));

            // Assign consecutive IDs (UNK=0, BOS=1, then rest)
            var identifierToToken = new Dictionary<string, int>();
            for (var i = 0; i < allTokens.Count; i++)
                identifierToToken[allTokens[i]] = i;

            var tokenToIdentifier = identifierToToken.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            _tokenLogProbabilities = tokenLogProbabilities;
            _identifierToToken = identifierToToken;
            _tokenToIdentifier = tokenToIdentifier;
            _unknownTokenIdentifier = identifierToToken[_unknownToken];
            _vocabularySize = identifierToToken.Count;

            BOS = identifierToToken[_bosToken];
            EOS = identifierToToken[_endOfSequenceToken];
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return text
                .Split(' ')
                .SelectMany(word => FindBestSegmentation(word))
                .Select(token => _identifierToToken.GetValueOrDefault(token, _unknownTokenIdentifier))
                .ToList();
        }

        public string Decode(IReadOnlyList<int> identifiers)
        {
            var tokens = identifiers
                .Select(id => _tokenToIdentifier.GetValueOrDefault(id, _unknownToken))
                .ToList();

            // Unigram typically concatenates tokens; spaces are either separate tokens or implied.
            // We'll simply join them.
            return string.Concat(tokens);
        }

        private IEnumerable<string> FindBestSegmentation(string word)
        {
            // prefixes[pos] = (logProbability, tokenList)
            var prefixes = new Dictionary<int, (double LogProbability, List<string> Tokens)>
            {
                [0] = (0.0, new List<string>())
            };

            for (var end = 1; end <= word.Length; end++)
            {
                var candidates = from start in Enumerable.Range(0, end)
                                 let token = word.Substring(start, end - start)
                                 let logProb = prefixes[start].LogProbability + GetLogProbability(token)
                                 select new { LogProbability = logProb, Tokens = prefixes[start].Tokens.Concat(new[] { token }).ToList() };

                var best = candidates.OrderByDescending(c => c.LogProbability).FirstOrDefault();
                if (best != null)
                    prefixes[end] = (best.LogProbability, best.Tokens);
                else
                    prefixes[end] = (double.NegativeInfinity, new List<string>());
            }

            return prefixes[word.Length].Tokens;
        }

        private double GetLogProbability(string token)
        {
            return _tokenLogProbabilities.GetValueOrDefault(token, _unknownTokenLogProbability);
        }
    }
}
