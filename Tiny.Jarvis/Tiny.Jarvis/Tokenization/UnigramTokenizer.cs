using Tiny.Jarvis.Tokenization.Trainers;

namespace Tiny.Jarvis.Tokenization
{
    public class UnigramTokenizer: ITokenizer
    {
        private readonly Dictionary<string, double> _tokenLogProbabilities;
        private readonly Dictionary<string, int> _tokenToIdentifier;
        private readonly Dictionary<int, string> _identifierToToken;
        private readonly int _unknownTokenIdentifier;
        private const string UnknownToken = "[UNK]";
        private const double UnknownTokenLogProbability = -100.0;
        private readonly int _vocabularySize;

        public int VocabSize => _vocabularySize;

        public UnigramTokenizer(List<string> docs, int unknownTokenIdentifier, int targetVocabularySize = 20)
        {
            _vocabularySize = targetVocabularySize;

            var tokenLogProbabilities = new UnigramTrainer().Train(docs, targetVocabularySize);

            // 2. Build token-to-ID map (simple assignment)
            int nextId = 1;
            var tokenToIdU = tokenLogProbabilities.Keys
                .OrderBy(t => t)
                .Select((token, index) => new KeyValuePair<string, int>(token, index))
                .ToDictionary();

            int unknownId = 0;
            tokenToIdU["[UNK]"] = unknownId;

            _tokenLogProbabilities = tokenLogProbabilities;
            _tokenToIdentifier = tokenToIdU;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _identifierToToken = tokenToIdU.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            if (!_tokenToIdentifier.ContainsKey(UnknownToken))
                _tokenToIdentifier[UnknownToken] = unknownTokenIdentifier;

            if (!_identifierToToken.ContainsKey(unknownTokenIdentifier))
                _identifierToToken[unknownTokenIdentifier] = UnknownToken;

            if (!_tokenLogProbabilities.ContainsKey(UnknownToken))
                _tokenLogProbabilities[UnknownToken] = UnknownTokenLogProbability;
        }

        public UnigramTokenizer(
            Dictionary<string, double> tokenLogProbabilities,
            Dictionary<string, int> tokenToIdentifier,
            int unknownTokenIdentifier)
        {
            _tokenLogProbabilities = tokenLogProbabilities;
            _tokenToIdentifier = tokenToIdentifier;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _identifierToToken = tokenToIdentifier.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            if (!_tokenToIdentifier.ContainsKey(UnknownToken))
                _tokenToIdentifier[UnknownToken] = unknownTokenIdentifier;

            if (!_identifierToToken.ContainsKey(unknownTokenIdentifier))
                _identifierToToken[unknownTokenIdentifier] = UnknownToken;

            if (!_tokenLogProbabilities.ContainsKey(UnknownToken))
                _tokenLogProbabilities[UnknownToken] = UnknownTokenLogProbability;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return text
                .Split(' ')
                .SelectMany(word => FindBestSegmentation(word))
                .Select(token => _tokenToIdentifier.GetValueOrDefault(token, _unknownTokenIdentifier))
                .ToList();
        }

        public string Decode(IReadOnlyList<int> identifiers)
        {
            var tokens = identifiers
                .Select(id => _identifierToToken.GetValueOrDefault(id, UnknownToken))
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

            for (int end = 1; end <= word.Length; end++)
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
            return _tokenLogProbabilities.GetValueOrDefault(token, UnknownTokenLogProbability);
        }
    }
}
