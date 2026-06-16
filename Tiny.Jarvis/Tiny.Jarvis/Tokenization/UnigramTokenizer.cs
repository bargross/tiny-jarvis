using Tiny.Jarvis.Tokenization.Trainers;
using Tiny.Jarvis.Training.Util;

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

        public Dictionary<string, int> IdentifierToToken => _identifierToToken;
        public int VocabSize => _vocabularySize;
        public int BOS { get; } // Beginning of Sequence token ID
        public int EOS { get; } // End of Sequence token ID
        public int UnknownTokenId => _unknownTokenIdentifier;

        public List<(string Left, string Right)>? MergeRules => null;
        public Dictionary<string, double>? TokenLogProbabilities => TokenLogProbabilities;

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

        public UnigramTokenizer(Dictionary<string, double> tokenLogProbabilities, Dictionary<string, int> identifierToToken, int unknownTokenIdentifier, int bOS, int eOS)
        {
            _tokenLogProbabilities = tokenLogProbabilities;
            _tokenToIdentifier = identifierToToken.ToDictionary(x => x.Value, x => x.Key);
            _identifierToToken = identifierToToken;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _vocabularySize = _tokenLogProbabilities.Count;
            BOS = bOS;
            EOS = eOS;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return Helpers.PreTokenize(text)
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
            return string.Join(" ", tokens);
        }

        private IEnumerable<string> FindBestSegmentation(string word)
        {
            var n = word.Length;
            var bestScore = new double[n + 1];
            var bestStart = new int[n + 1];

            Array.Fill(bestScore, double.NegativeInfinity);
            bestScore[0] = 0.0;
            bestStart[0] = -1;

            for (var end = 1; end <= n; end++)
            {
                for (var start = 0; start < end; start++)
                {
                    if (bestScore[start] == double.NegativeInfinity) continue;

                    var token = word.Substring(start, end - start);
                    var score = bestScore[start] + GetLogProbability(token);

                    if (score > bestScore[end])
                    {
                        bestScore[end] = score;
                        bestStart[end] = start;
                    }
                }
            }

            // Trace back
            var tokens = new List<string>();
            var pos = n;
            while (pos > 0)
            {
                var start = bestStart[pos];

                tokens.Add(word.Substring(start, pos - start));
                pos = start;
            }

            tokens.Reverse();

            return tokens;
        }

        private double GetLogProbability(string token) => _tokenLogProbabilities.GetValueOrDefault(token, _unknownTokenLogProbability);
        
    }
}
