using Tiny.Jarvis.Tokenization.Trainers;

namespace Tiny.Jarvis.Tokenization
{
    public class WordPieceTokenizer: ITokenizer
    {
        private readonly HashSet<string> _tokenVocabulary;
        private readonly Dictionary<string, int> _tokenToIdentifier;
        private readonly Dictionary<int, string> _identifierToToken;

        private readonly int _unknownTokenIdentifier;
        private const string UnknownToken = "[UNK]";
        private const string SubwordPrefix = "##";
        private readonly int _vocabularySize;

        public int VocabSize => _vocabularySize;

        public WordPieceTokenizer(List<string> docs, int unknownTokenIdentifier, int targetVocabularySize = 20)
        {
            _vocabularySize = targetVocabularySize;

            var vocabulary = new WordPieceTrainer().Train(docs, targetVocabularySize);

            // 2. Build token-to-ID map (simple assignment)
            var tokenToIdWP = vocabulary
                .OrderBy(t => t)
                .Select((token, index) => new KeyValuePair<string, int>(token, index))
                .ToDictionary();

            // Add [UNK] token
            const string unknown = "[UNK]";
            tokenToIdWP[unknown] = 0;
            vocabulary.Add(unknown);

            _tokenVocabulary = vocabulary;
            _tokenToIdentifier = tokenToIdWP;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _identifierToToken = tokenToIdWP.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            if (!_tokenVocabulary.Contains(UnknownToken))
                _tokenVocabulary.Add(UnknownToken);

            if (!_tokenToIdentifier.ContainsKey(UnknownToken))
                _tokenToIdentifier[UnknownToken] = unknownTokenIdentifier;

            if (!_identifierToToken.ContainsKey(unknownTokenIdentifier))
                _identifierToToken[unknownTokenIdentifier] = UnknownToken;
        }

        public WordPieceTokenizer(
            HashSet<string> tokenVocabulary,
            Dictionary<string, int> tokenToIdentifier,
            int unknownTokenIdentifier)
        {
            _tokenVocabulary = tokenVocabulary;
            _tokenToIdentifier = tokenToIdentifier;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _identifierToToken = tokenToIdentifier.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            if (!_tokenVocabulary.Contains(UnknownToken))
                _tokenVocabulary.Add(UnknownToken);

            if (!_tokenToIdentifier.ContainsKey(UnknownToken))
                _tokenToIdentifier[UnknownToken] = unknownTokenIdentifier;

            if (!_identifierToToken.ContainsKey(unknownTokenIdentifier))
                _identifierToToken[unknownTokenIdentifier] = UnknownToken;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return text
                .Split(' ')
                .SelectMany(word => SegmentWordByLongestMatch(word))
                .Select(token => _tokenToIdentifier.GetValueOrDefault(token, _unknownTokenIdentifier))
                .ToList();
        }

        public string Decode(IReadOnlyList<int> identifiers)
        {
            var tokens = identifiers
                .Select(id => _identifierToToken.GetValueOrDefault(id, UnknownToken))
                .ToList();

            // WordPiece uses "##" to indicate that a token is attached to the previous one.
            var result = new List<string>();
            foreach (var token in tokens)
            {
                if (token.StartsWith(SubwordPrefix))
                {
                    if (result.Any())
                        result[result.Count - 1] += token.Substring(SubwordPrefix.Length);
                    else
                        result.Add(token.Substring(SubwordPrefix.Length));
                }
                else
                {
                    result.Add(token);
                }
            }
            return string.Join("", result);
        }

        private IEnumerable<string> SegmentWordByLongestMatch(string remainingText)
        {
            if (string.IsNullOrEmpty(remainingText))
                yield break;

            var matchingTokens = _tokenVocabulary
                .Where(token => remainingText.StartsWith(token))
                .OrderByDescending(token => token.Length);

            var bestToken = matchingTokens.FirstOrDefault();
            if (bestToken != null)
            {
                yield return bestToken;
                string next = remainingText.Substring(bestToken.Length);
                foreach (var token in SegmentWordByLongestMatch(next))
                    yield return token;
            }
            else
            {
                // No token matches – use unknown token and advance one character
                yield return UnknownToken;
                foreach (var token in SegmentWordByLongestMatch(remainingText.Substring(1)))
                    yield return token;
            }
        }
    }
}
