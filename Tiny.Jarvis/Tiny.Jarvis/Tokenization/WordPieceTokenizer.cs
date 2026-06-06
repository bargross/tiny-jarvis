using Tiny.Jarvis.Tokenization.Trainers;

namespace Tiny.Jarvis.Tokenization
{
    public class WordPieceTokenizer: ITokenizer
    {
        private readonly HashSet<string> _tokenVocabulary;
        private readonly Dictionary<int, string> _tokenToIdentifier;
        private readonly Dictionary<string, int> _identifierToToken;

        private readonly int _unknownTokenIdentifier;
        private const string _unknownToken = "[UNK]";
        private const string _bosToken = "[BOS]";
        private const string _endOfSequenceToken = "[EOS]";
        private const string _subwordPrefix = "##";
        private readonly int _vocabularySize;

        public int VocabSize => _vocabularySize;
        public int BOS { get; } // Beginning of Sequence token ID
        public int EOS { get; } // End of Sequence token ID

        public WordPieceTokenizer(IEnumerable<string> docs, int targetVocabularySize = 20)
        {
            // Train WordPiece subword vocabulary (list of strings, no special tokens yet)
            var trainedTokens = new WordPieceTrainer().Train(docs, targetVocabularySize);

            // Prepare a set of all tokens (use a HashSet to avoid duplicates)
            var allTokensSet = new HashSet<string> { _unknownToken, _bosToken, _endOfSequenceToken };
            foreach (var token in trainedTokens)
                allTokensSet.Add(token);   // duplicates (like "[UNK]") are ignored

            // Convert to a sorted list for deterministic ordering
            var allTokensList = allTokensSet.OrderBy(t => t).ToList();

            // Build mapping: token string → integer ID
            var tokenToId = new Dictionary<string, int>();
            for (int i = 0; i < allTokensList.Count; i++)
                tokenToId[allTokensList[i]] = i;

            // Assign your fields exactly as in your original code
            _tokenVocabulary = allTokensSet;                                
            _identifierToToken = tokenToId;                                 // string → int
            _tokenToIdentifier = tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key); // int → string

            _unknownTokenIdentifier = _identifierToToken[_unknownToken];
            BOS = _identifierToToken[_bosToken];
            EOS = _identifierToToken[_endOfSequenceToken];

            _vocabularySize = _tokenToIdentifier.Count;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return text
                .Split(' ')
                .SelectMany(word => SegmentWordByLongestMatch(word))
                .Select(token => _identifierToToken.GetValueOrDefault(token, _unknownTokenIdentifier))
                .ToList();
        }

        public string Decode(IReadOnlyList<int> identifiers)
        {
            var tokens = identifiers
                .Select(id => _tokenToIdentifier.GetValueOrDefault(id, _unknownToken))
                .ToList();

            // WordPiece uses "##" to indicate that a token is attached to the previous one.
            var result = new List<string>();
            foreach (var token in tokens)
            {
                if (token.StartsWith(_subwordPrefix))
                {
                    if (result.Any())
                        result[result.Count - 1] += token.Substring(_subwordPrefix.Length);
                    else
                        result.Add(token.Substring(_subwordPrefix.Length));
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
                yield return _unknownToken;
                foreach (var token in SegmentWordByLongestMatch(remainingText.Substring(1)))
                    yield return token;
            }
        }
    }
}
