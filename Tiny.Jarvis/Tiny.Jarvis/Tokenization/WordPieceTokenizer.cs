using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Tokenization.Trainers;
using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.Tokenization
{
    public class WordPieceTokenizer: ITokenizer<string>
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

        public Dictionary<string, int> IdentifierToToken => _identifierToToken;
        public int VocabSize => _vocabularySize;
        public int BOS { get; } // Beginning of Sequence token ID
        public int EOS { get; } // End of Sequence token ID
        public int UnknownTokenId => _unknownTokenIdentifier;
        public List<(string Left, string Right)>? MergeRules => null;
        public Dictionary<string, double>? TokenLogProbabilities => null;
        public TokenizerStrategy Type => TokenizerStrategy.WordPiece;


        public WordPieceTokenizer(IEnumerable<string> docs, int targetVocabularySize = 20)
        {
            // Train WordPiece subword vocabulary (list of strings, no special tokens yet)
            var tokensByFrequency = new WordPieceTrainer().Train(docs, targetVocabularySize);

            var sortedTokens = tokensByFrequency.OrderByDescending(kv => kv.Value).Select(kv => kv.Key);

            // Prepare a set of all tokens (use a HashSet to avoid duplicates)
            var allTokensSet = new List<string> { _unknownToken, _bosToken, _endOfSequenceToken };
            foreach (var token in sortedTokens)
                allTokensSet.Add(token);   // duplicates (like "[UNK]") are ignored

            // Build mapping: token string → integer ID
            var tokenToId = new Dictionary<string, int>();
            for (var i = 0; i < allTokensSet.Count; i++)
                tokenToId[allTokensSet[i]] = i;

            // Assign your fields exactly as in your original code
            _tokenVocabulary = allTokensSet.ToHashSet();                                
            _identifierToToken = tokenToId;                                 // string → int
            _tokenToIdentifier = tokenToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key); // int → string

            _unknownTokenIdentifier = _identifierToToken[_unknownToken];

            BOS = _identifierToToken[_bosToken];
            EOS = _identifierToToken[_endOfSequenceToken];

            _vocabularySize = _tokenToIdentifier.Count;
        }

        public WordPieceTokenizer(Dictionary<string, int> identifierToToken, int unknownTokenIdentifier, int bOS, int eOS)
        {
            _tokenVocabulary = identifierToToken.Keys.ToHashSet();
            _tokenToIdentifier = identifierToToken.ToDictionary(x => x.Value, x => x.Key);
            _identifierToToken = identifierToToken;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _vocabularySize = _tokenVocabulary.Count;
            BOS = bOS;
            EOS = eOS;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            // Pre‑tokenize: split into words and punctuation
            var tokens = new List<int>();
            foreach (var segment in Helpers.PreTokenize(text))
            {
                var subwordTokens = SegmentWordIterative(segment);
                foreach (var sub in subwordTokens)
                    tokens.Add(_identifierToToken[sub]);
                
            }

            return tokens;
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
                    
                    else result.Add(token.Substring(_subwordPrefix.Length));
                }

                else result.Add(token);
            }

            return string.Join(" ", result);
        }

        private List<string> SegmentWordIterative(string word)
        {
            var result = new List<string>();
            var position = 0;
            var length = word.Length;
            while (position < length)
            {
                var found = false;
                var maxLen = Math.Min(length - position, 20);
                for (var len = maxLen; len >= 1; len--)
                {
                    var prefix = position == 0 ? "" : _subwordPrefix;
                    var candidate = prefix + word.Substring(position, len);
                    if (_tokenVocabulary.Contains(candidate))
                    {
                        result.Add(candidate);
                        position += len;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    result.Add(_unknownToken);
                    position++; // advance one character
                }
            }

            return result;
        }
    }
}
