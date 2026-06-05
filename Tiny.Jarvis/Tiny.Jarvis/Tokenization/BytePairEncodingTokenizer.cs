using Tiny.Jarvis.Tokenization.Trainers;

namespace Tiny.Jarvis.Tokenization
{
    public class BytePairEncodingTokenizer: ITokenizer
    {
        private readonly Dictionary<string, int> _identifierToToken;
        private readonly Dictionary<int, string> _tokenToIdentifier;
        private readonly List<(string Left, string Right)> _mergeRules;
        private readonly int _unknownTokenIdentifier;
        private readonly string _unknownTokenString = "[UNK]";
       
        public int VocabSize => _tokenToIdentifier.Count;
        public int Bos { get; } // Beginning of Sequence token ID

        public BytePairEncodingTokenizer(IEnumerable<string> docs, int unknownTokenIdentifier = -1, int numberOfMerges = 5)
        {
            var tokenizerTrainingData = docs.Select(text => new BytePairEncodingTrainer().Train(text, numberOfMerges)); // hard to track the token to identifier mapping across multiple documents, so we will merge them together in the next iteration

            _tokenToIdentifier = tokenizerTrainingData.SelectMany(kv => kv.IdentifierToToken)
                .GroupBy(kvp => kvp.Key)
                .ToDictionary(kvp => kvp.First().Value, kvp => kvp.First().Key);

            _identifierToToken = _tokenToIdentifier.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            _mergeRules = tokenizerTrainingData.SelectMany(kv => kv.MergeRules).ToList();
            _unknownTokenIdentifier = unknownTokenIdentifier;
            Bos = _tokenToIdentifier.Count;

            // Ensure unknown token exists in reverse map
            if (!_tokenToIdentifier.ContainsKey(unknownTokenIdentifier))
                _tokenToIdentifier[unknownTokenIdentifier] = _unknownTokenString;
        }

        public BytePairEncodingTokenizer(
            Dictionary<string, int> identifierToToken,
            List<(string Left, string Right)> mergeRules,
            int unknownTokenIdentifier)
        {
            _tokenToIdentifier = identifierToToken.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            _mergeRules = mergeRules;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _identifierToToken = identifierToToken;

            // Ensure unknown token exists in reverse map
            if (!_tokenToIdentifier.ContainsKey(unknownTokenIdentifier))
                _tokenToIdentifier[unknownTokenIdentifier] = _unknownTokenString;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return text
                .Split(' ')
                .SelectMany(word => TokenizeWord(word))
                .Select(token => _identifierToToken.GetValueOrDefault(token, _unknownTokenIdentifier))
                .ToList();
        }

        public string Decode(IReadOnlyList<int> identifiers)
        {
            var tokens = identifiers
                .Select(id => _tokenToIdentifier.GetValueOrDefault(id, _unknownTokenString))
                .ToList();

            // BPE typically uses a special token ("_") for spaces; here we assume spaces are preserved
            // by encoding the space character as a token. For simplicity, we concatenate and then
            // split by the space token if present. Alternatively, we can reconstruct word boundaries
            // from the original encoding logic. This version returns a raw concatenation.
            return string.Concat(tokens);
        }

        private IEnumerable<string> TokenizeWord(string word)
        {
            var currentTokens = word.Select(character => character.ToString()).ToList();

            foreach (var mergeRule in _mergeRules)
            {
                currentTokens = MergeAdjacentPairs(currentTokens, mergeRule);
            }

            return currentTokens;
        }

        private List<string> MergeAdjacentPairs(List<string> tokens, (string Left, string Right) mergeRule)
        {
            string delimiter = "|";
            string sequenceWithDelimiters = string.Join(delimiter, tokens);
            string mergedSequence = sequenceWithDelimiters.Replace(
                $"{mergeRule.Left}{delimiter}{mergeRule.Right}",
                mergeRule.Left + mergeRule.Right
            );
            return mergedSequence.Split(delimiter).ToList();
        }

        private Dictionary<int, string> AdjustTokens(IEnumerable<KeyValuePair<int, string>> tokensToIdentifiers)
        {
            var adjustedTokens = new Dictionary<int, string>();
            for (var index = 0; index < tokensToIdentifiers.Count(); index++)
            {
                var kvp = tokensToIdentifiers.ElementAt(index);
                if (kvp.Key != index)
                {
                    var kv = new KeyValuePair<int, string>(index, kvp.Value);
                }

                int id = kvp.Key;
                string token = kvp.Value;
                // Ensure the unknown token is included in the adjusted tokens
                if (token == _unknownTokenString)
                {
                    adjustedTokens[_unknownTokenIdentifier] = _unknownTokenString;
                }
                else
                {
                    adjustedTokens[id] = token;
                }
            }
            return adjustedTokens;
        }
    }
}
