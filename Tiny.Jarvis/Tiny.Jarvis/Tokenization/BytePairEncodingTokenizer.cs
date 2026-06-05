using Tiny.Jarvis.Tokenization.Trainers;

namespace Tiny.Jarvis.Tokenization
{
    public class BytePairEncodingTokenizer: ITokenizer
    {
        private readonly Dictionary<string, int> _tokenToIdentifier;
        private readonly Dictionary<int, string> _identifierToToken;
        private readonly List<(string Left, string Right)> _mergeRules;
        private readonly int _unknownTokenIdentifier;
        private readonly string _unknownTokenString = "[UNK]";
       
        public int VocabSize => _tokenToIdentifier.Count;

        public BytePairEncodingTokenizer(List<string> docs, int unknownTokenIdentifier, int numberOfMerges = 5)
        {
            var tokenizerTrainingData = docs.Select(text => new BytePairEncodingTrainer().Train(text, numberOfMerges));

            _tokenToIdentifier = tokenizerTrainingData.SelectMany(kv => kv.TokenToIdentifier).ToDictionary();
            _mergeRules = tokenizerTrainingData.SelectMany(kv => kv.MergeRules).ToList();
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _identifierToToken = _tokenToIdentifier.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            // Ensure unknown token exists in reverse map
            if (!_identifierToToken.ContainsKey(unknownTokenIdentifier))
                _identifierToToken[unknownTokenIdentifier] = _unknownTokenString;
        }

        public BytePairEncodingTokenizer(
            Dictionary<string, int> tokenToIdentifier,
            List<(string Left, string Right)> mergeRules,
            int unknownTokenIdentifier)
        {
            _tokenToIdentifier = tokenToIdentifier;
            _mergeRules = mergeRules;
            _unknownTokenIdentifier = unknownTokenIdentifier;
            _identifierToToken = tokenToIdentifier.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            // Ensure unknown token exists in reverse map
            if (!_identifierToToken.ContainsKey(unknownTokenIdentifier))
                _identifierToToken[unknownTokenIdentifier] = _unknownTokenString;
        }

        public IReadOnlyList<int> Encode(string text)
        {
            return text
                .Split(' ')
                .SelectMany(word => TokenizeWord(word))
                .Select(token => _tokenToIdentifier.GetValueOrDefault(token, _unknownTokenIdentifier))
                .ToList();
        }

        public string Decode(IReadOnlyList<int> identifiers)
        {
            var tokens = identifiers
                .Select(id => _identifierToToken.GetValueOrDefault(id, _unknownTokenString))
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
    }
}
