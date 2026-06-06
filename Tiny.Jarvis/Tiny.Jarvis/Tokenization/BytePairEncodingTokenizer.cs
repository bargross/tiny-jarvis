using Tiny.Jarvis.Tokenization.Trainers;

namespace Tiny.Jarvis.Tokenization
{
    public class BytePairEncodingTokenizer: ITokenizer
    {
        private readonly Dictionary<string, int> _identifierToToken;
        private readonly Dictionary<int, string> _tokenToIdentifier;
        private readonly List<(string Left, string Right)> _mergeRules;
        private readonly int _unknownTokenIdentifier;
        private readonly string _unknownToken = "[UNK]";
        private readonly string _bosToken = "[BOS]";

        private readonly int _vocabularySize;

        public int VocabSize => _vocabularySize;
        public int Bos { get; } // Beginning of Sequence token ID

        public BytePairEncodingTokenizer(IEnumerable<string> docs, int unknownTokenIdentifier = -1, int numberOfMerges = 5)
        {
            // Combine all documents into one large text (or pass as enumerable)
            string allText = string.Join("\n", docs);

            // Train BPE on the combined text (assuming the trainer can work on a single string)
            var trainingResult = new BytePairEncodingTrainer().Train(allText, numberOfMerges);
            // trainingResult should contain:
            //   - Vocabulary (HashSet<string>) of all subword tokens
            //   - MergeRules (List<(string,string)>)

            // Build a set of all tokens from the trained vocabulary and add special tokens
            var allTokensSet = new HashSet<string>(trainingResult.IdentifierToToken.Keys.ToList());

            allTokensSet.Add(_unknownToken);
            allTokensSet.Add(_bosToken);

            // Assign consecutive IDs, optionally forcing fixed IDs for special tokens like [UNK] and [BOS]
            var identifierToToken = new Dictionary<string, int>();
            identifierToToken[_unknownToken] = 0;
            identifierToToken[_bosToken] = 1;

            int nextId = 2;
            foreach (var token in allTokensSet.OrderBy(t => t))
            {
                if (token == _unknownToken || token == _bosToken)
                    continue;

                identifierToToken[token] = nextId++;
            }

            // Build reverse mapping
            var tokenToIdentifier = identifierToToken.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            // Store fields
            _identifierToToken = identifierToToken;
            _tokenToIdentifier = tokenToIdentifier;
            _mergeRules = trainingResult.MergeRules; // or new List<(string,string)> if not provided
            _vocabularySize = _identifierToToken.Count;
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
                .Select(id => _tokenToIdentifier.GetValueOrDefault(id, _unknownToken))
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
