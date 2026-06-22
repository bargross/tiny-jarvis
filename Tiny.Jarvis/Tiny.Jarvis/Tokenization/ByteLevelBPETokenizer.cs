using System.Text;
using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Comparers;
using Tiny.Jarvis.Training.Tokenization.Trainers;
using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.Training.Tokenization
{
    public class ByteLevelBPETokenizer : ITokenizer<byte[]>

    {
        private const string UnknownToken = "[UNK]";
        private const string BosToken = "[BOS]";
        private const string EosToken = "[EOS]";

        private readonly Dictionary<byte[], int> _identifierToToken;
        private readonly Dictionary<int, byte[]> _tokenToIdentifier;
        private readonly List<(byte[] Left, byte[] Right)> _mergeRules;
        private readonly int _unknownTokenIdentifier;
        private readonly int _bosId;
        private readonly int _eosId;
        private readonly int _vocabSize;

        public int BOS => _bosId;
        public int EOS => _eosId;
        public int VocabSize => _vocabSize;
        public List<(byte[] Left, byte[] Right)>? MergeRules => _mergeRules;
        public Dictionary<byte[], double>? TokenLogProbabilities => null;
        public Dictionary<byte[], int> IdentifierToToken => _identifierToToken;
        public int UnknownTokenId => _unknownTokenIdentifier;
        public TokenizerStrategy Type => TokenizerStrategy.ByteLevelBPE;

        /// <summary>
        /// Constructor – trains the tokenizer on the given corpus.
        /// </summary>
        public ByteLevelBPETokenizer(IEnumerable<string> corpus, int targetVocabSize)
        {
            // Instantiate trainer with the special token strings
            var trainer = new ByteLevelBPETrainer(UnknownToken, BosToken, EosToken);
            var (tokenToIdentifier, mergeRules) = trainer.Train(corpus, targetVocabSize);

            // Store results
            _tokenToIdentifier = tokenToIdentifier;
            _mergeRules = mergeRules;
            _unknownTokenIdentifier = 0;
            _bosId = 1;
            _eosId = 2;
            _vocabSize = tokenToIdentifier.Count;

            // Rebuild identifier to token from token to identifier
            _identifierToToken = new Dictionary<byte[], int>(new ByteArrayComparer());
            foreach (var kv in tokenToIdentifier)
                _identifierToToken[kv.Value] = kv.Key;
        }

        // Private constructor for loading from pre‑trained data (not shown)
        public ByteLevelBPETokenizer(
            Dictionary<int, byte[]> tokenToIdentifier,
            List<(byte[] Left, byte[] Right)> mergeRules,
            int unknownId,
            int bosId,
            int eosId)
        {
            _tokenToIdentifier = tokenToIdentifier;
            _mergeRules = mergeRules;
            _bosId = bosId;
            _eosId = eosId;
            _vocabSize = tokenToIdentifier.Count;

            // Rebuild tokenToId
            _identifierToToken = new Dictionary<byte[], int>(new ByteArrayComparer());

            foreach (var kv in _tokenToIdentifier)
                _identifierToToken[kv.Value] = kv.Key;

            _unknownTokenIdentifier = _identifierToToken[Encoding.UTF8.GetBytes(UnknownToken)];
        }

        public IReadOnlyList<int> Encode(string text)
        {
            // Uses Helpers.PreTokenize
            var tokens = Helpers.PreTokenize(text);
            var ids = new List<int>();
            foreach (var token in tokens)
            {
                var bytes = Encoding.UTF8.GetBytes(token);
                var tokenList = bytes.Select(b => new byte[] { b }).ToList();
                foreach (var merge in _mergeRules)
                {
                    var newList = new List<byte[]>();
                    var i = 0;
                    while (i < tokenList.Count)
                    {
                        if (i < tokenList.Count - 1 && tokenList[i].SequenceEqual(merge.Left) && tokenList[i + 1] == merge.Right)
                        {
                            newList.Add(Concat(merge.Left, merge.Right));
                            i += 2;
                        }
                        else
                        {
                            newList.Add(tokenList[i]);
                            i++;
                        }
                    }

                    tokenList = newList;
                }

                foreach (var mergedTokens in tokenList)
                {
                    if (_identifierToToken.TryGetValue(mergedTokens, out int id))
                        ids.Add(id);
                    
                    else ids.Add(_unknownTokenIdentifier);
                }
            }

            return ids;
        }

        public string Decode(IReadOnlyList<int> ids)
        {
            var bytes = new List<byte>();
            foreach (int id in ids)
            {
                if (_tokenToIdentifier.TryGetValue(id, out byte[] token))
                    bytes.AddRange(token);
                else
                {
                    if (id == _unknownTokenIdentifier)
                        bytes.AddRange(Encoding.UTF8.GetBytes(UnknownToken));
                    else if (id == _bosId)
                        bytes.AddRange(Encoding.UTF8.GetBytes(BosToken));
                    else if (id == _eosId)
                        bytes.AddRange(Encoding.UTF8.GetBytes(EosToken));
                }
            }

            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        private static byte[] Concat(byte[] a, byte[] b) 
        {
            var result = new byte[a.Length + b.Length];

            Buffer.BlockCopy(a, 0, result, 0, a.Length);
            Buffer.BlockCopy(b, 0, result, a.Length, b.Length);

            return result;
        }

        public void Save(string path) { /* ... */ }
        public static ByteLevelBPETokenizer Load(string path) { throw new NotImplementedException(); }
    }
}
