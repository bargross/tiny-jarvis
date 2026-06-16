using System.Text.Json;
using System.Text.Json.Serialization;
using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Orchestrators;

namespace Tiny.Jarvis.Training.Serializers
{
    public static class TokenizerSerializer
    {
        private class TokenizerData
        {
            public Dictionary<string, double>? TokenLogProbabilities { get; set; }
            public Dictionary<string, int>? IdentifierToToken { get; set; }
            public List<(string Left, string Right)>? MergeRules { get; set; }
            public int UnknownTokenId { get; set; }
            public int BosTokenId { get; set; }
            public int EosTokenId { get; set; }
        }

        /// <summary>
        /// Saves a WordPieceTokenizer to a JSON file.
        /// </summary>
        public static void Save(ITokenizer tokenizer, string filePath)
        {
            var data = new TokenizerData
            {
                IdentifierToToken = tokenizer.IdentifierToToken, // need a public getter, or make internal
                UnknownTokenId = tokenizer.UnknownTokenId,
                BosTokenId = tokenizer.BOS,
                EosTokenId = tokenizer.EOS
            };

            var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull});

            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads a WordPieceTokenizer from a JSON file.
        /// </summary>
        public static ITokenizer Load(string filePath, TokenizerStrategy strategy)
        {
            var json = File.ReadAllText(filePath);
            var data = JsonSerializer.Deserialize<TokenizerData>(json);

            switch(strategy)
            {
                case TokenizerStrategy.Chars:
                    return TokenizerGenerator.GetTokenizer(TokenizerStrategy.Chars, ["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,|!?-'\""]);
                case TokenizerStrategy.WordPiece:
                    return new WordPieceTokenizer(data.IdentifierToToken, data.UnknownTokenId, data.BosTokenId, data.EosTokenId);
                case TokenizerStrategy.Unigram:
                    return new UnigramTokenizer(data.TokenLogProbabilities, data.IdentifierToToken, data.UnknownTokenId, data.BosTokenId, data.EosTokenId);
                case TokenizerStrategy.BytePair:
                    return new BytePairEncodingTokenizer(data.MergeRules, data.IdentifierToToken, data.UnknownTokenId, data.BosTokenId, data.EosTokenId);
            }

            return null;
            //throw new ArgumentException("tokenizer data not found."); // might be best to let it flow so it creates a new one
        }
    }
}
