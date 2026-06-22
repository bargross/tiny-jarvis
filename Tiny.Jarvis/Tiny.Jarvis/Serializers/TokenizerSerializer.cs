using System.Text.Json;
using System.Text.Json.Serialization;
using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Orchestrators;
using Tiny.Jarvis.Training.Tokenization;

namespace Tiny.Jarvis.Training.Serializers
{
    public static class TokenizerSerializer
    {
        private class TokenizerDataType
        {
            [JsonConverter(typeof(JsonStringEnumConverter))]
            public TokenizerStrategy Type { get; set; }
        }

        private class TokenizerData<TVocabulary> : TokenizerDataType
        { 
            public Dictionary<TVocabulary, double>? TokenLogProbabilities { get; set; }
            public Dictionary<TVocabulary, int>? IdentifierToToken { get; set; }
            public List<(TVocabulary Left, TVocabulary Right)>? MergeRules { get; set; }
            public int UnknownTokenId { get; set; }
            public int BosTokenId { get; set; }
            public int EosTokenId { get; set; }
        }

        /// <summary>
        /// Saves a WordPieceTokenizer to a JSON file.
        /// </summary>
        public static void Save<TVocabulary>(ITokenizer<TVocabulary> tokenizer, string filePath)
        {
            var data = new TokenizerData<TVocabulary>
            {
                Type = tokenizer.Type,
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
        public static ITokenizer<TVocabulary> Load<TVocabulary>(string filePath, TokenizerStrategy strategy)
        {
            var json = File.ReadAllText(filePath);

            var data = JsonSerializer.Deserialize<TokenizerData<TVocabulary>>(json);

            if (data.Type != strategy)
                throw new Exception($"Tokenizer requested {strategy.ToString()} and deserialized tokenizer {data.Type.ToString()} types do not match.");

            switch(data.Type)
            {
                case TokenizerStrategy.Chars:
                    return TokenizerGenerator.GetTokenizer<string>(TokenizerStrategy.Chars, ["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,|!?-'\""]) as ITokenizer<TVocabulary>;
                case TokenizerStrategy.WordPiece:
                {
                        var identifierToToken = data.IdentifierToToken as Dictionary<string, int>;

                    return new WordPieceTokenizer(identifierToToken, data.UnknownTokenId, data.BosTokenId, data.EosTokenId) as ITokenizer<TVocabulary>;
                }
                case TokenizerStrategy.Unigram:
                {
                    var identifierToToken = data.IdentifierToToken as Dictionary<string, int>;
                    var tokenLogProbabilities = data.TokenLogProbabilities as Dictionary<string, double>;

                    return new UnigramTokenizer(tokenLogProbabilities, identifierToToken, data.UnknownTokenId, data.BosTokenId, data.EosTokenId) as ITokenizer<TVocabulary>;
                }
                case TokenizerStrategy.BytePair:
                { 
                    
                    var identifierToToken = data.IdentifierToToken as Dictionary<string, int>;
                    var mergeRules = data.MergeRules as List<(string left, string right)>;

                    return new BytePairEncodingTokenizer(mergeRules, identifierToToken, data.UnknownTokenId, data.BosTokenId, data.EosTokenId) as ITokenizer<TVocabulary>;
                }
                case TokenizerStrategy.ByteLevelBPE:
                {

                    var identifierToToken = data.IdentifierToToken as Dictionary<byte[], int>;
                    var mergeRules = data.MergeRules as List<(byte[] left, byte[] right)>;

                    return (ITokenizer<TVocabulary>)new ByteLevelBPETokenizer(identifierToToken.ToDictionary(kv => kv.Value, kv => kv.Key), mergeRules, data.UnknownTokenId, data.BosTokenId, data.EosTokenId);
                }
            }

            return null;  // might be best to let it flow so it creates a new one
        }
    }
}
