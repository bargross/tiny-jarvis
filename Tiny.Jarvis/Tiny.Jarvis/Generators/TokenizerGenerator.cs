using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Tokenization;

namespace Tiny.Jarvis.Training.Orchestrators
{
    public static class TokenizerGenerator
    {
        public static ITokenizer GetTokenizer(TokenizerStrategy? strategy, IEnumerable<string> docs, int vocabularySize = 20, int numOfMerges = 15)
        {
            if (strategy == null)
            {
                throw new ArgumentNullException(nameof(strategy));
            }

            return strategy switch
            {
                TokenizerStrategy.BytePair => new BytePairEncodingTokenizer(docs, vocabularySize, numOfMerges),
                TokenizerStrategy.WordPiece => new WordPieceTokenizer(docs, vocabularySize),
                TokenizerStrategy.Unigram => new UnigramTokenizer(docs, vocabularySize),
                TokenizerStrategy.Chars => new CharacterTokenizer(docs.First()),
                _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
            };
        }
    }
}
