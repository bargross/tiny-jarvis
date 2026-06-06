using Tiny.Jarvis.Enums;

namespace Tiny.Jarvis.Tokenization
{
    public static class SmartTokenizerGenerator
    {
        public static ITokenizer GetTokenizer(TokenizerStrategy? strategy, IEnumerable<string> docs, int vocabularySize = 20) //, int numOfMerges = 5)
        {
            if (strategy == null)
            {
                throw new ArgumentNullException(nameof(strategy));
            }

            return strategy switch
            {
                TokenizerStrategy.BytePair => new BytePairEncodingTokenizer(docs, vocabularySize),
                TokenizerStrategy.WordPiece => new WordPieceTokenizer(docs, vocabularySize),
                TokenizerStrategy.Unigram => new UnigramTokenizer(docs, vocabularySize),
                TokenizerStrategy.Simple => new SimpleTokenizer(docs),
                _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
            };
        }
    }
}
