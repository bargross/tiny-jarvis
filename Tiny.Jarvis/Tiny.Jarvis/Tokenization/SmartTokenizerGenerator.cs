using Tiny.Jarvis.Enums;

namespace Tiny.Jarvis.Tokenization
{
    public static class SmartTokenizerGenerator
    {
        public static ITokenizer GetTokenizer(TokenizerStrategy? strategy, IEnumerable<string> docs, int unknownTokenIdentifier = -1, int vocabularySize = 20) //, int numOfMerges = 5)
        {
            if (strategy == null)
            {
                throw new ArgumentNullException(nameof(strategy));
            }

            return strategy switch
            {
                //TokenizerStrategy.BytePair => new BytePairEncodingTokenizer(docs, unknownTokenIdentifier, vocabularySize),
                TokenizerStrategy.WordPiece => new WordPieceTokenizer(docs, unknownTokenIdentifier, vocabularySize),
                TokenizerStrategy.Unigram => new UnigramTokenizer(docs, unknownTokenIdentifier, vocabularySize),
                TokenizerStrategy.Simple => new SimpleTokenizer(docs, unknownTokenIdentifier),
                _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
            };
        }
    }
}
