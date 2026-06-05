using Tiny.Jarvis.Enums;

namespace Tiny.Jarvis.Tokenization
{
    public class SmartTokenizerGenerator(IEnumerable<string> docs, int unknownTokenIdentifier = -1, int vocabularySize = 20) //, int numOfMerges = 5)
    {
        //private BytePairEncodingTokenizer _bytePairEncodingTokenizer = new(docs, unknownTokenIdentifier, numOfMerges);
        private WordPieceTokenizer _wordPieceTokenizer = new(docs, unknownTokenIdentifier, vocabularySize);
        private UnigramTokenizer _unigramTokenizer = new(docs, unknownTokenIdentifier, vocabularySize);
        private SimpleTokenizer _simpleTokenizer = new(docs, unknownTokenIdentifier);


        public ITokenizer GetTokenizer(TokenizerStrategy? strategy)
        {
            if (strategy == null)
            {
                throw new ArgumentNullException(nameof(strategy));
            }

            return strategy switch
            {
                //TokenizerStrategy.BytePair => _bytePairEncodingTokenizer,
                TokenizerStrategy.WordPiece => _wordPieceTokenizer,
                TokenizerStrategy.Unigram => _unigramTokenizer,
                TokenizerStrategy.Simple => _simpleTokenizer,
                _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
            };
        }
    }
}
