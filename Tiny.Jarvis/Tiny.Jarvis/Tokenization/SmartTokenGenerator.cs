using Tiny.Jarvis.Enums;

namespace Tiny.Jarvis.Tokenization
{
    internal class SmartTokenGenerator(List<string> docs, int unknownTokenIdentifier = 0, int vocabularySize = 20)
    {
        private BytePairEncodingTokenizer _bytePairEncodingTokenizer = new(docs, unknownTokenIdentifier, vocabularySize);
        private WordPieceTokenizer _wordPieceTokenizer = new(docs, unknownTokenIdentifier, vocabularySize);
        private UnigramTokenizer _unigramTokenizer = new(docs, unknownTokenIdentifier, vocabularySize);
        private SimpleTokenizer _simpleTokenizer = new(docs, unknownTokenIdentifier);


        public ITokenizer GetTokenizer(TokenizerStrategy strategy)
        {
            return strategy switch
            {
                TokenizerStrategy.BytePair => _bytePairEncodingTokenizer,
                TokenizerStrategy.WordPiece => _wordPieceTokenizer,
                TokenizerStrategy.Unigram => _unigramTokenizer,
                TokenizerStrategy.Simple => _simpleTokenizer,
                _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
            };
        }
    }
}
