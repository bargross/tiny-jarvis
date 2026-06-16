using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Tokenization;

namespace Tiny.Jarvis.Training.Orchestrators
{
    public static class TokenizerGenerator
    {
        public static ITokenizer<TVocabulary> GetTokenizer<TVocabulary>(TokenizerStrategy? strategy, IEnumerable<string> docs, int vocabularySize = 20, int numOfMerges = 15)
        {
            if (strategy == null)
            {
                throw new ArgumentNullException(nameof(strategy));
            }
            
            if (typeof(TVocabulary) == typeof(string) || typeof(TVocabulary) == typeof(byte[]))
            {

                return strategy switch
                {
                    TokenizerStrategy.BytePair => new BytePairEncodingTokenizer(docs, vocabularySize, numOfMerges) as ITokenizer<TVocabulary>,
                    TokenizerStrategy.WordPiece => new WordPieceTokenizer(docs, vocabularySize) as ITokenizer<TVocabulary>,
                    TokenizerStrategy.Unigram => new UnigramTokenizer(docs, vocabularySize) as ITokenizer<TVocabulary>,
                    TokenizerStrategy.Chars => new CharacterTokenizer(docs.First()) as ITokenizer<TVocabulary>,

                    // limit the corpus for tokenizer trainer to 2500 for now, otherwise it takes too long to train
                    TokenizerStrategy.ByteLevelBPE => new ByteLevelBPETokenizer(docs.Take(1000), vocabularySize) as ITokenizer<TVocabulary>,
                    _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
                };
            }

            throw new ArgumentException("Tokenizer for type requested does not exist!");
        }
    }
}
