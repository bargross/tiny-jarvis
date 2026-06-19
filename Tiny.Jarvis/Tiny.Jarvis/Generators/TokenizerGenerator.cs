using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Tokenization;

namespace Tiny.Jarvis.Training.Orchestrators
{
    public static class TokenizerGenerator
    {
        public static ITokenizer<TVocabulary> GetTokenizer<TVocabulary>(TokenizerStrategy? strategy, IEnumerable<string> trainingDocuments, int vocabularySize = 20, int numOfMerges = 15)
        {
            if (strategy == null)
            {
                throw new ArgumentNullException(nameof(strategy));
            }
            
            if (typeof(TVocabulary) == typeof(string) || typeof(TVocabulary) == typeof(byte[]))
            {

                return strategy switch
                {
                    TokenizerStrategy.BytePair => new BytePairEncodingTokenizer(trainingDocuments, vocabularySize, numOfMerges) as ITokenizer<TVocabulary>,
                    TokenizerStrategy.WordPiece => new WordPieceTokenizer(trainingDocuments, vocabularySize) as ITokenizer<TVocabulary>,
                    TokenizerStrategy.Unigram => new UnigramTokenizer(trainingDocuments, vocabularySize) as ITokenizer<TVocabulary>,
                    TokenizerStrategy.Chars => new CharacterTokenizer(trainingDocuments.First()) as ITokenizer<TVocabulary>,

                    // limit the corpus for tokenizer trainer to 2500 for now, otherwise it takes too long to train
                    TokenizerStrategy.ByteLevelBPE => new ByteLevelBPETokenizer(trainingDocuments.Take(1000), vocabularySize) as ITokenizer<TVocabulary>,
                    _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
                };
            }

            throw new ArgumentException("Tokenizer for type requested does not exist!");
        }
    }
}
