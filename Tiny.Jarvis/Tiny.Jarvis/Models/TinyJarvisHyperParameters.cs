using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Training.Enums;

namespace Tiny.Jarvis.Training.Models
{
    public class TinyJarvisHyperParameters
    {
        public int EmbeddingSize { get; set; } = 36;
        public int LayerCount { get; set; } = 4; // just one transformer block for speed - try layerCount=2 to see improvement
        public int HeadCount { get; set; } = 4;
        public double LearningRate { get; set; } = 0.001;

        public TokenizerStrategy TokenizerStrategy { get; set; } = TokenizerStrategy.WordPiece;
        public OptimizerStrategy OptimizerStrategy { get; set; } = OptimizerStrategy.SGDMomentum;

        // set this based on the average length of your documents (in tokens) - it controls the context window size for the model, so longer is generally better for performance but increases training time and memory usage
        public int MaxSequenceLength { get; set; } = 34;

        // TODO: it might be worth trying different values for different tokenizers to see if some converge faster than others (e.g. character-level tokenizers will likely require more steps than word-level ones)
        public int MaxNumberOfSteps { get; set; } = 10000; // increase this for better performance - the optimal number depends on the size of your dataset and the complexity of the task
        public int NumOfMerges { get; set; } = 150;
        public double MaxGradNorm { get; set; } = 0.8;

        private int? _vocabularySize;
        public int VocabularySize 
        {
            get => _vocabularySize ?? (TokenizerStrategy is TokenizerStrategy.WordPiece
                        or TokenizerStrategy.Unigram ? 250 : 0);
            
            set => _vocabularySize = value;
            
        }

        // for loading in case of going with previous run
        public string? LoadModelFile { get; set; }
        public string? LoadTokenizerFile { get; set; }
        public string? LoadOptimizerFile { get; set; }

        public string? SaveModelFile { get; set; }
        public string? SaveTokenizerFile { get; set; }
        public string? SaveOptimizerFile { get; set; }
    }
}
