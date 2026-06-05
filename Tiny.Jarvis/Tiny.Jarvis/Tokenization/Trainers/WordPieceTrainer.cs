namespace Tiny.Jarvis.Tokenization.Trainers
{
    internal class WordPieceTrainer
    {
        public HashSet<string> Train(IEnumerable<string> trainingCorpus, int targetVocabularySize)
        {
            // Start with all characters as initial vocabulary
            var vocabulary = trainingCorpus
                .SelectMany(sentence => sentence.Split(' '))
                .SelectMany(word => word.ToCharArray())
                .Select(character => character.ToString())
                .Distinct()
                .ToHashSet();

            while (vocabulary.Count < targetVocabularySize)
            {
                // Find the best pair of adjacent tokens to merge (by likelihood increase)
                var bestMerge = FindBestMerge(trainingCorpus, vocabulary);
                if (bestMerge == null) break;
                vocabulary.Add(bestMerge);
            }

            return vocabulary;
        }

        private string FindBestMerge(IEnumerable<string> trainingCorpus, HashSet<string> currentVocabulary)
        {
            // Generate all possible merges from existing vocabulary
            var possibleMerges = from first in currentVocabulary
                                 from second in currentVocabulary
                                 select first + second;

            double bestScore = double.NegativeInfinity;
            string bestMerge = null;

            foreach (var candidateMerge in possibleMerges)
            {
                double likelihoodIncrease = ComputeLikelihoodIncrease(trainingCorpus, currentVocabulary, candidateMerge);
                if (likelihoodIncrease > bestScore)
                {
                    bestScore = likelihoodIncrease;
                    bestMerge = candidateMerge;
                }
            }

            return bestMerge;
        }

        private double ComputeLikelihoodIncrease(IEnumerable<string> corpus, HashSet<string> vocabulary, string candidateMerge)
        {
            // Simplified: measure how many new word segmentations become possible
            // Real WordPiece uses a proper probabilistic model
            var extendedVocabulary = new HashSet<string>(vocabulary) { candidateMerge };

            int originalTokenCount = corpus
                .SelectMany(sentence => sentence.Split(' '))
                .SelectMany(word => SegmentWordWithLongestMatch(word, vocabulary))
                .Count();

            int newTokenCount = corpus
                .SelectMany(sentence => sentence.Split(' '))
                .SelectMany(word => SegmentWordWithLongestMatch(word, extendedVocabulary))
                .Count();

            // Fewer tokens (more merges) is better → higher likelihood
            return originalTokenCount - newTokenCount;
        }

        private IEnumerable<string> SegmentWordWithLongestMatch(string word, HashSet<string> vocabulary)
        {
            // Same recursive longest-match logic from earlier
            if (string.IsNullOrEmpty(word)) yield break;
            var matches = vocabulary.Where(token => word.StartsWith(token)).OrderByDescending(t => t.Length);
            string best = matches.FirstOrDefault();
            if (best != null)
            {
                yield return best;
                foreach (var token in SegmentWordWithLongestMatch(word.Substring(best.Length), vocabulary))
                    yield return token;
            }
            else
            {
                yield return "[UNK]";
                foreach (var token in SegmentWordWithLongestMatch(word.Substring(1), vocabulary))
                    yield return token;
            }
        }
    }
}
