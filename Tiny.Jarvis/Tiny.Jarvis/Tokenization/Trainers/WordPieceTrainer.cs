using System.Diagnostics.Metrics;

namespace Tiny.Jarvis.Tokenization.Trainers
{
    internal class WordPieceTrainer
    {
        public HashSet<string> Train(IEnumerable<string> trainingCorpus, int targetVocabularySize)
        {
            Console.WriteLine("Beginning tokenizer WordPiece training:...");
            // Start with all characters as initial vocabulary
            var vocabulary = trainingCorpus
                .SelectMany(sentence => sentence.Split(' '))
                .SelectMany(word => word.ToCharArray())
                .Select(character => character.ToString())
                .Distinct()
                .ToHashSet();

            Console.WriteLine("Looking for best merge of pairs:...");
            Console.WriteLine(Environment.NewLine);

            var counter = 1;
            while (vocabulary.Count < targetVocabularySize)
            {
                var percentage = counter * 100.0 / targetVocabularySize;
                Console.WriteLine($"\rTraining: {percentage:F2}% complete.");

                // Find the best pair of adjacent tokens to merge (by likelihood increase)
                var bestMerge = FindBestMerge(trainingCorpus, vocabulary);
                if (bestMerge == null) break;

                vocabulary.Add(bestMerge);
                counter++;
            }

            Console.WriteLine("Training complete.");
            Console.WriteLine(Environment.NewLine);
            return vocabulary;
        }

        private string FindBestMerge(IEnumerable<string> trainingCorpus, HashSet<string> currentVocabulary)
        {
            // Generate all possible merges from existing vocabulary
            var possibleMerges = from first in currentVocabulary
                                 from second in currentVocabulary
                                 select first + second;

            var bestScore = double.NegativeInfinity;
            string bestMerge = null;

            var counter = 1;
            foreach (var candidateMerge in possibleMerges)
            {
                var percentage = counter * 100.0 / possibleMerges.Count();
                Console.WriteLine($"\rBest Merge Completion Rate: {percentage:F2}% complete.");

                var likelihoodIncrease = ComputeLikelihoodIncrease(trainingCorpus, currentVocabulary, candidateMerge);
                if (likelihoodIncrease > bestScore)
                {
                    bestScore = likelihoodIncrease;
                    bestMerge = candidateMerge;
                }

                counter++;
            }

            Console.WriteLine(Environment.NewLine);

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
            int position = 0;
            int length = word.Length;
            //int iteration = 0;

            //Console.WriteLine("Looking for segment word with longest match.");
            while (position < length)
            {
                bool matched = false;

                // Try to find the longest token starting at current position.
                // Limit max token length to 20 for performance (adjustable).
                int maxTokenLen = Math.Min(length - position, 20);
                //Console.WriteLine($"Max Token Length: {maxTokenLen} for Iteration: {iteration + 1}");
                for (var tokenLen = maxTokenLen; tokenLen >= 1; tokenLen--)
                {
                    var candidate = word.Substring(position, tokenLen);
                    //Console.WriteLine($"Candidate {candidate} for token length: {tokenLen}.");
                    if (vocabulary.Contains(candidate))
                    {
                        //Console.WriteLine($"Candidate {candidate} is a match, adding...");
                        yield return candidate;

                        position += tokenLen;
                        //Console.WriteLine($"incrementing position to {position}");

                        matched = true;
                        //Console.WriteLine($"Match found, escaping...");
                        break;
                    }
                }

                // No known token → take a single character (prevents infinite loop).
                if (!matched)
                {
                    //Console.WriteLine($"No match found...");

                    yield return word[position].ToString();
                    
                    position++;
                }

                //iteration++;
            }

            //Console.WriteLine("match found..");
        }
    }
}
