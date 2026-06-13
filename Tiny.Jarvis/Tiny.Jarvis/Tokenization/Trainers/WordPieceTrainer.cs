namespace Tiny.Jarvis.Tokenization.Trainers
{
    internal class WordPieceTrainer
    {
        /// <summary>
        /// Trains a subword vocabulary using a BPE‑like frequency‑based merge algorithm.
        /// Each iteration merges the most frequent adjacent pair of symbols.
        /// </summary>
        public HashSet<string> Train(IEnumerable<string> trainingCorpus, int targetVocabularySize)
        {
            Console.WriteLine("Beginning tokenizer training (frequency‑based merge)...");

            // Count word frequencies (how many times each word appears)
            var wordFreq = new Dictionary<string, int>();
            foreach (var sentence in trainingCorpus)
            {
                foreach (var word in sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    wordFreq.TryGetValue(word, out int count);
                    wordFreq[word] = count + 1;
                }
            }

            // Initial vocabulary: all characters
            var vocab = new HashSet<string>();
            foreach (var word in wordFreq.Keys)
                foreach (char c in word)
                    vocab.Add(c.ToString());

            // Perform merges until target size reached or no more pairs
            int mergeCount = 0;
            int maxMerges = targetVocabularySize * 2; // safety

            while (vocab.Count < targetVocabularySize && mergeCount < maxMerges)
            {
                // Count frequencies of all adjacent symbol pairs across the corpus
                var pairFreq = new Dictionary<string, int>();
                foreach (var (word, freq) in wordFreq)
                {
                    // Segment the word using current vocabulary (greedy longest match)
                    var symbols = SegmentWord(word, vocab);
                    for (int i = 0; i < symbols.Count - 1; i++)
                    {
                        var pair = symbols[i] + symbols[i + 1];

                        pairFreq.TryGetValue(pair, out int current);
                        pairFreq[pair] = current + freq;
                    }
                }

                if (pairFreq.Count == 0) break;

                // Find the pair with highest frequency
                string bestPair = pairFreq.OrderByDescending(kv => kv.Value).First().Key;
                vocab.Add(bestPair);
                mergeCount++;

                // Show progress
                if (mergeCount % 50 == 0 || vocab.Count >= targetVocabularySize)
                {
                    double percent = vocab.Count * 100.0 / targetVocabularySize;
                    Console.WriteLine($"\rTraining: {percent:F2}% complete (vocab size: {vocab.Count})");
                }
            }

            Console.WriteLine($"\nTraining complete. Final vocabulary size: {vocab.Count}");
            return vocab;
        }

        /// <summary>
        /// Segments a word into known symbols using greedy longest match (iterative).
        /// </summary>
        private List<string> SegmentWord(string word, HashSet<string> vocab)
        {
            var tokens = new List<string>();
            var pos = 0;
            var len = word.Length;
            while (pos < len)
            {
                var matched = false;
                var maxLen = Math.Min(len - pos, 20); // limit token length
                for (var l = maxLen; l >= 1; l--)
                {
                    string candidate = word.Substring(pos, l);
                    if (vocab.Contains(candidate))
                    {
                        tokens.Add(candidate);
                        pos += l;
                        matched = true;
                        break;
                    }
                }

                if (!matched)
                {
                    tokens.Add(word[pos].ToString());
                    pos++;
                }
            }

            return tokens;
        }
    }

}
