using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.Tokenization.Trainers
{
    public class WordPieceTrainer
    {
        private const string SubwordPrefix = "##";

        /// <summary>
        /// Trains a subword vocabulary using the WordPiece algorithm.
        /// Merges are selected by maximising score = freq(pair) / (freq(a) * freq(b)),
        /// which approximates the likelihood gain of merging pair (a, b).
        /// </summary>
        public Dictionary<string, int> Train(IEnumerable<string> trainingCorpus, int targetVocabularySize)
        {
            Console.WriteLine("Beginning tokenizer training (WordPiece)...");

            // ── Count word frequencies ────────────────────────────────────────
            var wordFreq = new Dictionary<string, int>();
            foreach (var sentence in trainingCorpus)
                foreach (var word in Helpers.PreTokenize(sentence))
                {
                    wordFreq.TryGetValue(word, out int count);
                    wordFreq[word] = count + 1;
                }
            

            // ── Build initial character-level vocabulary with ## prefixes ─────
            //   - First char of each word: plain  e.g. 'h'
            //   - Subsequent chars:        ##     e.g. '##e', '##l', '##l', '##o'
            var vocab = new HashSet<string>();
            foreach (var word in wordFreq.Keys)
                for (int i = 0; i < word.Length; i++)
                {
                    var ch = word[i].ToString();

                    vocab.Add(i == 0 ? ch : SubwordPrefix + ch);
                }

            // ── Cache the initial segmentation of every word ──────────────────
            //   Segmentation respects the ## prefix: only the first symbol is bare.
            var wordSegments = wordFreq.Keys.ToDictionary(w => w, w => SegmentWord(w, vocab));

            // ── Merge loop ────────────────────────────────────────────────────
            var mergeCount = 0;
            //var tokenFrequency = new Dictionary<string, int>();
            while (vocab.Count < targetVocabularySize)
            {
                // Count individual token frequencies and pair frequencies
                var tokenFreq = new Dictionary<string, int>();
                var pairFreq = new Dictionary<(string left, string right), int>();

                foreach (var (word, freq) in wordFreq)
                {
                    var symbols = wordSegments[word];
                    foreach (var sym in symbols)
                    {
                        tokenFreq.TryGetValue(sym, out int tf);
                        tokenFreq[sym] = tf + freq;
                    }

                    for (var i = 0; i < symbols.Count - 1; i++)
                    {
                        var pair = (symbols[i], symbols[i + 1]);

                        pairFreq.TryGetValue(pair, out int pf);
                        pairFreq[pair] = pf + freq;
                    }
                }

                if (pairFreq.Count == 0) break;

                // WordPiece score: freq(a,b) / (freq(a) * freq(b))
                // Use double to avoid integer overflow on larger corpora.
                var bestPair = default((string left, string right));
                var bestScore = double.MinValue;

                foreach (var (pair, pf) in pairFreq)
                {
                    var score = pf / ((double)tokenFreq[pair.left] * tokenFreq[pair.right]);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestPair = pair;
                    }
                }

                // Form the merged token.
                // If B starts with ##, strip its prefix before concatenating
                // (e.g. "un" + "##der" → "under", then re-prefix if needed).
                var mergedRaw = bestPair.left + (bestPair.right.StartsWith(SubwordPrefix) 
                    ? bestPair.right.Substring(SubwordPrefix.Length) : bestPair.right);

                // The merged token keeps the ## prefix of A (if A had one)
                // because it occupies the same position in the word.
                if (!vocab.Add(mergedRaw))
                {
                    // No progress; break out of the while loop to avoid infinite loops
                    break;
                }

                mergeCount++;

                // Re-segment only words that contain the merged pair ──────────
                foreach (var word in wordFreq.Keys)
                {
                    var symbols = wordSegments[word];
                    var affected = false;
                    for (var i = 0; i < symbols.Count - 1; i++)
                    {
                        if (symbols[i] == bestPair.left && symbols[i + 1] == bestPair.right)
                        {
                            affected = true;
                            break;
                        }
                    }

                    if (affected)
                        wordSegments[word] = SegmentWord(word, vocab);
                }

                if (mergeCount % 50 == 0 || vocab.Count >= targetVocabularySize)
                {
                    var pct = vocab.Count * 100.0 / targetVocabularySize;
                    Console.WriteLine($"Training: {pct:F1}% (vocab={vocab.Count}, merges={mergeCount})");
                }
            }

            // After the merge loop, compute final frequencies
            var finalTokenFreq = new Dictionary<string, int>();
            foreach (var (word, freq) in wordFreq)
            {
                var symbols = SegmentWord(word, vocab); // use the final vocab
                foreach (var sym in symbols)
                {
                    finalTokenFreq.TryGetValue(sym, out int current);
                    finalTokenFreq[sym] = current + freq;
                }
            }

            Console.WriteLine($"Training complete. Vocab size: {vocab.Count}, merges: {mergeCount}");
            Console.WriteLine(Environment.NewLine);

            return finalTokenFreq;
        }

        /// <summary>
        /// Greedy longest-match segmentation that respects the ## continuation prefix.
        /// The first symbol in a word is looked up bare; all subsequent ones with ##.
        /// Falls back to the bare character (or ##char) if no longer match exists.
        /// </summary>
        private List<string> SegmentWord(string word, HashSet<string> vocab)
        {
            var tokens = new List<string>();
            var pos = 0;
            while (pos < word.Length)
            {
                var isFirst = pos == 0;
                var matched = false;
                var maxLen = Math.Min(word.Length - pos, 20);

                for (var l = maxLen; l >= 1; l--)
                {
                    var raw = word.Substring(pos, l);
                    var candidate = isFirst ? raw : SubwordPrefix + raw;

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
                    // Single character fallback — always add even if not in vocab yet
                    var raw = word[pos].ToString();
                    var candidate = isFirst ? raw : SubwordPrefix + raw;

                    tokens.Add(candidate);
                    pos++;
                }
            }

            return tokens;
        }
    }
}