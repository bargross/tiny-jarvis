
using System.Text;
using Tiny.Jarvis.Training.Comparers;
using Tiny.Jarvis.Training.Util;

namespace Tiny.Jarvis.Training.Tokenization.Trainers
{
    /// <summary>
    /// Trains a Byte‑Level BPE model.
    /// </summary>
    public class ByteLevelBPETrainer
    {
        private readonly string _unknownToken;
        private readonly string _bosToken;
        private readonly string _eosToken;

        public ByteLevelBPETrainer(string unknownToken, string bosToken, string eosToken)
        {
            _unknownToken = unknownToken;
            _bosToken = bosToken;
            _eosToken = eosToken;
        }

        public (Dictionary<int, byte[]> IdToToken, List<(byte[] Left, byte[] Right)> MergeRules) Train(
            IEnumerable<string> corpus,
            int targetVocabSize)
        {
            Console.WriteLine("Starting Byte‑Level BPE training...");
            Console.WriteLine($"Target vocabulary size: {targetVocabSize} (including special tokens).");

            // 1. Pre‑tokenise and convert to byte sequences
            var wordByteSequences = new List<List<byte[]>>();
            var totalWords = 0;
            foreach (var doc in corpus)
            {
                var tokens = Helpers.PreTokenize(doc);
                foreach (var token in tokens)
                {
                    var bytes = Encoding.UTF8.GetBytes(token);
                    var initialTokens = bytes.Select(b => new byte[] { b }).ToList();
                    wordByteSequences.Add(initialTokens);
                    totalWords++;
                }
            }
            Console.WriteLine($"Pre‑tokenised {totalWords} words.");

            // Build initial id→token mapping: special tokens first (IDs 0,1,2)
            var idToToken = new Dictionary<int, byte[]>
            {
                [0] = Encoding.UTF8.GetBytes(_unknownToken),
                [1] = Encoding.UTF8.GetBytes(_bosToken),
                [2] = Encoding.UTF8.GetBytes(_eosToken)
            };
            var nextId = 3;

            // Add all bytes (0-255) as initial tokens
            for (var b = 0; b <= 255; b++)
            {
                idToToken[nextId] = new byte[] { (byte)b };
                nextId++;
            }

            // Build temporary token→id mapping for fast lookup during training
            var tokenToId = new Dictionary<byte[], int>(new ByteArrayComparer());
            foreach (var kv in idToToken)
                tokenToId[kv.Value] = kv.Key;

            var initialVocabSize = idToToken.Count;
            Console.WriteLine($"Initial vocabulary size (including bytes and special tokens): {initialVocabSize}");

            // Train merges
            var mergeRules = new List<(byte[] Left, byte[] Right)>();
            var currentVocabSize = tokenToId.Count;
            var maxMerges = targetVocabSize - currentVocabSize;
            if (maxMerges < 0) maxMerges = 0;

            Console.WriteLine($"Maximum number of merges to perform: {maxMerges}");

            var mergeCount = 0;
            var reportInterval = Math.Max(1, maxMerges / 20);

            while (mergeCount < maxMerges && wordByteSequences.Count > 0)
            {
                // Count pair frequencies
                var pairFreq = new Dictionary<(byte[] Left, byte[] Right), int>();
                foreach (var seq in wordByteSequences)
                {
                    for (var i = 0; i < seq.Count - 1; i++)
                    {
                        var left = seq[i];
                        var right = seq[i + 1];
                        var key = (left, right);

                        pairFreq.TryGetValue(key, out var current);
                        pairFreq[key] = current + 1;
                    }
                }

                if (pairFreq.Count == 0)
                {
                    Console.WriteLine("No more pairs to merge. Stopping early.");
                    break;
                }

                // Find most frequent pair
                var bestPair = pairFreq.MaxBy(kvp => kvp.Value).Key;

                // Create merged token (concatenation)
                var merged = new byte[bestPair.Left.Length + bestPair.Right.Length];

                Buffer.BlockCopy(bestPair.Left, 0, merged, 0, bestPair.Left.Length);
                Buffer.BlockCopy(bestPair.Right, 0, merged, bestPair.Left.Length, bestPair.Right.Length);

                // Skip if already exists
                if (tokenToId.ContainsKey(merged))
                {
                    mergeCount++; // or track separately and break if all pairs are exhausted

                    continue;
                }

                // Record merge rule and add new token
                mergeRules.Add(bestPair);

                tokenToId[merged] = nextId;
                idToToken[nextId] = merged;

                nextId++;

                // Apply merge to all sequences
                for (var wordIndex = 0; wordIndex < wordByteSequences.Count; wordIndex++)
                {
                    var seq = wordByteSequences[wordIndex];
                    var newSeq = new List<byte[]>();
                    var i = 0;
                    while (i < seq.Count)
                    {
                        if (i < seq.Count - 1 && seq[i] == bestPair.Left && seq[i + 1] == bestPair.Right)
                        {
                            newSeq.Add(merged);
                            i += 2;
                        }
                        else
                        {
                            newSeq.Add(seq[i]);
                            i++;
                        }
                    }

                    wordByteSequences[wordIndex] = newSeq;
                }

                mergeCount++;
                currentVocabSize = idToToken.Count;

                // Report progress
                var percent = Math.Round(((double)mergeCount / maxMerges * 100), 2);

                Console.WriteLine($"Progress: {percent}% | Merges: {mergeCount}/{maxMerges} | Current vocab size: {currentVocabSize}");
            }

            var finalVocabSize = idToToken.Count;
            Console.WriteLine($"\nTraining complete.");
            Console.WriteLine($"Final vocabulary size: {finalVocabSize} (target was {targetVocabSize})");
            Console.WriteLine($"Total merges performed: {mergeRules.Count}");
            Console.WriteLine("Returning trained model.");

            return (idToToken, mergeRules);
        }
    }
}
