namespace Tiny.Jarvis.Genetic.Crossover
{
    /// <summary>
    /// Co‑existence crossover: averages the first segment (up to co‑existence point),
    /// then copies the tail from parentB after a random cut point, and applies mutation.
    /// </summary>
    public class CoexistenceCrossover : ICrossover
    {
        public void Crossover(int[] parentA, int[] parentB, double mutationProbability, int minGeneValue, int maxGeneValue, Random random)
        {
            if (parentA.Length != parentB.Length)
                throw new ArgumentException("Parent arrays must have the same length.");

            var length = parentA.Length;
            // Co‑existence cut point is assumed to be the middle of the chromosome (or could be a parameter)
            var coexistencePoint = length / 2;

            // Random cut point between coexistencePoint and length
            var cutPoint = random.Next(coexistencePoint, length);

            // Mutation rate: originally random between 1 and 4; we map it to a probability (0.0–1.0)
            // Original: mutate_rate = random(1,4) → 1..4. We'll use the method's mutationProbability as before.
            // But to keep the original intent, we'll compute a simple integer threshold:
            var mutateRate = random.Next(1, 5); // 1..4 inclusive

            // Average the first segment (0 to coexistencePoint-1)
            for (var i = 0; i < coexistencePoint; i++)
                parentA[i] = (parentA[i] + parentB[i]) / 2;

            // Copy tail from parentB from cutPoint to the end
            for (var i = cutPoint; i < length; i++)
                parentA[i] = parentB[i];

            // Mutation: each gene may be replaced with a random value
            var mutationProb = Math.Clamp(mutationProbability, 0.0, 1.0);
            var threshold = (int)(mutationProb * 100);
            for (var i = 0; i < length; i++)
            {
                var chance = random.Next(1, 101); // 1..100
                if (chance <= threshold)
                    parentA[i] = random.Next(minGeneValue, maxGeneValue + 1);
            }
        }
    }
}
