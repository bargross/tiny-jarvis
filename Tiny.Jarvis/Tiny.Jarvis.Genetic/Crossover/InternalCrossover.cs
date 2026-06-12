using Tiny.Jarvis.Genetic.Helpers;

namespace Tiny.Jarvis.Genetic.Crossover
{
    /// <summary>
    /// Non‑standard crossover operator where strands are averaged.
    /// The operator swaps strands by averaging interval values of the chromosome.
    /// 
    /// Drawback: This technique reaches a cap at which the chromosome(s) cannot evolve further,
    /// causing the ranking mechanism to deem all individuals incapable (or all capable).
    /// To avoid premature convergence, a random strand swap and mutation have been added.
    /// </summary>
    public class InternalCrossover : ICrossover
    {
        public void Crossover(int[] parentA, int[] parentB, double mutationProbability, int minGeneValue, int maxGeneValue, Random random)
        {
            if (parentA.Length != parentB.Length)
                throw new ArgumentException("Parent arrays must have the same length.");

            var length = parentA.Length;
            var coexistanceCpoint = length / 2;  // original used a parameter; here we assume half the length
            var crossoverPoint = random.Next(coexistanceCpoint, length);
            var internalIndex = coexistanceCpoint - 1;
            var container = new int[length];

            // 1. Symmetric averaging on the first segment
            var reverseDistance = coexistanceCpoint - (coexistanceCpoint / 2);
            var linearDistance = coexistanceCpoint / 2;

            for (var i = 0; i < coexistanceCpoint && internalIndex >= 0; i++, internalIndex--)
            {
                if (i <= linearDistance && internalIndex >= reverseDistance)
                {
                    var sumA = parentA[i] + parentA[internalIndex];
                    var sumB = parentB[i] + parentB[internalIndex];

                    container[i] = (sumA + sumB) / 2;
                }
                else container[i] = parentA[i]; // fallback – keep parentA value
            }

            // 2. If container is different from parentA, use it; otherwise, fall back to simple averaging
            if (!ParentComparer.AreEqual(container, parentA))
                Array.Copy(container, parentA, length);

            else
            {
                // Fallback: average whole arrays until different from container
                while (true)
                {
                    for (var i = 0; i < length; i++)
                        parentA[i] = (parentA[i] + parentB[i]) / 2;

                    if (!ParentComparer.AreEqual(parentA, container))
                        break;
                }
            }

            // Copy tail from parentB after crossover point
            for (int i = crossoverPoint; i < length; i++)
                parentA[i] = parentB[i];

            // 4. Mutation: each gene may be replaced with a random value
            var mutationProb = Math.Clamp(mutationProbability, 0.0, 1.0);
            var mutateThreshold = (int)(mutationProb * 100); // roughly equivalent to original random(1,20) range
            for (var i = 0; i < length; i++)
            {
                var chance = random.Next(1, 101); // 1..100
                if (chance <= mutateThreshold)
                    parentA[i] = random.Next(minGeneValue, maxGeneValue + 1);
            }
        }
    }
}
