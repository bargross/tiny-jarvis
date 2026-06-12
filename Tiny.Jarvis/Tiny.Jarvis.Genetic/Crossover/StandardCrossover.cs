using Tiny.Jarvis.Genetic.Helpers;

namespace Tiny.Jarvis.Genetic.Crossover
{
    public class AverageCrossover : ICrossover
    {
        public void Crossover(int[] parentA, int[] parentB, double mutationProbability, int minGeneValue, int maxGeneValue, Random random)
        {
            if (parentA.Length != parentB.Length)
                throw new ArgumentException("Parent arrays must have the same length.");

            var length = parentA.Length;
            var originalA = (int[])parentA.Clone(); // snapshot of original parentA

            // 1. Average crossover (integer division truncates)
            for (var i = 0; i < length; i++)
                parentA[i] = (parentA[i] + parentB[i]) / 2;

            // 2. Mutation: with given probability, replace a random gene with a new random value
            if (random.NextDouble() < mutationProbability)
            {
                var index = random.Next(length);
                parentA[index] = random.Next(minGeneValue, maxGeneValue + 1);
            }

            // 3. If the child is identical to the original parentA OR to parentB, force a change
            if (ParentComparer.AreEqual(parentA, originalA) || ParentComparer.AreEqual(parentA, parentB))
            {
                while (true)
                {
                    var index = random.Next(length);
                    parentA[index] = random.Next(minGeneValue, maxGeneValue + 1);
                    if (!ParentComparer.AreEqual(parentA, originalA) && !ParentComparer.AreEqual(parentA, parentB))
                        break;
                }
            }
        }
    }
}
