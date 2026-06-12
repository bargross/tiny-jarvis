using System;
using System.Collections.Generic;
using System.Text;

namespace Tiny.Jarvis.Genetic.Mutate
{
    public interface IMutator
    {
        /// <summary>
        /// Mutates a chromosome in place by randomly changing each gene with a given probability.
        /// </summary>
        /// <param name="chromosome">The chromosome (array of integers) to mutate.</param>
        /// <param name="mutationProbability">Probability (0.0–1.0) that each gene is mutated.</param>
        /// <param name="minGeneValue">Minimum allowed integer value (inclusive).</param>
        /// <param name="maxGeneValue">Maximum allowed integer value (inclusive).</param>
        void Mutate(int[] chromosome, double mutationProbability, int minGeneValue, int maxGeneValue, Random random);
    }
}
