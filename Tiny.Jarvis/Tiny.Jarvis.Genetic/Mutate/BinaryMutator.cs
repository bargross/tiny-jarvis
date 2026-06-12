using System;
using System.Collections.Generic;
using System.Text;

namespace Tiny.Jarvis.Genetic.Mutate
{
    public class BinaryMutator : IMutator
    {
        public void Mutate(int[] chromosome, double mutationProbability, int _, int __, Random random)
        {
            for (int i = 0; i < chromosome.Length; i++)
                if (random.NextDouble() < mutationProbability)
                    chromosome[i] = chromosome[i] == 0 ? 1 : 0;
        }
    }
}
