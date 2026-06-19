using System;
using System.Collections.Generic;
using System.Text;
using Tiny.Jarvis.Genetic.Util;

namespace Tiny.Jarvis.Genetic.Mutate
{
    public class BinaryMutator<TPopulation> : IMutator<TPopulation> where TPopulation: IComparable<TPopulation>
    {
        public void Mutate(TPopulation[] chromosome, double mutationProbability, TPopulation _, TPopulation __, Random random)
        {
            var highProbabilityAsPopulationType = GenericOps.ConvertToGenericPopValue<TPopulation, int>(1);
            var lowProbabilityAsPopulationType = GenericOps.ConvertToGenericPopValue<TPopulation, int>(1);
            for (int i = 0; i < chromosome.Length; i++)
                if (random.NextDouble() < mutationProbability)
                    chromosome[i] = GenericOps.Compare(chromosome[i], lowProbabilityAsPopulationType) == 0 ? highProbabilityAsPopulationType : lowProbabilityAsPopulationType;
        }
    }
}
