using Tiny.Jarvis.Genetic.Util;

namespace Tiny.Jarvis.Genetic.Mutate
{
    public class StandardMutator<TPopulation> : IMutator<TPopulation>
    {
        public void Mutate(TPopulation[] chromosome, double mutationProbability, TPopulation minGeneValue, TPopulation maxGeneValue, Random random)
        {
            if (chromosome == null)
                throw new ArgumentNullException(nameof(chromosome));

            if (mutationProbability < 0.0 || mutationProbability > 1.0)
                throw new ArgumentOutOfRangeException(nameof(mutationProbability), "Probability must be between 0.0 and 1.0.");

            for (var i = 0; i < chromosome.Length; i++)
                if (random.NextDouble() < mutationProbability)
                    chromosome[i] = GenericOps.GetNextByType(random, minGeneValue, maxGeneValue);
        }
    }
}
