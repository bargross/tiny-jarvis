namespace Tiny.Jarvis.Genetic.Mutate
{
    public class StandardMutator : IMutator
    {
        public void Mutate(int[] chromosome, double mutationProbability, int minGeneValue, int maxGeneValue, Random random)
        {
            if (chromosome == null)
                throw new ArgumentNullException(nameof(chromosome));

            if (mutationProbability < 0.0 || mutationProbability > 1.0)
                throw new ArgumentOutOfRangeException(nameof(mutationProbability), "Probability must be between 0.0 and 1.0.");

            for (var i = 0; i < chromosome.Length; i++)
                if (random.NextDouble() < mutationProbability)
                    chromosome[i] = random.Next(minGeneValue, maxGeneValue + 1);
        }
    }
}
