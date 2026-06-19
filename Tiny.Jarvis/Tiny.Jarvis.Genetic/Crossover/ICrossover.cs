namespace Tiny.Jarvis.Genetic.Crossover
{
    public interface ICrossover<TPopulation>
    {
        /// <summary>
        /// Applies crossover to modify parentA in place.
        /// </summary>
        /// <param name="parentA">First parent (will be overwritten with the child).</param>
        /// <param name="parentB">Second parent (unchanged).</param>
        /// <param name="mutationProbability">Probability of mutating a random gene (0.0 to 1.0).</param>
        /// <param name="minGeneValue">Minimum allowed integer value for a gene (inclusive).</param>
        /// <param name="maxGeneValue">Maximum allowed integer value for a gene (inclusive).</param>
        void Crossover(TPopulation[] parentA, TPopulation[] parentB, double mutationProbability, TPopulation minGeneValue, TPopulation maxGeneValue, Random random);
    }
}
