namespace Tiny.Jarvis.Genetic.Population
{
    /// <summary>
    /// Initialises a population (chromosome) by filling it with random integer values.
    /// </summary>
    public class RandomPopulationInitializer : IPopulationInitializer
    {
        public void Initialize(int[] array, int minGeneValue, int maxGeneValue, Random random) 
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));
            if (minGeneValue > maxGeneValue)
                throw new ArgumentException("minGeneValue must be less than or equal to maxGeneValue");

            for (int i = 0; i < array.Length; i++)
                array[i] = random.Next(minGeneValue, maxGeneValue + 1);
        }
    }
}
