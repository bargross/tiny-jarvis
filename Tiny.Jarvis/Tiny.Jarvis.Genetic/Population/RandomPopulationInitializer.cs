using Tiny.Jarvis.Genetic.Util;

namespace Tiny.Jarvis.Genetic.Population
{
    /// <summary>
    /// Initialises a population (chromosome) by filling it with random integer values.
    /// </summary>
    public class RandomPopulationInitializer<IPopulationType> : IPopulationInitializer<IPopulationType> where IPopulationType: IComparable<IPopulationType>
    {
        public void Initialize(IPopulationType[] array, IPopulationType? minGeneValue, IPopulationType? maxGeneValue, Random random) 
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            if (GenericOps.Compare(minGeneValue, maxGeneValue) > 0)
                throw new ArgumentException("minGeneValue must be less than or equal to maxGeneValue");

            for (int i = 0; i < array.Length; i++)
                array[i] = GenericOps.GetNextByType(random, minGeneValue, maxGeneValue);
        }
    }
}
