namespace Tiny.Jarvis.Genetic.Population
{
    /// <summary>
    /// Initialises a population (chromosome) by filling it with random integer values.
    /// </summary>
    public class RandomPopulationInitializer<IPopulationType> : IPopulationInitializer<IPopulationType> where IPopulationType : struct
    {
        public void Initialize(IPopulationType[] array, int minGeneValue, int maxGeneValue, Random random) 
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));
            if (minGeneValue > maxGeneValue)
                throw new ArgumentException("minGeneValue must be less than or equal to maxGeneValue");

            switch(typeof(IPopulationType))
            {
                case Type intType when intType == typeof(int):
                    for (int i = 0; i < array.Length; i++)
                        array[i] = (IPopulationType)(object)random.Next(minGeneValue, maxGeneValue + 1);

                    break;
                case Type intType when intType == typeof(double):
                    for (int i = 0; i < array.Length; i++)
                        array[i] = (IPopulationType)(object)random.NextDouble();

                    break;
                case Type intType when intType == typeof(float):
                    for (int i = 0; i < array.Length; i++)
                        array[i] = (IPopulationType)(object)random.NextDouble();

                    break;
                case Type intType when intType == typeof(decimal):
                    for (int i = 0; i < array.Length; i++)
                        array[i] = (IPopulationType)(object)random.NextDouble();

                    break;
                case Type intType when intType == typeof(long):
                    for (int i = 0; i < array.Length; i++)
                        array[i] = (IPopulationType)(object)random.Next(minGeneValue, maxGeneValue + 1);

                    break;

                default: throw new InvalidOperationException("Type not permitted");
            }
        }
    }
}
