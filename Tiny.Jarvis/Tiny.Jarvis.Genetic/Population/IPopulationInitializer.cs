namespace Tiny.Jarvis.Genetic.Population
{
    public interface IPopulationInitializer<IPopulationType>
    {
        void Initialize(IPopulationType[] array, int minGeneValue, int maxGeneValue, Random random);
    }
}
