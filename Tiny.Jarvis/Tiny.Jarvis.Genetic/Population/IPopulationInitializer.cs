namespace Tiny.Jarvis.Genetic.Population
{
    public interface IPopulationInitializer<IPopulationType>
    {
        void Initialize(IPopulationType[] array, IPopulationType? minGeneValue, IPopulationType? maxGeneValue, Random random);
    }
}
