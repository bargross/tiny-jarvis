using System;
using System.Collections.Generic;
using System.Text;

namespace Tiny.Jarvis.Genetic.Population
{
    public interface IPopulationInitializer
    {
        void Initialize(int[] array, int minGeneValue, int maxGeneValue, Random random);
    }
}
