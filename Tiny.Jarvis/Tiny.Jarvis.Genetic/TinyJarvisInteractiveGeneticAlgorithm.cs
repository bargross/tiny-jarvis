using Tiny.Jarvis.Genetic.Crossover;
using Tiny.Jarvis.Genetic.Models;
using Tiny.Jarvis.Genetic.Mutate;
using Tiny.Jarvis.Genetic.Population;
using Tiny.Jarvis.Genetic.Roulette;

namespace Tiny.Jarvis.Genetic
{
    public class TinyJarvisInteractiveGeneticAlgorithm<TPopulation> where TPopulation : IComparable<TPopulation>
    {
        private readonly IReadOnlyDictionary<CrossoverType, ICrossover<TPopulation>> _crossovers;
        private readonly IMutator<TPopulation> _mutator;
        private readonly IPopulationInitializer<TPopulation> _initializer;
        private readonly IRouletteSelector _roulette;
        private readonly Random _random;

        public CrossoverType CrossoverType { get; set; } = CrossoverType.Average;
        public double CrossoverProbability { get; set; } = 0.8;
        public double MutationProbability { get; set; } = 0.05;
        public TPopulation MinGeneValue { get; set; }
        public TPopulation MaxGeneValue { get; set; }
        public int EliteCount { get; set; } = 1;
        public int PopulationSize { get; set; }
        public int ChromosomeLength { get; set; }
        public int MaxGenerations { get; set; }

        // Problem-specific delegates
        public Func<TPopulation[], double> FitnessFunction { get; set; } = null!;
        public Func<int, double, TPopulation[][], bool> TerminationCondition { get; set; } = (_, _, _) => false;

        public TinyJarvisInteractiveGeneticAlgorithm(
            IReadOnlyDictionary<CrossoverType, ICrossover<TPopulation>> crossovers,
            IMutator<TPopulation>? mutator = null,
            IPopulationInitializer<TPopulation>? initializer = null,
            IRouletteSelector? roulette = null,
            Random? random = null)
        {
            _crossovers = crossovers ?? throw new ArgumentNullException(nameof(crossovers));
            _mutator = mutator ?? new StandardMutator<TPopulation>();
            _initializer = initializer ?? new RandomPopulationInitializer<TPopulation>();
            _roulette = roulette ?? new RouletteSelector();
            _random = random ?? Random.Shared;
        }

        public TPopulation[] Run()
        {
            // Initialise population
            var population = new TPopulation[PopulationSize][];
            for (var i = 0; i < PopulationSize; i++)
            {
                population[i] = new TPopulation[ChromosomeLength];
                _initializer.Initialize(population[i], MinGeneValue, MaxGeneValue, _random);
            }

            var fitness = new double[PopulationSize];
            var generation = 0;

            while (true)
            {
                // Evaluate
                var totalFitness = 0.0;
                for (var i = 0; i < PopulationSize; i++)
                {
                    fitness[i] = FitnessFunction(population[i]);
                    totalFitness += fitness[i];
                }

                // Best so far
                var bestIndex = Array.IndexOf(fitness, fitness.Max());
                var bestFitness = fitness[bestIndex];
                var bestChromosome = (TPopulation[])population[bestIndex].Clone();

                // Termination
                if (generation >= MaxGenerations || TerminationCondition(generation, bestFitness, population))
                    return bestChromosome;

                // Next generation
                var newPopulation = new TPopulation[PopulationSize][];
                var eliteIndices = GetTopIndices(fitness, EliteCount);
                for (var i = 0; i < EliteCount; i++)
                    newPopulation[i] = (TPopulation[])population[eliteIndices[i]].Clone();

                var newIdx = EliteCount;
                var crossover = _crossovers[CrossoverType];

                while (newIdx < PopulationSize)
                {
                    var parentAIdx = _roulette.Select(fitness, totalFitness, _random);
                    var parentBIdx = _roulette.Select(fitness, totalFitness, _random);

                    var childA = (TPopulation[])population[parentAIdx].Clone();
                    var childB = (TPopulation[])population[parentBIdx].Clone();

                    if (_random.NextDouble() < CrossoverProbability)
                        crossover.Crossover(childA, childB, MutationProbability, MinGeneValue, MaxGeneValue, _random);

                    else
                    {
                        // No crossover – still mutate independently
                        _mutator.Mutate(childA, MutationProbability, MinGeneValue, MaxGeneValue, _random);
                        _mutator.Mutate(childB, MutationProbability, MinGeneValue, MaxGeneValue, _random);
                    }

                    newPopulation[newIdx++] = childA.Cast<TPopulation>().ToArray();
                    if (newIdx < PopulationSize)
                        newPopulation[newIdx++] = childB.Cast<TPopulation>().ToArray();
                }

                population = newPopulation;
                generation++;
            }
        }

        private static int[] GetTopIndices(double[] fitness, int count)
        {
            return fitness.Select((v, i) => (v, i))
                          .OrderByDescending(x => x.v)
                          .Take(count)
                          .Select(x => x.i)
                          .ToArray();
        }
    }
}
