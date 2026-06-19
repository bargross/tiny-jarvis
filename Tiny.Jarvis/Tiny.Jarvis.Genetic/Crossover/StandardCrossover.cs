using Tiny.Jarvis.Genetic.Helpers;
using Tiny.Jarvis.Genetic.Util;

namespace Tiny.Jarvis.Genetic.Crossover
{
    public class AverageCrossover<TPopulation> : ICrossover<TPopulation>
    {
        public void Crossover(TPopulation[] parentA, TPopulation[] parentB, double mutationProbability, TPopulation minGeneValue, TPopulation maxGeneValue, Random random)
        {
            if (parentA.Length != parentB.Length)
                throw new ArgumentException("Parent arrays must have the same length.");

            var length = parentA.Length;
            var originalA = (TPopulation[])parentA.Clone(); // snapshot of original parentA
            var divisibleNumberOfParents = GenericOps.ConvertToGenericPopValue<TPopulation, int>(2);

            // 1. Average crossover (integer division truncates)
            for (var i = 0; i < length; i++)
            {
                var sumOfChromosomes = GenericOps.PerformMathematicalOperation(parentA[i], parentB[i], Enums.MathOperation.Addition);

                parentA[i] = GenericOps.PerformMathematicalOperation(sumOfChromosomes, divisibleNumberOfParents, Enums.MathOperation.Division);
            }

            // 2. Mutation: with given probability, replace a random gene with a new random value
            if (random.NextDouble() < mutationProbability)
            {
                var index = random.Next(length);
                parentA[index] = GenericOps.GetNextByType(random, minGeneValue, maxGeneValue);
            }

            // 3. If the child is identical to the original parentA OR to parentB, force a change
            if (ParentComparer.AreEqual(parentA, originalA) || ParentComparer.AreEqual(parentA, parentB))
            {
                while (true)
                {
                    var index = random.Next(length);
                    parentA[index] = GenericOps.GetNextByType(random, minGeneValue, maxGeneValue);
                    if (!ParentComparer.AreEqual(parentA, originalA) && !ParentComparer.AreEqual(parentA, parentB))
                        break;
                }
            }
        }
    }


    //     public class AverageCrossover<IPopulationType> : ICrossover<IPopulationType> where IPopulationType: struct
    
    //    {
    //        public void Crossover(IPopulationType[] parentA, IPopulationType[] parentB, double mutationProbability, int minGeneValue, int maxGeneValue, Random random)
    //    {
    //        if (parentA.Length != parentB.Length)
    //            throw new ArgumentException("Parent arrays must have the same length.");

    //        var length = parentA.Length;
    //        var originalA = (IPopulationType[])parentA.Clone(); // snapshot of original parentA

    //        // 1. Average crossover (integer division truncates)
    //        //for (var i = 0; i < length; i++)
    //        //    parentA[i] = (parentA[i] + parentB[i]) / 2;
    //        //if ()
    //        parentA = Convert(parentA, parentB, (int v, int c) => (IPopulationType)(object)(v + c / 2));

    //        // 2. Mutation: with given probability, replace a random gene with a new random value
    //        if (random.NextDouble() < mutationProbability)
    //        {
    //            var index = random.Next(length);
    //            //parentA[index] = random.Next(minGeneValue, maxGeneValue + 1);
    //        }

    //        // 3. If the child is identical to the original parentA OR to parentB, force a change
    //        if (ParentComparer.AreEqual(parentA, originalA) || ParentComparer.AreEqual(parentA, parentB))
    //        {
    //            while (true)
    //            {
    //                var index = random.Next(length);
    //                //parentA[index] = random.Next(minGeneValue, maxGeneValue + 1);
    //                if (!ParentComparer.AreEqual(parentA, originalA) && !ParentComparer.AreEqual(parentA, parentB))
    //                    break;
    //            }
    //        }
    //    }

    //    //public 

    //    public IPopulationType[] Convert(IPopulationType[] containerA, IPopulationType[] containerB, Func<int, int, IPopulationType> mutator) =>
    //        containerA.Select(((value, idx) => mutator.Invoke((int)(object)containerA[idx], (int)(object)containerB[idx]))).ToArray();
    //    public IPopulationType[] Convert(IPopulationType[] containerA, IPopulationType[] containerB, Func<double, double, IPopulationType> mutator) =>
    //        containerA.Select(((value, idx) => mutator.Invoke((double)(object)containerA[idx], (double)(object)containerB[idx]))).ToArray();
    //    public IPopulationType[] Convert(IPopulationType[] containerA, IPopulationType[] containerB, Func<float, float, IPopulationType> mutator) =>
    //        containerA.Select(((value, idx) => mutator.Invoke((float)(object)containerA[idx], (float)(object)containerB[idx]))).ToArray();
    //    public IPopulationType[] Convert(IPopulationType[] containerA, IPopulationType[] containerB, Func<long, long, IPopulationType> mutator) =>
    //        containerA.Select(((value, idx) => mutator.Invoke((long)(object)containerA[idx], (long)(object)containerB[idx]))).ToArray();
    //}
}
