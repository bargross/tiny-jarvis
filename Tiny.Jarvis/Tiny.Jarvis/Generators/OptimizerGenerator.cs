using Tiny.Jarvis.Optimisers;
using Tiny.Jarvis.Training.Enums;
using Tiny.Jarvis.Training.Models;
using Tiny.Jarvis.Training.Optimization;

namespace Tiny.Jarvis.Training.Generators
{
    public static class OptimizerGenerator
    {
        public static IOptimizer GetOptimizer(OptimizerStrategy? strategy, IEnumerable<Value> parameters, double learningRate, int totalNumOfSteps, double momentum = 0.9, double weightDecay = 0.0)
        {
            if (strategy == null)
            {
                throw new ArgumentNullException(nameof(strategy));
            }

            return strategy switch
            {
                OptimizerStrategy.Adam => new AdamOptimiser(parameters, learningRate, totalNumOfSteps),
                OptimizerStrategy.SGDMomentum => new SGDMomentumOptimiser(parameters, learningRate, momentum, weightDecay),
                _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, null)
            };
        }
    }
}
