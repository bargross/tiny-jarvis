using Tiny.Jarvis.Optimisers;
using Tiny.Jarvis.Training.Enums;
using Tiny.Jarvis.Training.Optimization;

namespace Tiny.Jarvis.Training.Serializers
{
    public static class OptimizerSerializer
    {
        public static void Save(IOptimizer optimizer, string filePath)
        {
            var state = optimizer.GetState();
            using var writer = new BinaryWriter(File.Open(filePath, FileMode.Create));

            if (state.Step.HasValue) writer.Write(state.Step.Value);

            if (state.Momentum is not null)
            {
                writer.Write(state.Momentum.Length);

                foreach (var m in state.Momentum) 
                    writer.Write(m);

                foreach (var v in state.SquaredGradAvg)
                    writer.Write(v);
            }

            if (state.Velocities is not null)
            {
                writer.Write(state.Velocities.Length);

                foreach (var m in state.Velocities)
                    writer.Write(m);
            }
        }

        public static IOptimizer Load(string filePath, OptimizerStrategy strategy, double learningRate, int totalSteps, double maxGradNorm = 1.0, double momentum = 0.9, double weightDecay = 0.0)
        {
            using var reader = new BinaryReader(File.OpenRead(filePath));
            
            var step = reader.ReadInt32();
            var length = reader.ReadInt32();
            var momentums = new double[length];
            var squaredGradAvg = new double[length];
            var velocities = new double[length];

            if (strategy == OptimizerStrategy.Adam) 
            {
                for (int i = 0; i < length; i++)
                    momentums[i] = reader.ReadDouble();

                for (int i = 0; i < length; i++) 
                    squaredGradAvg[i] = reader.ReadDouble();
            }
            else for (int i = 0; i < length; i++)
                    velocities[i] = reader.ReadDouble();

            return strategy switch
            {
                OptimizerStrategy.Adam => new AdamOptimiser(step, momentums, squaredGradAvg, learningRate, totalSteps, maxGradNorm),
                OptimizerStrategy.SGDMomentum => new SGDMomentumOptimiser(step, velocities, learningRate, momentum, weightDecay)
            };
        }
    }
}
