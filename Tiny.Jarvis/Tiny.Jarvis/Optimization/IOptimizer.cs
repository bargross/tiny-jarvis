using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Optimization
{
    public interface IOptimizer
    {
        int CurrentStep { get; }
        OptimizerState GetState();
        void SetParameters(List<Value> parameters);
        void ZeroGrad(); 
        void Step(int step);
    }
}
