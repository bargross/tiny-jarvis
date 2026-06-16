using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Optimization
{
    public interface IOptimizer
    {
        OptimizerState GetState();
        void SetParameters(List<Value> parameters);
        void ZeroGrad(); 
        void Step(int? step = null);
    }
}
