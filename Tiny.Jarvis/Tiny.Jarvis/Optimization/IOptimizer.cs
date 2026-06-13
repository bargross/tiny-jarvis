namespace Tiny.Jarvis.Training.Optimization
{
    public interface IOptimizer
    {
        void ZeroGrad(); 
        void Step(int? step = null);
    }
}
