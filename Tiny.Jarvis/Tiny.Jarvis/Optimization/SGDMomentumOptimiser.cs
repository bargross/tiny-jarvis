using Tiny.Jarvis.Training.Models;
using Tiny.Jarvis.Training.Optimization;

namespace Tiny.Jarvis.Optimisers
{
    /// <summary>
    /// Stochastic Gradient Descent with Momentum optimizer.
    /// </summary>
    public class SGDMomentumOptimiser: IOptimizer
    {
        private readonly List<Value> _parameters;
        private readonly double _learningRate;
        private readonly double _momentum;
        private readonly double _weightDecay;
        private readonly List<double> _velocities;

        /// <summary>
        /// Creates a new SGD with Momentum optimiser.
        /// </summary>
        /// <param name="parameters">List of trainable Value objects (model parameters).</param>
        /// <param name="learningRate">Step size (e.g., 0.001).</param>
        /// <param name="momentum">Momentum factor (e.g., 0.9).</param>
        /// <param name="weightDecay">L2 regularisation coefficient (e.g., 0.0001). Set to 0 to disable.</param>
        public SGDMomentumOptimiser(
            IEnumerable<Value> parameters,
            double learningRate = 0.001,
            double momentum = 0.9,
            double weightDecay = 0.0)
        {
            _parameters = parameters.ToList();
            _learningRate = learningRate;
            _momentum = momentum;
            _weightDecay = weightDecay;
            _velocities = new List<double>(new double[parameters.Count()]);
        }

        /// <summary>
        /// Resets all gradients to zero.
        /// </summary>
        public void ZeroGrad() => _parameters.ForEach(param => param.Grad = 0.0);
        
        /// <summary>
        /// Performs a single optimisation step: updates all parameters using SGD with Momentum.
        /// </summary>
        public void Step(int? step = null) // not required but is to match the interface
        {
            for (int i = 0; i < _parameters.Count; i++)
            {
                var param = _parameters[i];

                // Apply weight decay (if any)
                var grad = param.Grad;
                if (_weightDecay != 0.0)
                    grad += _weightDecay * param.Data;

                // Update velocity: v = momentum * v - learningRate * grad
                _velocities[i] = _momentum * _velocities[i] - _learningRate * grad;

                // Update parameter: param = param + v
                param.Data += _velocities[i];
            }
        }
    }
}