using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Optimization;

internal class AdamOptimiser: IOptimizer
{
    // make the below optional with these as defaults
    private const double MomentumSmoothing = 0.85;
    private const double SquaredGradSmoothing = 0.99;
    private const double Epsilon = 1e-8;

    private readonly List<Value> _parameters;
    private readonly double[] _momentum;
    private readonly double[] _squaredGradAvg;
    private readonly double _baseLearningRate;
    private readonly int _totalSteps;

    public AdamOptimiser(IEnumerable<Value> parameters, double learningRate, int totalSteps)
    {
        var paramCount = parameters.Count();

        _parameters = [..parameters];
        _momentum = new double[paramCount];
        _squaredGradAvg = new double[paramCount];
        _baseLearningRate = learningRate;
        _totalSteps = totalSteps;
    }

    // Reset every parameter's gradient to zero. Call before each Backward.
    public void ZeroGrad() => _parameters.ForEach(param => param.Grad = 0.0);

    // Apply one Adam update to every parameter using its current Grad.
    public void Step(int? computedStep = null)
    {
        if (computedStep is null) throw new ArgumentException("Missing step");

        var step = computedStep.Value;

        // Compute the total L2 norm of all gradients
        var gradNormSq = 0.0;
        foreach (var param in _parameters)
            gradNormSq += param.Grad * param.Grad;

        var gradNorm = Math.Sqrt(gradNormSq);

        // If norm exceeds threshold, scale all gradients
        var maxNorm = 1.0;   // common value, can be tuned
        if (gradNorm > maxNorm)
        {
            var scale = maxNorm / gradNorm;
            foreach (var param in _parameters)
                param.Grad *= scale;
        }

        var currentLearningRate = _baseLearningRate * (1 - (double)step / _totalSteps);
        for (var i = 0; i < _parameters.Count; i++)
        {
            Value p = _parameters[i];
            _momentum[i] = MomentumSmoothing * _momentum[i] + (1 - MomentumSmoothing) * p.Grad;
            
            _squaredGradAvg[i] = SquaredGradSmoothing * _squaredGradAvg[i] + (1 - SquaredGradSmoothing) * Math.Pow(p.Grad, 2);

            var correctedMomentum = _momentum[i] / (1 - Math.Pow(MomentumSmoothing, step + 1));
            var correctedSquaredGrad = _squaredGradAvg[i] / (1 - Math.Pow(SquaredGradSmoothing, step + 1));
            
            p.Data -= currentLearningRate * correctedMomentum / (Math.Sqrt(correctedSquaredGrad) + Epsilon);
        }
    }
}