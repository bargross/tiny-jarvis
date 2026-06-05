using Tiny.Jarvis.Models;

namespace Tiny.Jarvis.Optimization;

// Encapsulates the Adam update from Chapter 7 (momentum, squared-gradient
// average, bias correction) along with the linear learning-rate decay used
// across the course. See Chapter 7 for the underlying maths.
internal class AdamOptimiser
{
    private const double MomentumSmoothing = 0.85;
    private const double SquaredGradSmoothing = 0.99;
    private const double Epsilon = 1e-8;

    private readonly IReadOnlyList<Value> _parameters;
    private readonly double[] _momentum;
    private readonly double[] _squaredGradAvg;
    private readonly double _baseLearningRate;
    private readonly int _totalSteps;

    public AdamOptimiser(IReadOnlyList<Value> parameters, double learningRate, int totalSteps)
    {
        _parameters = parameters;
        _momentum = new double[parameters.Count];
        _squaredGradAvg = new double[parameters.Count];
        _baseLearningRate = learningRate;
        _totalSteps = totalSteps;
    }

    // Reset every parameter's gradient to zero. Call before each Backward.
    public void ZeroGrad()
    {
        foreach (Value value in _parameters)
        {
            value.Grad = 0;
        }
    }

    // Apply one Adam update to every parameter using its current Grad.
    public void Step(int step)
    {
        var currentLearningRate = _baseLearningRate * (1 - (double)step / _totalSteps);
        for (int i = 0; i < _parameters.Count; i++)
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