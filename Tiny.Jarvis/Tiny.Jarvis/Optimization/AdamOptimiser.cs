using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Optimization;

public class AdamOptimiser: IOptimizer
{
    // make the below optional with these as defaults
    private const double MomentumSmoothing = 0.85;
    private const double SquaredGradSmoothing = 0.99;
    private const double Epsilon = 1e-8;

    private List<Scalar> _parameters;
    private readonly double[] _momentum;
    private readonly double[] _squaredGradAvg;
    private readonly double _baseLearningRate;
    private readonly int _totalSteps;
    private readonly double _maxGradNorm;
    private int _step = 0;

    public int CurrentStep => _step;

    public AdamOptimiser(IEnumerable<Scalar> parameters, double learningRate, int totalSteps, double maxGradNorm = 1.0)
    {
        _parameters = parameters.ToList();
        var paramCount = _parameters.Count;

        _momentum = new double[paramCount];
        _squaredGradAvg = new double[paramCount];
        _baseLearningRate = learningRate;
        _totalSteps = totalSteps;
        _maxGradNorm = maxGradNorm;
    }

    public AdamOptimiser(int step, double[] momentum, double[] squaredGradAvg, double learningRate, int totalSteps, double maxGradNorm = 1.0)
    {
        _parameters = new List<Scalar>();

        _step = step;
        _momentum = momentum;
        _squaredGradAvg = squaredGradAvg;
        _baseLearningRate = learningRate;
        _totalSteps = totalSteps;
        _maxGradNorm = maxGradNorm;
    }

    public void ZeroGrad() => _parameters.ForEach(param => param.Grad = 0.0);

    // Apply one Adam update to every parameter using its current Grad.
    public void Step(int step)
    {
        _step = step;

        // Compute the total L2 norm of all gradients
        var gradNormSq = 0.0;
        foreach (var param in _parameters)
            gradNormSq += param.Grad * param.Grad;

        var gradNorm = Math.Sqrt(gradNormSq);

        // If norm exceeds threshold, scale all gradients
        if (gradNorm > _maxGradNorm)
        {
            var scale = _maxGradNorm / gradNorm;
            foreach (var param in _parameters)
                param.Grad *= scale;
        }

        var currentLearningRate = _baseLearningRate * (1 - (double)step / _totalSteps);
        var nextStep = step + 1; // always 1-indexed regardless of what caller passes
        var momentumCorrection = 1 - Math.Pow(MomentumSmoothing, nextStep);
        var squaredGradCorrection = 1 - Math.Pow(SquaredGradSmoothing, nextStep);

        for (var i = 0; i < _parameters.Count; i++)
        {
            Scalar p = _parameters[i];
            _momentum[i] = MomentumSmoothing * _momentum[i] + (1 - MomentumSmoothing) * p.Grad;
            
            _squaredGradAvg[i] = SquaredGradSmoothing * _squaredGradAvg[i] + (1 - SquaredGradSmoothing) * Math.Pow(p.Grad, 2);

            var correctedMomentum = _momentum[i] / momentumCorrection;
            var correctedSquaredGrad = _squaredGradAvg[i] / squaredGradCorrection;

            p.Data -= currentLearningRate * correctedMomentum / (Math.Sqrt(correctedSquaredGrad) + Epsilon);
        }
    }

    public OptimizerState GetState() => new OptimizerState 
    {
        Step = _step,
        Momentum = _momentum,
        SquaredGradAvg = _squaredGradAvg,
    };

    public void SetParameters(List<Scalar> parameters) => _parameters.AddRange(parameters);
}