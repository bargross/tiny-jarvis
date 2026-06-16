namespace Tiny.Jarvis.Training.Models
{
    public class OptimizerState
    {
        public int? Step { get; set; }
        public double[]? Momentum { get; set; }
        public double[]? SquaredGradAvg { get; set; }
        public double[]? Velocities { get; set; }
    }
}
