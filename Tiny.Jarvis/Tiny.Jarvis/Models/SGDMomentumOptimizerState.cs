namespace Tiny.Jarvis.Training.Models
{
    public class SGDMomentumOptimizerState
    {
        //public int Step { get; set; } // not needed for the SGD current implementation
        public double[] Velocities { get; set; }
    }
}
