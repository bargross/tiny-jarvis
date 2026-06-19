namespace Tiny.Jarvis.Training.Models
{
    public class LayerWeights
    {
        public Scalar[][] Query { get; set; }
        public Scalar[][] Key { get; set; }
        public Scalar[][] Value { get; set; }
        public Scalar[][] Output { get; set; }
        public Scalar[][] FeedForwardOne { get; set; }
        public Scalar[][] FeedForwardTwo { get; set; }
    }
}
