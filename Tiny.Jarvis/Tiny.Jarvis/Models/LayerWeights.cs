namespace Tiny.Jarvis.Training.Models
{
    internal class LayerWeights
    {
        public Value[][] Query { get; set; }
        public Value[][] Key { get; set; }
        public Value[][] Value { get; set; }
        public Value[][] Output { get; set; }
        public Value[][] FeedForwardOne { get; set; }
        public Value[][] FeedForwardTwo { get; set; }
    }
}
