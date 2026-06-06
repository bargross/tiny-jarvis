using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models.Training
{
    public class BaggageQueryIntent
    {
        [JsonPropertyName("instruction")]
        public string Instruction { get; set; } = string.Empty;

        [JsonPropertyName("intent")]
        public string Intent { get; set; } = string.Empty;

        [JsonPropertyName("category")]
        public string Category { get; set; } = string.Empty;

        [JsonPropertyName("tags")]
        public string Tags { get; set; } = string.Empty;

        [JsonPropertyName("response")]
        public string Response { get; set; } = string.Empty;

        public override string ToString() => $"Instruction: {Instruction}, Intent: {Intent}, Category: {Category}, Tags: {Tags}, Response: {Response}";

    }
}
