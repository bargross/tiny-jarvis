using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models
{
    public class BaggageQueryIntentData
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

        public override string ToString() => $"Instruction: {Instruction} {Environment.NewLine} Response: {Response}";
        public (string userPromptName, string botResponsePromptName) GetPromptNames => (nameof(Instruction), nameof(Response));

    }
}
