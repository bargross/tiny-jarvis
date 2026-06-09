using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models
{
    public class PromptResponseData
    {
        [JsonPropertyName("prompt")]
        public string Prompt { get; set; }

        [JsonPropertyName("response")]
        public string Response { get; set; }

        //[JsonPropertyName("helpfulness")]
        //public int Helpfulness { get; set; }

        //[JsonPropertyName("correctness")]
        //public int Correctness { get; set; }

        //[JsonPropertyName("coherence")]
        //public int Coherence { get; set; }

        //[JsonPropertyName("complexity")]
        //public int Complexity { get; set; }

        //[JsonPropertyName("verbosity")]
        //public int Verbosity { get; set; }

        public override string ToString() => $"user: {Prompt}, resposne: {Response}";
    }
}
