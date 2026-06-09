using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models
{
    public class PickleDocument
    {
        [JsonPropertyName("sentences")]
        public List<List<string>> Sentences { get; set; }
    }
}
