using System;
using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models
{
    public class ConversationEntry
    {
        [JsonPropertyName("from")]
        public string From { get; set; }

        [JsonPropertyName("value")]
        public string Value { get; set; }
    }
}
