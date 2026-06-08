using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models
{
    public class GPTConversationData
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }

        [JsonPropertyName("conversations")]
        public List<ConversationEntry> Conversations { get; set; }

        public (string userPromptName, string botResponsePromptName) GetPromptNames => ("Human", "Assistant");

        public override string ToString() => string.Join($"{Environment.NewLine}", Conversations.Select(x => x.Value));
    }
}
