using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models
{
    public class ChatAlpacaConversation
    {
        [JsonPropertyName("id")]
        public string Id { get; set; }

        [JsonPropertyName("conversations")]
        public List<ChatAlpacaTurn> Conversations { get; set; }
    }
}
