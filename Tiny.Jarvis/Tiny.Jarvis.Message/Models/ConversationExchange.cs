namespace Tiny.Jarvis.Message.Models
{
    public class ConversationExchange
    {
        public Message? UserPrompt { get; set; }
        public Message? AssistantResposne { get; set; }

        public override string ToString() => $"{UserPrompt?.ToString() ?? "user: "} {AssistantResposne?.ToString() ?? "assistant: "}";
    }
}
