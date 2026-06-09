namespace Tiny.Jarvis.Message.Models
{
    public class Message
    {
        public string From { get; set; }
        public string Content { get; set; }
        public DateTime CreatedAt { get; set; }

        public override string ToString() => $"{From ?? ""}: {Content ?? ""}";
    }
}
