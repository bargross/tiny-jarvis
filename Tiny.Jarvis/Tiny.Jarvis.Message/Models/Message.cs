namespace Tiny.Jarvis.Message.Models
{
    public class Message
    {
        public string Role { get; set; }
        public string Content { get; set; }
        public DateTime CreatedAt { get; set; }

        public override string ToString() => $"{Role}: {Content}";
    }
}
