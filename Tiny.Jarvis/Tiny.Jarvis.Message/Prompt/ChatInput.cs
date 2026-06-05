using ChatMessage = Tiny.Jarvis.Message.Models.Message;

namespace Tiny.Jarvis.Message.Prompt
{
    public static class ChatInput
    {
        public static ChatMessage GetUserInput(List<ChatMessage> history)
        {
            var userResponse = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userResponse))
            {
                Console.WriteLine("Input cannot be empty. Please enter a valid message.");
                return GetUserInput(history);
            }

            var message = new ChatMessage { Role = "user", Content = userResponse, CreatedAt = DateTime.UtcNow };

            history.Add(message);

            return message;
        }
    }
}
