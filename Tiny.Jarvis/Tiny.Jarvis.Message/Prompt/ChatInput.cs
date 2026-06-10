using ChatMessage = Tiny.Jarvis.Message.Models.Message;

namespace Tiny.Jarvis.Message.Prompt
{
    public static class ChatInput
    {
        public static ChatMessage GetUserInput(string? promptName = null)
        {
            var userResponse = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userResponse))
            {
                Console.WriteLine("Input cannot be empty. Please enter a valid message.");
                return GetUserInput(promptName);
            }

            return new ChatMessage { From = promptName ?? "user", Content = userResponse, CreatedAt = DateTime.UtcNow };
        }
    }
}
