using ChatMessage = Tiny.Jarvis.Message.Models.Message;

namespace Tiny.Jarvis.Message.Prompt
{
    public static class ChatInput
    {
        public static ChatMessage GetUserInput()
        {
            var userResponse = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userResponse))
            {
                Console.WriteLine("Input cannot be empty. Please enter a valid message.");
                return GetUserInput();
            }

            return new ChatMessage { Role = "user", Content = userResponse, CreatedAt = DateTime.UtcNow };
        }
    }
}
