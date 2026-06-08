using Tiny.Jarvis.Message.Enums;
using ChatMessage = Tiny.Jarvis.Message.Models.Message;

namespace Tiny.Jarvis.Message.Prompt
{
    public class ChatOutput
    {
        private static int _outputDelay = 250;

        public static void Reply(string response, List<ChatMessage> history)
        {
            var message = new ChatMessage
            {
                Role = Role.Assistant.ToString(),
                Content = response,
                CreatedAt = DateTime.UtcNow
            };

            history.Add(message);

            var responseParts = response.Split(' ');

            Console.Write($"{message.Role}: ");
            foreach (var responsePart in responseParts)
            {
                Console.Write($" {responsePart}");

                Thread.Sleep(_outputDelay);
            }

            Console.Write(Environment.NewLine);

            //Console.WriteLine(message.ToString());
        }

        public static bool ShouldEnd(List<ChatMessage> history)
        {
            var endMessage = new ChatMessage
            {
                Role = Role.Assistant.ToString(),
                Content = "End chant? (y/n)",
                CreatedAt = DateTime.UtcNow
            };

            Console.WriteLine("End chant? (y/n)");
            
            var userInput = ChatInput.GetUserInput();

            history.Add(endMessage);

            var inputResponse = userInput.Content.Trim().ToLower();
            if (inputResponse == "y")
            {
                Console.WriteLine("Chat ended.");
                return true;
            }
            else
            {
                Console.WriteLine("Continuing chat...");
                return false;
            }
        }
    }
}
