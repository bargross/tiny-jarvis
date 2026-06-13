using Tiny.Jarvis.Message.Models;
using ChatMessage = Tiny.Jarvis.Message.Models.Message;

namespace Tiny.Jarvis.Message.Prompt
{
    public class ChatOutput
    {
        public static void Reply(string response, List<ConversationExchange> history, string promptName)
        {
            var message = new ChatMessage
            {
                From = promptName,
                Content = response,
                CreatedAt = DateTime.UtcNow
            };

            var conversation = history.Last();

            conversation.AssistantResposne = message;

            Console.WriteLine(message.ToString());

            Console.WriteLine(Environment.NewLine);
        }

        public static bool ShouldEnd(List<ConversationExchange> history, string userPromptName)
        {
            var assistantResponse = new ChatMessage
            {
                From = "assistant",
                Content = "End chant? (y/n)",
                CreatedAt = DateTime.UtcNow
            };

            Console.WriteLine("End chant? (y/n)");

            var exchange = new ConversationExchange 
            { 
                UserPrompt = ChatInput.GetUserInput(), 
                AssistantResposne = assistantResponse 
            };

            history.Add(exchange);

            var inputResponse = exchange.UserPrompt.Content.Trim().ToLower();
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
