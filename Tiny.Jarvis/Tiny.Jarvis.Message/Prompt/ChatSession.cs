using Tiny.Jarvis.Message.Models;
using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Tokenization;

namespace Tiny.Jarvis.Message.Prompt
{
    public class ChatSession(TinyJarvisModel model, ITokenizer tokenizer)
    {
        private readonly List<ConversationExchange> _history = new();

        private bool _running = true;  // optional if you want external control

        public void Run()
        {
            Console.WriteLine("Chat started. Type '/end' or '/exit' to stop.\n");

            while (_running)
            {
                Console.Write("You: ");
                var conversationExchange = new ConversationExchange
                {
                    UserPrompt = ChatInput.GetUserInput("user"),
                };

                // Termination check
                if (string.IsNullOrEmpty(conversationExchange.UserPrompt.Content) ||
                    conversationExchange.UserPrompt.Content.Equals("/end", StringComparison.OrdinalIgnoreCase) ||
                    conversationExchange.UserPrompt.Content.Equals("/exit", StringComparison.OrdinalIgnoreCase))
                {
                    _running = !ChatOutput.ShouldEnd(_history);

                    if (!_running) continue;  
                }

                _history.Add(conversationExchange);

                Console.WriteLine($"Prompt: {conversationExchange.ToString()}");
                Console.WriteLine(Environment.NewLine);

                // Build prompt with history
                var botPromptName = "assistant";
                var prompt = string.Join(Environment.NewLine, _history.Select(x => x.ToString())) + $" {botPromptName}:";

                // Get encoded sequence with Bos at the beginning
                var tokens = tokenizer.Encode(prompt);

                //Console.WriteLine($"BOS token: {tokenizer.BOS}"); // debug
                //Console.WriteLine($"tokens before Generate is called: {string.Join(",", tokens)}"); // debug

                var responseTokens = model.Generate(tokens, maxNewTokens: 100, temperature: 0.8, topK: 50, topP: 0.95);

                //Console.WriteLine($"Generated response tokens: {string.Join(",", responseTokens)}"); // debug
                var response = tokenizer.Decode(responseTokens);

                //Console.WriteLine($"Generated response: {response}"); // debug
                // Clean and display
                response = CleanResponse(response);
                
                ChatOutput.Reply(response, _history, botPromptName);
            }
        }

        private string CleanResponse(string response)
        {
            // Stop if model starts generating a new user/assistant tag
            int idx = response.IndexOf("user:", StringComparison.OrdinalIgnoreCase);
            if (idx >= 0) response = response.Substring(0, idx);

            idx = response.IndexOf("assistant:", StringComparison.OrdinalIgnoreCase);
            if (idx >= 0) response = response.Substring(0, idx);

            return response.Trim();
        }
    }
}
