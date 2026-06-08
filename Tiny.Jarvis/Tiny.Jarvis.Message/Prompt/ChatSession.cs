using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Tokenization;
using ChatMessage = Tiny.Jarvis.Message.Models.Message;

namespace Tiny.Jarvis.Message.Prompt
{
    public class ChatSession(TinyJarvisModel model, ITokenizer tokenizer, (string userPromptName, string botPromptName)? promptNames)
    {
        private readonly List<ChatMessage> _history = new();

        private bool _running = true;  // optional if you want external control

        public void Run()
        {
            Console.WriteLine("Chat started. Type '/end' or '/exit' to stop.\n");

            while (_running)
            {
                Console.Write("You: ");
                var inputMessage = ChatInput.GetUserInput(promptNames?.userPromptName);

                // Termination check
                if (string.IsNullOrEmpty(inputMessage.Content) ||
                    inputMessage.Content.Equals("/end", StringComparison.OrdinalIgnoreCase) ||
                    inputMessage.Content.Equals("/exit", StringComparison.OrdinalIgnoreCase))
                {
                    _running = ChatOutput.ShouldEnd(_history);
                }

                _history.Add(inputMessage);

                Console.WriteLine($"User Requests: {inputMessage.Content}");
                Console.WriteLine($"Prompt: {inputMessage.ToString()}");
                Console.WriteLine(Environment.NewLine);

                // Build prompt with history
                var prompt = string.Join(Environment.NewLine, _history.Select(x => x.ToString())) + Environment.NewLine + (promptNames?.botPromptName ?? "assistant: ");

                // Get encoded sequence with Bos at the beginning
                var tokens = tokenizer.Encode(prompt);

                Console.WriteLine($"BOS token: {tokenizer.BOS}");
                Console.WriteLine($"tokens before Generate is called: {string.Join(",", tokens)}");

                var responseTokens = model.Generate(tokens, maxNewTokens: 100, temperature: 0.8, topK: 50, topP: 0.95);

                Console.WriteLine($"Generated response tokens: {string.Join(",", responseTokens)}");
                var response = tokenizer.Decode(responseTokens);

                Console.WriteLine($"Generated response: {response}");
                // Clean and display
                response = CleanResponse(response);
                
                ChatOutput.Reply(response, _history, promptNames?.botPromptName);
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
