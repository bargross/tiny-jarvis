using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Tokenization;
using ChatMessage = Tiny.Jarvis.Message.Models.Message;

namespace Tiny.Jarvis.Message.Prompt
{
    public class ChatSession
    {
        private readonly TinyJarvisModel _model;
        private readonly ITokenizer _tokenizer;
        private readonly List<ChatMessage> _history = new();
        private bool _running = true;  // optional if you want external control

        public ChatSession(TinyJarvisModel model, ITokenizer tokenizer)
        {
            _model = model;
            _tokenizer = tokenizer;
        }

        public void Run()
        {
            Console.WriteLine("Chat started. Type '/end' or '/exit' to stop.\n");

            while (_running)
            {
                Console.Write("You: ");
                var inputMessage = ChatInput.GetUserInput(_history);

                // Termination check
                if (string.IsNullOrEmpty(inputMessage.Content) ||
                    inputMessage.Content.Equals("/end", StringComparison.OrdinalIgnoreCase) ||
                    inputMessage.Content.Equals("/exit", StringComparison.OrdinalIgnoreCase))
                {
                    _running = ChatOutput.ShouldEnd(_history);
                }

                // Build prompt with history
                var prompt = string.Join("\n", _history.Select(x => x.ToString())) + "\nassistant:";

                // Generate response
                var inputIds = _tokenizer.Encode(prompt);
                var responseIds = _model.Generate(inputIds, maxNewTokens: 100, temperature: 0.8, topK: 50, topP: 0.95);
                var response = _tokenizer.Decode(responseIds);

                // Clean and display
                response = CleanResponse(response);
                
                ChatOutput.Reply(response, _history);
            }
        }

        private string CleanResponse(string response)
        {
            // Stop if model starts generating a new user/assistant tag
            int idx = response.IndexOf("\nuser:", StringComparison.OrdinalIgnoreCase);
            if (idx >= 0) response = response.Substring(0, idx);

            idx = response.IndexOf("\nassistant:", StringComparison.OrdinalIgnoreCase);
            if (idx >= 0) response = response.Substring(0, idx);

            return response.Trim();
        }
    }
}
