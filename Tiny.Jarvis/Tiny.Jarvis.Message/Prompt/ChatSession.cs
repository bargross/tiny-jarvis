using Tiny.Jarvis.Genetic;
using Tiny.Jarvis.Message.Models;
using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Tokenization;

namespace Tiny.Jarvis.Message.Prompt
{
    public class ChatSession(TinyJarvisModel model, ITokenizer tokenizer, TinyJarvisInteractiveGeneticAlgorithm geneticAlgorithm)
    {
        private readonly List<ConversationExchange> _history = new();

        private bool _running = true;  // optional if you want external control

        public void Run()
        {
            Console.WriteLine("Chat started. Type 'end' or 'exit' to stop, 'chg usr-prt' to change user prompt & 'chg bot-prt' to change the bot prompt \n");
            string userPrompt = "user";
            string botPrompt = "assistant";

            while (_running)
            {
                var conversationExchange = new ConversationExchange
                {
                    UserPrompt = ChatInput.GetUserInput(userPrompt),
                };

                // Termination check
                var userPromptContentAsLowerCase = conversationExchange?.UserPrompt?.Content?.ToLower();
                if (!string.IsNullOrEmpty(userPromptContentAsLowerCase))
                {
                
                   if (userPromptContentAsLowerCase.Equals("end", StringComparison.OrdinalIgnoreCase) || userPromptContentAsLowerCase.Equals("exit", StringComparison.OrdinalIgnoreCase))
                    {
                        _running = !ChatOutput.ShouldEnd(_history, botPrompt);
                    }

                    if (userPromptContentAsLowerCase.Equals("chg usr-prt", StringComparison.OrdinalIgnoreCase))
                    {
                        userPrompt = Console.ReadLine() ?? "user"; // with default
                    }

                    if (userPromptContentAsLowerCase.Equals("chg bot-prt", StringComparison.OrdinalIgnoreCase))
                    {
                        botPrompt = Console.ReadLine() ?? "assistant"; // with default
                    }

                    if (!_running) continue;  
                }


                _history.Add(conversationExchange);

                Console.WriteLine(conversationExchange.ToString());
                Console.WriteLine(Environment.NewLine);

                // Build prompt with history
                var prompt = string.Join(Environment.NewLine, _history.Select(x => x.ToString()));
                Console.WriteLine(prompt);

                // Get encoded sequence with Bos at the beginning
                var tokens = tokenizer.Encode(prompt);

                //Console.WriteLine($"BOS token: {tokenizer.BOS}"); // debug
                //Console.WriteLine($"tokens before Generate is called: {string.Join(",", tokens)}"); // debug

                // start the GA and run
                //var bestChromosome = geneticAlgorithm.Run();

                // Decode the best parameters
                var bestTopK = 20;
                var bestTemperature = 0.6;
                var bestTopP = 0.8;

                var responseTokens = model.Generate(tokens, maxNewTokens: 100, temperature: bestTemperature, topK: bestTopK, topP: bestTopP);

                //Console.WriteLine($"Generated response tokens: {string.Join(",", responseTokens)}"); // debug
                var response = tokenizer.Decode(responseTokens);

                //Console.WriteLine($"Generated response: {response}"); // debug
                // Clean and display
                response = CleanResponse(response);
                
                ChatOutput.Reply(response, _history, botPrompt);
            }
        }

        private string CleanResponse(string response)
        {
            // Stop if model starts generating a new user/assistant tag
            var idx = response.IndexOf("user:", StringComparison.OrdinalIgnoreCase);
            if (idx >= 0) response = response.Substring(0, idx);

            idx = response.IndexOf("assistant:", StringComparison.OrdinalIgnoreCase);
            if (idx >= 0) response = response.Substring(0, idx);

            return response.Trim();
        }
    }
}
