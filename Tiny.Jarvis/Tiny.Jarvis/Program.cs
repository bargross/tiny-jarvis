using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Util;

namespace Tiny.Jarvis;

public static class Program
{
    public static void Main(string[] args)
    {
        string chapter = args.Length > 0 ? args[0].ToLowerInvariant() : "";
        var docRef = "./Assets/train.csv";

        var result = Document.LoadDocs(docRef, new Random(42));
        var tokenType = TokenType.Word;

        switch (chapter)
        {
            case "full":
                ModelTrainer.Run(docRef, tokenType);
                break;

            default:
                Console.WriteLine("Tiny.Jarvis project is ready.");
                Console.WriteLine($"Dataset exists: {File.Exists(docRef)}");

                if (File.Exists(docRef))
                {
                    int lineCount = File.ReadAllLines(docRef).Length;
                    Console.WriteLine($"Dataset lines: {lineCount}");
                }

                break;
        }
    }
}