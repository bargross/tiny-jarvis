using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Message.Prompt;
using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Tokenization;
using Tiny.Jarvis.Training.Models.Training;
using Tiny.Jarvis.Training.Trainers;
using Tiny.Jarvis.Util;

// docs, variables etc...
var dirPathRef = "C:/Users/Leon/Documents/Projects/tiny-jarvis/Tiny.Jarvis/Assets";

// begins the chat and continues until the user exits
BeginChat();

void BeginChat()
{
    var filePath = SelectTrainingFile(dirPathRef);
    var format = Document.GetFormat(filePath);
    var list = new List<dynamic>();
    var random = new Random(42);
    var model = null as TinyJarvisModel;
    var tokenizer = null as ITokenizer;
    var tokenizerStrategy = TokenizerStrategy.WordPiece;

     // set this based on the average length of your documents (in tokens) - it controls the context window size for the model, so longer is generally better for performance but increases training time and memory usage
    var maxSequenceLength = 128;

    // TODO: it might be worth trying different values for different tokenizers to see if some converge faster than others (e.g. character-level tokenizers will likely require more steps than word-level ones)
    var maxNumberOfSteps = 15000; // increase this for better performance - the optimal number depends on the size of your dataset and the complexity of the task
    if (format == "json")
    {
        var trainingDocsName = Path.GetFileNameWithoutExtension(filePath);
        if (filePath.Contains("passenger-register-titanic-dataset"))
        {
            var docs = Document.LoadFromJson<TitanicPassenger>(filePath, random);

            // Train (or load) the model
            var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(docs.Select(doc => doc.ToString()), tokenizerStrategy, maxSequenceLength, maxNumberOfSteps);

            model = _model;
            tokenizer = _tokenizer;
        }

        if (filePath.Contains("bitext-travel-llm-chatbot-training-dataset"))
        {
            var docs = Document.LoadFromJson<BaggageQueryIntent>(filePath, random);

            // Train (or load) the model
            var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(docs.Select(doc => doc.ToString()), tokenizerStrategy, maxSequenceLength, maxNumberOfSteps);

            model = _model;
            tokenizer = _tokenizer;
        }
    } 
    else
    {
        var docs = Document.LoadFromFile(filePath, random);

        // Train (or load) the model
        var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(docs.Select(doc => doc.ToString()), tokenizerStrategy, maxSequenceLength, maxNumberOfSteps);

        model = _model;
        tokenizer = _tokenizer;
    }

    // Now use the same model for chat
    Console.WriteLine("Training complete. Starting chat...");
    Console.WriteLine(Environment.NewLine);
    var chat = new ChatSession(model, tokenizer);

    chat.Run();
}

string SelectTrainingFile(string pathToDir)
{
    var filesAvailable = new DirectoryInfo(pathToDir)
    .GetFiles("*", new EnumerationOptions
    {
        RecurseSubdirectories = false,
        AttributesToSkip = FileAttributes.None,
        IgnoreInaccessible = false,
        MatchType = MatchType.Simple,
        ReturnSpecialDirectories = false
    })
    .Select(fp => fp.FullName)
    .ToArray();

    Console.WriteLine($"Select Among Files Available inputs >> [0 -> {filesAvailable.Length - 1}]:");
    Console.WriteLine("------------------------------------------------------------------------------");
    Console.WriteLine(Environment.NewLine);
    for (var fileIndex = 0; fileIndex < filesAvailable.Length; fileIndex++)
    {
        Console.WriteLine($"{fileIndex}. {filesAvailable[fileIndex].Split('\\').Last()}");
    }

    Console.WriteLine(Environment.NewLine);
    Console.Write("Enter File Number: ");
    var userInput = Console.ReadLine();

    if (!int.TryParse(userInput, out var index))
    {
        Console.WriteLine("Invalid input, it must be a number corresponding to the file index.");
        return SelectTrainingFile(pathToDir);
    }

    var filePath = null as string;
    if (index >= 0 && index < filesAvailable.Length)
    {
        var chosenFile = filesAvailable[index];

        Console.WriteLine(Environment.NewLine);
        Console.WriteLine($"File chosen: {chosenFile}");
        Console.WriteLine(Environment.NewLine);

        return chosenFile;
    }

    return SelectTrainingFile(pathToDir);
}