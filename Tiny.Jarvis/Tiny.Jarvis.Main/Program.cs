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

    if (format == "json")
    {
        var trainingDocsName = Path.GetFileNameWithoutExtension(filePath);
        if (filePath.Contains("passenger-register-titanic-dataset"))
        {
            var docs = Document.LoadFromJson<TitanicPassenger>(filePath, random);

            // Train (or load) the model
            var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(docs.Select(doc => doc.ToString()).ToList(), tokenizerStrategy);

            model = _model;
            tokenizer = _tokenizer;
        }

        if (filePath.Contains("bitext-travel-llm-chatbot-training-dataset"))
        {
            var docs = Document.LoadFromJson<BaggageQueryIntent>(filePath, random);

            // Train (or load) the model
            var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(docs.Select(doc => doc.ToString()).ToList(), tokenizerStrategy);

            model = _model;
            tokenizer = _tokenizer;
        }
    } 
    else
    {
        var docs = Document.LoadDocs(filePath, random);

        // Train (or load) the model
        var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(docs.Select(doc => doc.ToString()).ToList(), tokenizerStrategy);

        model = _model;
        tokenizer = _tokenizer;
    }

    // Now use the same model for chat
    Console.WriteLine("Training complete. Starting chat...");
    var chat = new ChatSession(model, tokenizer);

    chat.Run();
}

string SelectTrainingFile(string pathToDir)
{
    var filesAvailable = new DirectoryInfo(pathToDir)
    .GetFiles("*", new EnumerationOptions
    {
        RecurseSubdirectories = false,   // since all files are in same dir
        AttributesToSkip = FileAttributes.None,  // don't skip hidden/system
        IgnoreInaccessible = false,      // throw if can't read (to see error)
        MatchType = MatchType.Simple,    // disable complex pattern matching
        ReturnSpecialDirectories = false
    })
    .Select(fp => fp.FullName)
    .ToArray();

    Console.WriteLine($"Select Among Files Available inputs >> [0 -> {filesAvailable.Length - 1}]:");
    for (var fileIndex = 0; fileIndex < filesAvailable.Length; fileIndex++)
    {
        Console.WriteLine($"{fileIndex}. {filesAvailable[fileIndex].Split('\\').Last()}");
    }

    Console.WriteLine(Environment.NewLine);
    var userInput = Console.ReadLine();
    //var resultParsed = int.TryParse(userInput, out var index);

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

        return chosenFile;
    }

    return SelectTrainingFile(pathToDir);
}