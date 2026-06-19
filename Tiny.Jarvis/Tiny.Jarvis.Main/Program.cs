using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Genetic;
using Tiny.Jarvis.Genetic.Crossover;
using Tiny.Jarvis.Genetic.Models;
using Tiny.Jarvis.Message.Prompt;
using Tiny.Jarvis.Training.Enums;
using Tiny.Jarvis.Training.Models;
using Tiny.Jarvis.Training.Trainers;
using Tiny.Jarvis.Util;

// begins the chat and continues until the user exits
BeginChat();

void BeginChat()
{
    // docs, variables etc...
    var assetsPath = Path.Combine(FindSolutionRoot(), "Tiny.Jarvis", "Assets");
    var dirPathRef = Path.GetFullPath(assetsPath);

    var savedRunsDir = Path.Combine(FindSolutionRoot(), "Tiny.Jarvis", "SavedRuns");

    var pathToModelSavedRuns = Path.GetFullPath(Path.Combine(savedRunsDir, "models"));
    var pathToTokenizerSavedRuns = Path.GetFullPath(Path.Combine(savedRunsDir, "tokenizers"));
    var pathToOptimizerSavedRuns = Path.GetFullPath(Path.Combine(savedRunsDir, "optimizers"));
    var fileStaveTimestamp = DateTime.UtcNow;
    var random = new Random(42);

    EnsureWritePermissionsOnWindows(pathToModelSavedRuns);
    EnsureWritePermissionsOnWindows(pathToTokenizerSavedRuns);
    EnsureWritePermissionsOnWindows(pathToOptimizerSavedRuns);

    var (isNewModel, specialName) = GetModelOptionalSpecialName();
    
    var hyperParams = new TinyJarvisHyperParameters
    {
        TokenizerStrategy = TokenizerStrategy.Chars,
        OptimizerStrategy = OptimizerStrategy.Adam,
        EmbeddingSize = 64,
        MaxSequenceLength = 42,
        LearningRate = 0.0003,
        NumOfMerges = 200,
        VocabularySize = 600,
        MaxNumberOfSteps = 10000,
        MaxGradNorm = 1.0,
        SaveModelFile = GetUniqueFileNameWithTimestamp(pathToModelSavedRuns, "model-run", "bin", specialName, fileStaveTimestamp),
        SaveOptimizerFile = GetUniqueFileNameWithTimestamp(pathToOptimizerSavedRuns, "optimizer-run", "bin", specialName, fileStaveTimestamp),
        SaveTokenizerFile = GetUniqueFileNameWithTimestamp(pathToOptimizerSavedRuns, "tokenizer-run", "json", specialName, fileStaveTimestamp)
    };

    if (!isNewModel)
    {
        var modelSaveFile = LoadFromPreviousRun(pathToModelSavedRuns, "model");
        var tokenizerSaveFile = LoadFromPreviousRun(pathToTokenizerSavedRuns, "tokenizer");
        var optimizerSaveFile = LoadFromPreviousRun(pathToOptimizerSavedRuns, "optimizer");

        hyperParams.LoadModelFile = modelSaveFile;
        hyperParams.LoadTokenizerFile = tokenizerSaveFile;
        hyperParams.LoadOptimizerFile = optimizerSaveFile;

        hyperParams.LoadedFromPreviousRun = true;
    }

    //var vocabularySize = 64; // only for tokenizers other than Character

    // Get the training data
    var filePaths = SelectFiles(dirPathRef);

    Console.WriteLine("Chosen training files:");
    Console.WriteLine("-------------------------");
    foreach (var filePath in filePaths)
        Console.WriteLine(filePath);

    Console.WriteLine(Environment.NewLine);
    var trainingDocuments = GetDocs(filePaths, random);


    // Train (or load) the model
    var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(trainingDocuments, hyperParams);

    // Now use the same model for chat
    Console.WriteLine("Training complete. Starting chat...");
    Console.WriteLine(Environment.NewLine);

    var chat = new ChatSession<double>(_model, _tokenizer, CreateGeneticAlgorithm<double>());

    chat.Run();
}

void EnsureWritePermissionsOnWindows(string directoryPath)
{
    if (!OperatingSystem.IsWindows())
        return; // only applies to Windows

    var dirInfo = new DirectoryInfo(directoryPath);
    if (!dirInfo.Exists)
    {
        Directory.CreateDirectory(directoryPath);
        dirInfo = new DirectoryInfo(directoryPath);
    }

    try
    {
        var security = dirInfo.GetAccessControl();
        var currentUser = WindowsIdentity.GetCurrent().User;
        security.AddAccessRule(new FileSystemAccessRule(
            currentUser,
            FileSystemRights.FullControl,
            InheritanceFlags.ContainerInherit | InheritanceFlags.ObjectInherit,
            PropagationFlags.None,
            AccessControlType.Allow));
        dirInfo.SetAccessControl(security);
    }
    catch (UnauthorizedAccessException)
    {
        // You may not have permission to modify ACLs; fall back to running as admin.
        Console.WriteLine("Unable to modify folder permissions; run as administrator if needed.");
    }
}

(bool isNewModel, string specialName) GetModelOptionalSpecialName()
{
    Console.WriteLine("New model?");
    var response = Console.ReadLine();
    Console.WriteLine(Environment.NewLine);

    var specialName = "";
    if (response == "y")
    {
        Console.Write("Special model name: ");
        specialName = Console.ReadLine();
        Console.WriteLine(Environment.NewLine);

        if (string.IsNullOrWhiteSpace(specialName))
            return GetModelOptionalSpecialName();   
    }

    return (response == "y" ,specialName);
}

string? LoadFromPreviousRun(string path, string item)
{
    Console.WriteLine($"Load {item} which run? (select index)");

    var files = SelectFiles(path, false);

    if (files.Count > 1) throw new ArgumentException("can only load 1 model at a time.");

    return files.FirstOrDefault();
}

string GetUniqueFileNameWithTimestamp(string directory, string baseName, string extension, string specialName, DateTime timestamp)
{
    var safeBase = string.Concat(baseName.Split(Path.GetInvalidFileNameChars()));

    if (string.IsNullOrWhiteSpace(safeBase)) safeBase = "file";

    var timestampAndSpecialName = timestamp.ToString("yyyy-MM-dd_HH-mm-ss");
    if (!string.IsNullOrWhiteSpace(specialName)) timestampAndSpecialName = $"{timestampAndSpecialName}_{specialName}";

    return Path.Combine(directory, $"{safeBase}_{timestampAndSpecialName}.{extension}");
}

TinyJarvisInteractiveGeneticAlgorithm<double> CreateGeneticAlgorithm<TPopulation>(int populationSize = 30, int chromosomeLength = 3, int maxGenerations = 100)
{
    var crossovers = new Dictionary<CrossoverType, ICrossover<TPopulation>>
    {
        { CrossoverType.Average, new AverageCrossover<TPopulation>() },
        { CrossoverType.Internal, new InternalCrossover<TPopulation>() },
        { CrossoverType.Coexistence, new CoexistenceCrossover<TPopulation>() }
    };

    // Instantiate the GA engine
    return new TinyJarvisInteractiveGeneticAlgorithm<double>(crossovers)
    {
        CrossoverType = CrossoverType.Coexistence,   // can be changed
        CrossoverProbability = 0.8,
        MutationProbability = 0.1,
        MinGeneValue = 1,
        MaxGeneValue = 100,
        EliteCount = 2,
        PopulationSize = populationSize,
        ChromosomeLength = chromosomeLength,
        MaxGenerations = maxGenerations,

        // Fitness function: decode genes and compute a performance metric
        FitnessFunction = (chromosome) =>
        {
            var topK = chromosome[0];
            var temperature = chromosome[1] / 100.0;
            var topP = chromosome[2] / 100.0;

            // Simulate model evaluation – replace with real evaluation
            // Higher fitness is better.
            var coherence = 0.5 * Math.Log(topK + 1) + 0.3 * temperature + 0.2 * topP;

            // Add some noise to avoid trivial solution
            var fitness = coherence + new Random().NextDouble() * 0.1;

            return fitness;
        },

        // Termination condition: stop after 100 generations or when best fitness > 0.95
        TerminationCondition = (gen, bestFitness, _) => gen >= maxGenerations || bestFitness >= 0.95
    };
}

List<string> GetDocs(List<string> filePaths, Random random)
{
    var filePathToFormat = filePaths.ToDictionary(path => path, path => Document.GetFormat(path));
    var docs = new List<string>();
    string[] acceptableJsonFormats = ["json", "jsonl"];

    foreach (var kvp in filePathToFormat)
        if (acceptableJsonFormats.Contains(kvp.Value)) {
            
            if (kvp.Key.Contains("passenger-register-titanic-dataset"))
                docs.AddRange(Document.LoadFromJson<TitanicPassengerData>(kvp.Key).Select(x => x.ToString()));

            if (kvp.Key.Contains("bitext-travel-llm-chatbot-training-dataset.jsonl")) 
                docs.AddRange(Document.LoadFromJson<BaggageQueryIntentData>(kvp.Key).Select(x => x.ToString()));

            if (kvp.Key.Contains("pickle-dataset-all-training"))
                docs.AddRange(Document.LoadFromJson<PickleDocument>(kvp.Key).SelectMany(doc => doc.Sentences).SelectMany(x => x));

            if (kvp.Key.Contains("helpsteer-training"))
                docs.AddRange(Document.LoadFromJson<PromptResponseData>(kvp.Key).Select(doc => doc.ToString()));

            if (kvp.Key.Contains("chatalpaca-10k"))
                docs.AddRange(ConvertChatAlpacaToTinyJarvisFormat(Document.LoadFromJson<ChatAlpacaConversation>(kvp.Key)));
        }
        else docs.AddRange(Document.LoadFromFile(kvp.Key, random));

    return docs;
}

/// <summary>
/// Converts a list of deserialized ChatAlpaca conversations into the format expected by TinyJarvis.
/// Each conversation becomes a single string with alternating "user: ... assistant: ..." turns.
/// </summary>
/// <param name="conversations">List of ChatAlpaca conversation objects.</param>
/// <returns>List of formatted strings, one per conversation.</returns>
List<string> ConvertChatAlpacaToTinyJarvisFormat(List<ChatAlpacaConversation> conversations)
{
    var result = new List<string>(conversations.Count);

    foreach (var conv in conversations)
    {
        var turns = new StringBuilder();

        foreach (var turn in conv.Conversations)
        {
            // Map role: "human" -> "user", "gpt" -> "assistant"
            string? role = turn.From?.ToLowerInvariant() switch
            {
                "human" => "user",
                "gpt" => "assistant",
                _ => null
            };

            if (role == null) continue; // skip unknown roles

            if (turns.Length > 0)
                turns.Append(' ');
            turns.Append($"{role}: {turn.Value}");
        }

        if (turns.Length > 0)
            result.Add(turns.ToString());
    }

    return result;
}

string FindSolutionRoot()
{
    var directory = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
    while (directory != null)
    {
        if (File.Exists(Path.Combine(directory.FullName, "tiny-jarvis.sln")) ||
            Directory.Exists(Path.Combine(directory.FullName, ".git")))
            return directory.FullName;

        directory = directory.Parent;
    }

    throw new DirectoryNotFoundException("Solution root not found.");
}

List<string> SelectFiles(string pathToDir, bool flexibleFetch = true)
{
    var files = new List<string>();
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
        Console.WriteLine($"{fileIndex}. {filesAvailable[fileIndex].Split('\\').Last()}");
    
    Console.WriteLine(Environment.NewLine);

    if (!flexibleFetch)
    {
        AddFile(pathToDir, files, filesAvailable);

        return files;
    }

    var fetch = true;
    while(fetch)
    {
        AddFile(pathToDir, files, filesAvailable);

        Console.WriteLine(Environment.NewLine);
        Console.Write($"Fetch Another (y/n): ");

        var userResponseInput = Console.ReadLine()?.ToLower();
        fetch = userResponseInput == "y" || userResponseInput == "yes";

        if (fetch)
            for (var fileIndex = 0; fileIndex < filesAvailable.Length; fileIndex++)
                Console.WriteLine($"{fileIndex}. {filesAvailable[fileIndex].Split('\\').Last()}");
    }

    Console.WriteLine(Environment.NewLine);

    return files;
}

void AddFile(string pathToDir, List<string> files, string[] filesAvailable)
{
    Console.WriteLine(Environment.NewLine);
    Console.Write("Enter File Number: ");
    var userInput = Console.ReadLine();

    if (!int.TryParse(userInput, out var index))
    {
        Console.WriteLine("Invalid input, it must be a number corresponding to the file index.");

        AddFile(pathToDir, files, filesAvailable);
    }

    if (index >= 0 && index < filesAvailable.Length)
        files.Add(filesAvailable[index]);

    return;
}