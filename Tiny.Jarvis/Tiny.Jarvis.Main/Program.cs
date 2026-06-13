using Tiny.Jarvis.Enums;
using Tiny.Jarvis.Genetic;
using Tiny.Jarvis.Genetic.Crossover;
using Tiny.Jarvis.Genetic.Models;
using Tiny.Jarvis.Message.Prompt;
using Tiny.Jarvis.Training.Models;
using Tiny.Jarvis.Training.Trainers;
using Tiny.Jarvis.Util;

// docs, variables etc...
var assetsPath = Path.Combine(FindSolutionRoot(), "Tiny.Jarvis", "Assets");
var dirPathRef = Path.GetFullPath(assetsPath);

// begins the chat and continues until the user exits
BeginChat();

void BeginChat()
{
    var random = new Random(42);
    var tokenizerStrategy = TokenizerStrategy.WordPiece;

     // set this based on the average length of your documents (in tokens) - it controls the context window size for the model, so longer is generally better for performance but increases training time and memory usage
    var maxSequenceLength = 34;

    // TODO: it might be worth trying different values for different tokenizers to see if some converge faster than others (e.g. character-level tokenizers will likely require more steps than word-level ones)
    var maxNumberOfSteps = 10000; // increase this for better performance - the optimal number depends on the size of your dataset and the complexity of the task
    int? vocabularySize = tokenizerStrategy == TokenizerStrategy.WordPiece || tokenizerStrategy == TokenizerStrategy.Unigram ? 250 : null;
    

    var geneticAlgorithm = CreateGeneticAlgorithm();

    //var vocabularySize = 64; // only for tokenizers other than Character

    // Get the training data
    var filePaths = SelectTrainingFile(dirPathRef, new List<string>());

    Console.WriteLine("Chosen training files:");
    Console.WriteLine("-------------------------");
    foreach (var filePath in filePaths)
        Console.WriteLine(filePath);

    Console.WriteLine(Environment.NewLine);
    var docs = GetDocs(filePaths, random);


    // Train (or load) the model
    var (_model, _tokenizer) = TinyJarvisModelTrainer.Train(docs, tokenizerStrategy, maxSequenceLength, maxNumberOfSteps, vocabularySize);

    // Now use the same model for chat
    Console.WriteLine("Training complete. Starting chat...");
    Console.WriteLine(Environment.NewLine);
    var chat = new ChatSession(_model, _tokenizer, geneticAlgorithm);

    chat.Run();
}

TinyJarvisInteractiveGeneticAlgorithm CreateGeneticAlgorithm(int populationSize = 30, int chromosomeLength = 3, int maxGenerations = 100)
{
    var crossovers = new Dictionary<CrossoverType, ICrossover>
{
    { CrossoverType.Average, new AverageCrossover() },
    { CrossoverType.Internal, new InternalCrossover() },
    { CrossoverType.Coexistence, new CoexistenceCrossover() }
};

    // 2. Instantiate the GA engine
    return new TinyJarvisInteractiveGeneticAlgorithm(crossovers)
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
            {
                var documents = Document.LoadFromJson<BaggageQueryIntentData>(kvp.Key);

                docs.AddRange(documents.Select(x => x.ToString()));
            }

            if (kvp.Key.Contains("pickle-dataset-all-training"))
                docs.AddRange(Document.LoadFromJson<PickleDocument>(kvp.Key).SelectMany(doc => doc.Sentences).SelectMany(x => x));

            if (kvp.Key.Contains("helpsteer-training"))
            {
                var documents = Document.LoadFromJson<PromptResponseData>(kvp.Key);

                docs.AddRange(documents.Select(doc => doc.ToString()));
            }
        }
        else docs.AddRange(Document.LoadFromFile(kvp.Key, random));

    return docs;
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

List<string> SelectTrainingFile(string pathToDir, List<string> files)
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
        Console.WriteLine($"{fileIndex}. {filesAvailable[fileIndex].Split('\\').Last()}");

    AddFile(pathToDir, files, filesAvailable);

    var fetch = true;
    while(fetch)
    {

        Console.WriteLine(Environment.NewLine);
        Console.Write($"Fetch Another (y/n): ");
        var userResponseInput = Console.ReadLine()?.ToLower();

        fetch = userResponseInput == "y" || userResponseInput == "yes";

        if (fetch) AddFile(pathToDir, files, filesAvailable);

        else break;
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