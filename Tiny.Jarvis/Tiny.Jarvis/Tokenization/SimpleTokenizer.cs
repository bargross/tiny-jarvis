using Tiny.Jarvis.Extensions;

namespace Tiny.Jarvis.Tokenization;

public class SimpleTokenizer: ITokenizer
{
    private readonly Dictionary<int, string> _tokenToIdentifier;
    private readonly Dictionary<string, int> _identifierToToken;
    private readonly int _unknownTokenIdentifier;
    private readonly List<string> _allWords;
    private const string UnknownToken = "[UNK]";

    public int Bos { get; } // Beginning of Sequence token ID
    public int VocabSize { get; } // total number of unique tokens

    public SimpleTokenizer(List<string> docs, int unknownTokenIdentifier = 0, int? vocabularySize = null)
    {
        _unknownTokenIdentifier = unknownTokenIdentifier;
        _allWords = [.. docs.SelectMany(line => line.DetectDelimeterAndSplit()).Distinct().OrderBy(w => w).ToList()];
        _tokenToIdentifier = _allWords.Select((doc, index) => new KeyValuePair<int, string>(index, doc)).ToDictionary();
        _identifierToToken = _tokenToIdentifier.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        _tokenToIdentifier[unknownTokenIdentifier] = UnknownToken;

        Bos = _allWords.Count;
        VocabSize = vocabularySize == null ? _allWords.Count + 1 : vocabularySize.Value;
    }

    public IReadOnlyList<int> Encode(string text)
    {
        return text
            .DetectDelimeterAndSplit()
            .Select(token => _identifierToToken.GetValueOrDefault(token, _unknownTokenIdentifier))
            .ToList();
    }

    public string Decode(IReadOnlyList<int> identifiers)
    {
        var tokens = identifiers
            .Select(id => _tokenToIdentifier.GetValueOrDefault(id, UnknownToken))
            .ToList();

        return string.Join(" ", tokens);
    }
}