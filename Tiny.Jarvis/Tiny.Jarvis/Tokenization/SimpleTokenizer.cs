using Tiny.Jarvis.Extensions;

namespace Tiny.Jarvis.Tokenization;

public class SimpleTokenizer: ITokenizer
{
    private readonly Dictionary<int, string> _tokenToIdentifier;
    private readonly Dictionary<string, int> _identifierToToken;
    private readonly int _unknownTokenIdentifier;
    private const string _unknownToken = "[UNK]";
    private const string _bosToken = "[BOS]";
    private const string _eosToken = "[EOS]";

    public int BOS { get; } // Beginning of Sequence token ID
    public int EOS { get; } // End of Sequence token ID
    public int VocabSize { get; } // total number of unique tokens

    public SimpleTokenizer(IEnumerable<string> docs, int? vocabularySize = null)
    {

        var tokensToIdentifier = docs.Select((doc, index) => new KeyValuePair<int, string>(index, doc)).ToDictionary();

        // Build deterministic list of all tokens (special tokens first, then sorted)
        var allTokens = new List<string> { _unknownToken, _bosToken, _eosToken };

        allTokens.AddRange(tokensToIdentifier.Values.OrderBy(t => t));

        // Assign consecutive IDs (UNK=0, BOS=1, then rest)
        var identifierToToken = new Dictionary<string, int>();
        for (int i = 0; i < allTokens.Count; i++)
            identifierToToken[allTokens[i]] = i;

        _tokenToIdentifier = allTokens.Select((doc, index) => new KeyValuePair<int, string>(index, doc)).ToDictionary();
        _identifierToToken = _tokenToIdentifier.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

        BOS = _identifierToToken[_bosToken];
        EOS = _identifierToToken[_eosToken];
        VocabSize = vocabularySize == null ? _identifierToToken.Count : vocabularySize.Value;
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
            .Select(id => _tokenToIdentifier.GetValueOrDefault(id, _unknownToken))
            .ToList();

        return string.Join(" ", tokens);
    }
}