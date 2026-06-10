namespace Tiny.Jarvis.Tokenization;

public class CharacterTokenizer : ITokenizer
{
    private readonly Dictionary<int, string> _tokenToIdentifier; // ID -> character string
    private readonly Dictionary<string, int> _identifierToToken; // character string -> ID
    private readonly int _unknownTokenIdentifier;
    private const string _unknownToken = "[UNK]"; // not single token but works for now
    private const string _bosToken = "[BOS]";  // not single token but works for now
    private const string _eosToken = "[EOS]";  // not single token but works for now

    public int BOS { get; private set; }
    public int EOS { get; private set; }
    public int VocabSize { get; private set; }

    // Constructor using an explicit alphabet string (no training on docs)
    public CharacterTokenizer(string allowedChars)
    {
        // Build the set of character tokens (each char becomes a token string)
        var uniqueChars = allowedChars.Distinct().OrderBy(c => c).Select(c => c.ToString()).ToList();

        // Special tokens (use multi‑character strings for readability)
        // Note: These strings will be treated as single tokens.
        // They are not single characters, but for a character tokenizer you can keep them as is.
        // For single char, replace with e.g., "\u0002", "\u0003", "\uFFFD".
        var allTokens = new List<string> { _unknownToken, _bosToken, _eosToken};
        allTokens.AddRange(uniqueChars);

        // Build string→ID map
        var identifierToToken = allTokens.Select((val, idx) => (val, idx)).ToDictionary(x => x.val, x => x.idx);

        // Build reverse ID→string map
        var tokenToIdentifier = identifierToToken.ToDictionary(x => x.Value, x => x.Key);

        _identifierToToken = identifierToToken;
        _tokenToIdentifier = tokenToIdentifier;

        _unknownTokenIdentifier = _identifierToToken[_unknownToken];
        BOS = _identifierToToken[_bosToken];
        EOS = _identifierToToken[_eosToken];

        VocabSize = _identifierToToken.Count;
    }

    // Encode: convert a string into a list of token IDs (one per character)
    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (char c in text)
        {
            var charStr = c.ToString();
            if (_identifierToToken.TryGetValue(charStr, out var id))
                ids.Add(id);
            
            else ids.Add(_unknownTokenIdentifier);
        }

        return ids;
    }

    // Decode: convert a list of token IDs back into a string
    public string Decode(IReadOnlyList<int> identifiers)
    {
        var chars = new List<char>();
        foreach (var id in identifiers)
        {
            var token = _tokenToIdentifier.GetValueOrDefault(id, _unknownToken);

            if (token == _bosToken || token == _eosToken || token == _unknownToken)
                continue; // skip special tokens when decoding for user display
            
            // If the token is a single character, use it directly.
            if (token.Length == 1)
                chars.Add(token[0]);

            // fallback (should not happen for normal characters)
            else chars.Add('?');
        }

        return new string(chars.ToArray());
    }
}