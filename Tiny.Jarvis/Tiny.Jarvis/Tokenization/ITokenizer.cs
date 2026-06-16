namespace Tiny.Jarvis.Tokenization
{
    public interface ITokenizer
    {
        Dictionary<string, int> IdentifierToToken { get; }
        List<(string Left, string Right)>? MergeRules { get; }
        Dictionary<string, double>? TokenLogProbabilities { get; }

        int VocabSize { get; }
        int UnknownTokenId { get; }
        int BOS { get; }
        int EOS { get; }

        IReadOnlyList<int> Encode(string text);
        string Decode(IReadOnlyList<int> identifiers);
    }
}
