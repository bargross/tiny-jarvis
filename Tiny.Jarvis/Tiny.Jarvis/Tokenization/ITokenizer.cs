namespace Tiny.Jarvis.Tokenization
{
    public interface ITokenizer<TIdentifier>
    {
        Dictionary<TIdentifier, int> IdentifierToToken { get; }
        List<(TIdentifier Left, TIdentifier Right)>? MergeRules { get; }
        Dictionary<TIdentifier, double>? TokenLogProbabilities { get; }

        int VocabSize { get; }
        int UnknownTokenId { get; }
        int BOS { get; }
        int EOS { get; }

        IReadOnlyList<int> Encode(string text);
        string Decode(IReadOnlyList<int> identifiers);
    }
}
