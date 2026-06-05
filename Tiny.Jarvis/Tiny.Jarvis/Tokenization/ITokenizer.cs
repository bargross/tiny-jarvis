namespace Tiny.Jarvis.Tokenization
{
    internal interface ITokenizer
    {
        int VocabSize { get; }

        IReadOnlyList<int> Encode(string text);
        string Decode(IReadOnlyList<int> identifiers);
    }
}
