namespace Tiny.Jarvis.Tokenization
{
    public interface ITokenizer
    {
        int VocabSize { get; }
        int Bos { get; }

        IReadOnlyList<int> Encode(string text);
        string Decode(IReadOnlyList<int> identifiers);
    }
}
