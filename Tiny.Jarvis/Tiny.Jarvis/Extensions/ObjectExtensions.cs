namespace Tiny.Jarvis.Extensions
{
    internal static class ObjectExtensions
    {
        public static List<TResult> SelectRow<TSource, TResult>(this TSource[][] source, Func<IEnumerable<TSource>, TResult> selector)
        {
            if (source.Length == 0)
                throw new ArgumentException("Source array is empty.", nameof(source));

            var result = new List<TResult>();
            for (int i = 0; i < source.Length; i++)
            {
                result.Add(selector(source[i]));
            }

            return result;
        }

        public static List<TSource> GetRow<TSource>(this TSource[][] source, int row)
        {
            if (row < 0 || row >= source.Length)
                return [];

            var result = new List<TSource>();
            for (int j = 0; j < source[row].Length; j++)
                result.Add(source[row][j]);

            return result;
        }
    }
}
