namespace Tiny.Jarvis.Extensions
{
    internal static class ObjectExtensions
    {
        public static List<TResult> SelectRow<TSource, TResult>(this TSource[][] source, Func<IEnumerable<TSource>, TResult> selector)
        {
            if (source.Length == 0)
                throw new ArgumentException("Source array is empty.", nameof(source));

            var modifiedResults = new List<TResult>();
            for (var i = 0; i < source.Length; i++)
                modifiedResults.Add(selector(source[i]));

            return modifiedResults;
        }

        public static List<TSource> GetRow<TSource>(this TSource[][] source, int row)
        {
            if (row < 0 || row >= source.Length)
                return [];

            var result = new List<TSource>();
            for (var dimension = 0; dimension < source[row].Length; dimension++) // dimension = column
                result.Add(source[row][dimension]);

            return result;
        }
    }
}
