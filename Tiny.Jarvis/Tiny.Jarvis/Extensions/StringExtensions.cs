namespace Tiny.Jarvis.Extensions
{
    internal static class StringExtensions
    {
        private static readonly char[] CandidateDelimiters = new[] { ',', '\t', ';', '|', ' ' };

        public static char? DetectDelimiter(this string line, char[]? candidateDelimiters = null)
        {
            if (candidateDelimiters == null)
            {
                candidateDelimiters = CandidateDelimiters;
            }

            foreach (char delimiter in candidateDelimiters)
            {
                string[] parts = line.Split(delimiter);
                // If the array length is greater than 1, the delimiter was found.
                if (parts.Length > 1)
                {
                    return delimiter;
                }
            }

            return null;
        }

        public static string[] DetectDelimeterAndSplit(this string line)
        {
            var delimiter = line.DetectDelimiter();

            if (delimiter == null) return [line];

            return line.Split(delimiter.Value);
        }
    }
}
