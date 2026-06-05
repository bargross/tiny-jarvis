using System.Text.RegularExpressions;

namespace Tiny.Jarvis.Util
{
    internal static class Document
    {
        /// <summary>
        /// Loads documents from a text file, one per line, shuffled.
        /// </summary>
        public static List<string> LoadDocs(string path, Random random)
        {
            var lines = File.ReadAllLines(path) as IEnumerable<string>;

            if (path.Substring(path.Length - 4) == ".csv")
            {
                var headers = lines.First();
                lines = lines.Skip(1)
                    .Select(l => Regex.Replace(l.Trim(), @"[\""]", "", RegexOptions.None))
                    .SelectMany(line => line.Split(",")
                        .Select((word, i) => $"{headers[i]}: {word}"));
            }
            else
            {
                lines = lines.Select(l => l.Trim());
            }

            return lines.Where(l => !string.IsNullOrEmpty(l)).ToList();
        }
    }
}
