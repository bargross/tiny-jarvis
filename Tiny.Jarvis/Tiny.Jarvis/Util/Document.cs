using System.Text.RegularExpressions;
using Tiny.Jarvis.Extensions;

namespace Tiny.Jarvis.Util
{
    public static class Document
    {
        /// <summary>
        /// Loads documents from a text file, one per line, shuffled.
        /// </summary>
        public static List<string> LoadFromFile(string path, Random random)
        {
            var lines = null as IEnumerable<string>;

            var fileFormat = GetFormat(path);
            if (!IsValidFormat(fileFormat))
                throw new ArgumentException($"Unsupported file format: {fileFormat}");
            
            if (fileFormat == "csv")
            {
                lines = File.ReadAllLines(path) as IEnumerable<string>;

                var headers = lines.First();
                lines = lines.Skip(1)
                    .Select(l => Regex.Replace(l.Trim(), @"[\""]", "", RegexOptions.None))
                    .SelectMany(line => line.Split(",")
                        .Select((word, i) => $"{headers[i]}: {word}"));
            }
            else
            {
                lines = File.ReadAllLines(path)
                    .Select(l => l.Trim());
            }

            return lines.Where(l => !string.IsNullOrEmpty(l)).ToList();
        }

        public static List<TValue> LoadFromJson<TValue>(string path, Random random)
        {
            var fileFormat = GetFormat(path);
            if (!IsValidFormat(fileFormat))
                throw new ArgumentException($"Unsupported file format: {fileFormat}, Supported formats are: csv, txt, json");

            var jsonString = File.ReadAllText(path);
            var jsonData = System.Text.Json.JsonSerializer.Deserialize<List<TValue>>(jsonString);

            return jsonData ?? new List<TValue>();
        }

        public static string GetFormat(string path) 
        {
            var fileName = GetFileName(path);
            var dotExtensionOccurances = fileName.CountOccurrences(".");
            if (!fileName.Contains('.') || dotExtensionOccurances > 1)
                throw new ArgumentException("Invalid path format. Expected a file path with an extension, e.g., 'data.csv' or 'data.json'.");

            var pathParts = fileName.Split('.');

            return pathParts.Last();
        }

        private static bool IsValidFormat(string format) =>  format == "csv" || format == "txt" || format == "json";

        private static string GetFileName(string path) => path.Split("\\").Last();
    }
}
