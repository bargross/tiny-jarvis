using Tiny.Jarvis.MLModels;
using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Serializers
{
    public static class ModelSerializer
    {
        /// <summary>
        /// Saves the model's weights and hyperparameters to a binary file.
        /// </summary>
        /// <param name="model">The TinyJarvisModel to save.</param>
        /// <param name="filePath">Path to the output file (e.g., "model.bin").</param>
        public static void Save(TinyJarvisModel model, TinyJarvisHyperParameters hParams)
        {
            using var writer = new BinaryWriter(File.Open(hParams.SaveModelFile, FileMode.Create));

            // Write hyperparameters
            writer.Write(hParams.EmbeddingSize);
            writer.Write(hParams.HeadCount);
            writer.Write(hParams.LayerCount);
            writer.Write(hParams.MaxSequenceLength);
            writer.Write(hParams.VocabularySize); // from tokenizer, but stored for consistency

            // Helper to write a jagged float array
            void WriteValueMatrix(Scalar[][] matrix)
            {
                var rows = matrix.Length;
                var cols = matrix[0].Length;

                writer.Write(rows);
                writer.Write(cols);

                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        writer.Write(matrix[i][j].Data); // write the double value
            }

            // Embeddings
            WriteValueMatrix(model.TokenEmbeddings);
            WriteValueMatrix(model.PositionEmbeddings);

            // Layers
            writer.Write(model.Layers.Count);
            foreach (var layer in model.Layers)
            {
                WriteValueMatrix(layer.Query);
                WriteValueMatrix(layer.Key);
                WriteValueMatrix(layer.Value);
                WriteValueMatrix(layer.Output);
                WriteValueMatrix(layer.FeedForwardOne);
                WriteValueMatrix(layer.FeedForwardTwo);
            }

            // Output head
            WriteValueMatrix(model.OutputHead);
        }

        /// <summary>
        /// Loads a model from a previously saved binary file.
        /// </summary>
        /// <param name="filePath">Path to the saved model file.</param>
        /// <param name="tokenizer">The tokenizer instance (must have matching vocab size).</param>
        /// <param name="random">Random number generator (used for re‑initialisation, though weights are overwritten).</param>
        /// <returns>A new TinyJarvisModel instance with the loaded weights.</returns>
        public static TinyJarvisModel Load<IVocabulary>(string filePath, int bos, int eos, Random random)
        {
            using var reader = new BinaryReader(File.OpenRead(filePath));

            var embeddingSize = reader.ReadInt32();
            var headCount = reader.ReadInt32();
            var layerCount = reader.ReadInt32();
            var maxSeqLen = reader.ReadInt32();
            var savedVocabSize = reader.ReadInt32(); // optional

            // Helper to read a Value[][] matrix
            Scalar[][] ReadValueMatrix()
            {
                var rows = reader.ReadInt32();
                var cols = reader.ReadInt32();
                var matrix = new Scalar[rows][];
                for (int i = 0; i < rows; i++)
                {
                    matrix[i] = new Scalar[cols];
                    for (int j = 0; j < cols; j++)
                    {
                        double data = reader.ReadDouble();
                        matrix[i][j] = new Scalar(data); // create Value with the loaded data
                    }
                }
                return matrix;
            }

            // Overwrite embeddings (assign to private fields – you need public properties or internal setters)
            var tokenEmbeddings = ReadValueMatrix();
            var positionEmbeddings = ReadValueMatrix();

            // Layers
            int loadedLayerCount = reader.ReadInt32();
            if (loadedLayerCount != layerCount)
                throw new InvalidDataException("Layer count mismatch");

            var layers = new List<LayerWeights>();
            for (int i = 0; i < layerCount; i++)
            {
                layers.Add(new LayerWeights
                {
                    Query = ReadValueMatrix(),
                    Key = ReadValueMatrix(),
                    Value = ReadValueMatrix(),
                    Output = ReadValueMatrix(),
                    FeedForwardOne = ReadValueMatrix(),
                    FeedForwardTwo = ReadValueMatrix()
                });
            }

            var outputHead = ReadValueMatrix();

            return new TinyJarvisModel(embeddingSize, headCount, layerCount, maxSeqLen, tokenEmbeddings, positionEmbeddings, outputHead, layers, random, bos, eos);
        }
    }
}
