using System.Numerics;
using Tiny.Jarvis.Extensions;
using Tiny.Jarvis.Training.Kernel;
using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Util
{
    public static class Calculate
    {
        /// <summary>
        /// Matrix-vector multiply. Each row of weights is multiplied element‑wise with input and summed.
        /// </summary>
        public static List<Value> Linear(List<Value> input, Value[][] weights) =>
            [.. weights.SelectRow(row => Dot(row, input))];

        public static List<Value> Softmax(List<Value> logits, bool useGpu = false)
        {
            if (useGpu && GpuSoftmax.IsInitialized)
                return GpuSoftmax.Softmax(logits);

            else return SoftmaxCpu(logits);
        }

        /// <summary>
        /// Converts raw logits into a probability distribution using softmax.
        /// </summary>
        public static List<Value> SoftmaxCpu(List<Value> logits)
        {
            if (logits == null || logits.Count == 0)
                return new List<Value>();

            var maxLogit = logits.Max(value => value.Data);
            var exponentials = logits.Select(value => (value - maxLogit).Exp()).ToList();
            var sumOfExponentials = new Value(0);

            foreach (Value exponential in exponentials)
                sumOfExponentials += exponential;

            return exponentials
                .Select(exponential => exponential / sumOfExponentials)
                .ToList();
        }

        /// <summary>
        /// Computes the dot product of two sequences of Value objects.
        /// Uses SIMD (Vector<double>) for fast numeric computation, but still records
        /// the full computation graph for backpropagation.
        /// </summary>
        public static Value Dot(IEnumerable<Value> leftSequence, IEnumerable<Value> rightSequence)
        {
            // Materialise to arrays for indexed access and SIMD
            var leftValues = leftSequence as Value[] ?? leftSequence.ToArray();
            var rightValues = rightSequence as Value[] ?? rightSequence.ToArray();

            int length = leftValues.Length;

            // Extract raw data into double arrays for SIMD
            var leftData = new double[length];
            var rightData = new double[length];
            for (int index = 0; index < length; index++)
            {
                leftData[index] = leftValues[index].Data;
                rightData[index] = rightValues[index].Data;
            }

            // SIMD‑accelerated dot product
            var dotProduct = 0.0;
            var vectorSize = Vector<double>.Count;
            var i = 0;
            var sumVector = Vector<double>.Zero;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                var leftVector = new Vector<double>(leftData, i);
                var rightVector = new Vector<double>(rightData, i);
                sumVector += leftVector * rightVector;
            }

            // Horizontal sum of the vector
            dotProduct = Vector.Dot(sumVector, Vector<double>.One);

            // Remaining elements (if length not multiple of vectorSize)
            for (; i < length; i++)
                dotProduct += leftData[i] * rightData[i];

            // Build the local gradients: d(dot)/d(left_i) = right_i, d(dot)/d(right_i) = left_i
            var allInputs = leftValues.Concat(rightValues).ToArray();
            var localGradients = new double[allInputs.Length];
            for (var index = 0; index < length; index++)
            {
                localGradients[index] = rightData[index];
                localGradients[length + index] = leftData[index];
            }

            // Return a single Value node that knows how to backpropagate
            return new Value(dotProduct, allInputs, localGradients);
        }

        /// <summary>
        /// Generates a normally distributed (Gaussian) random number using the Box‑Muller transform.
        /// </summary>
        public static double RandomBellCurve(Random randomGenerator, double mean, double standardDeviation)
        {
            var uniformA = 1.0 - randomGenerator.NextDouble();
            var uniformB = 1.0 - randomGenerator.NextDouble();

            var z = Math.Sqrt(-2.0 * Math.Log(uniformA)) * Math.Sin(2.0 * Math.PI * uniformB);

            return mean + standardDeviation * z;
        }

        /// <summary>
        /// RMSNorm: rescales a vector so its root‑mean‑square is close to 1. Keeps activations stable.
        /// </summary>
        public static List<Value> RmsNorm(List<Value> activations)
        {
            var sumOfSquares = new Value(0);
            foreach (Value activation in activations)
                sumOfSquares += activation * activation;

            Value meanSquare = sumOfSquares / activations.Count;
            Value scale = (meanSquare + 1e-5).Pow(-0.5);

            return activations.Select(activation => activation * scale).ToList();
        }

        /// <summary>
        /// In‑place temperature scaling of logits.
        /// </summary>
        public static void ApplyTemperature(List<Value> logits, double temperature)
        {
            if (Math.Abs(temperature - 1.0) > 1e-8)
                for (var index = 0; index < logits.Count; index++)
                    logits[index] /= temperature;   // Divides Value by double – works if operator/ is defined
        }

        public static Value CrossEntropyLoss(List<Value> logits, int nextTokenId)
        {
            // Find the maximum logit for numerical stability (prevents overflow in exp)
            var maxLogit = logits.Max(v => v.Data);

            // Compute exponentiated logits (shifted by maxLogit) and their sum
            var exponentiatedLogits = new double[logits.Count];
            var sumOfExponentiatedLogits = 0.0;
            for (var i = 0; i < logits.Count; i++)
            {
                exponentiatedLogits[i] = Math.Exp(logits[i].Data - maxLogit);
                sumOfExponentiatedLogits += exponentiatedLogits[i];
            }

            // Compute softmax probabilities
            var softmaxProbabilities = new double[logits.Count];
            for (var i = 0; i < logits.Count; i++)
                softmaxProbabilities[i] = exponentiatedLogits[i] / sumOfExponentiatedLogits;

            // Extract the probability of the target token and compute negative log loss
            var targetProbability = softmaxProbabilities[nextTokenId];
            var clampedProbability = targetProbability < 1e-8 ? 1e-8 : targetProbability;
            var lossValue = -Math.Log(clampedProbability);

            // Prepare for backpropagation: local gradient of loss w.r.t each logit
            //    d(loss)/d(logit_i) = softmaxProbability_i - (1 if i == target else 0)
            var localGradients = new double[logits.Count];
            for (var i = 0; i < logits.Count; i++)
                localGradients[i] = softmaxProbabilities[i] - (i == nextTokenId ? 1.0 : 0.0);

            // Return a single Value node that encapsulates the loss and its local gradients
            return new Value(lossValue, logits.ToArray(), localGradients);
        }
    }
}
