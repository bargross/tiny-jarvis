using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Tiny.Jarvis.Training.Models;

namespace Tiny.Jarvis.Training.Kernel
{
    public static class GpuSoftmax
    {
        private static Context? _context;
        private static Accelerator? _accelerator;
        private static Action<Index1D, ArrayView<float>, ArrayView<float>, int>? _softmaxKernel;

        public static bool IsInitialized => _accelerator != null;

        public static void Initialize(bool useGpu = false)
        {
            // The context holds the compiler and global state.
            _context = Context.CreateDefault();

            // Get the first GPU device (prefer CPU = false)
            var device = _context.GetPreferredDevice(useGpu);
            if (device == null)
                throw new Exception("No GPU device found");

            // Create accelerator
            _accelerator = device.CreateAccelerator(_context);
            Console.WriteLine($"Using GPU: {_accelerator.Name}");

            // Load kernel
            _softmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(SoftmaxKernel);
        }

        private static void SoftmaxKernel(Index1D idx, ArrayView<float> logits, ArrayView<float> output, int length)
        {
            // Only the first thread does all the work (simple, not parallel)
            if (idx.X == 0)
            {
                // Find max
                var max = float.MinValue;
                for (var i = 0; i < length; i++)
                {
                    var val = logits[i];
                    if (val > max) max = val;
                }

                // Compute exponentials and sum
                var sum = 0.0f;
                for (var i = 0; i < length; i++)
                {
                    var exp = XMath.Exp(logits[i] - max);
                    output[i] = exp;
                    sum += exp;
                }

                // Normalise
                for (var i = 0; i < length; i++)
                    output[i] /= sum;
            }
        }

        public static List<Value> Softmax(List<Value> logits)
        {
            if (_accelerator == null || _softmaxKernel == null)
                throw new InvalidOperationException("GPU not initialized");

            var length = logits.Count;
            var logitsArray = logits.Select(v => (float)v.Data).ToArray();

            // Allocate GPU buffers
            using var bufferLogits = _accelerator.Allocate1D(logitsArray);
            using var bufferOutput = _accelerator.Allocate1D<float>(length);

            // Launch kernel with 1 thread (gridDim = 1, blockDim = 1)
            _softmaxKernel(1, bufferLogits.View, bufferOutput.View, length);
            _accelerator.Synchronize();

            // Copy result back to CPU
            var resultArray = bufferOutput.GetAsArray1D();
            return resultArray.Select(f => new Value(f)).ToList();
        }

        public static void Dispose()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
        }
    }
}
