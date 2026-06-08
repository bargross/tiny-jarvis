using Tiny.Jarvis.Models;

namespace Tiny.Jarvis.Extensions
{
    internal static class ValueExtensions
    {
        // Convenience overload: allocates fresh buffers on each call.
        // Good for one-off use in the early chapters. The 3-argument version below
        // lets the training loop in Chapter 7 reuse buffers across thousands of steps.
        public static void Backward(this Value value) => value.Backward([], [], new Stack<(Value, int)>());

        public static void Backward(this Value value,
            List<Value> topo,
            HashSet<Value> visited,
            Stack<(Value current, int inputIndex)> stack
        ) {
            if (value.Grad == 0) value.Grad = 1.0;

            //Console.WriteLine("Backward called, Loss Value: " + value.ToString());
            // Iterative topological sort
            stack.Push((value, 0));

            while (stack.TryPop(out var result))
            {
                var current = result.current;
                if (current == null) continue;

                var inputIndex = result.inputIndex;
                Value[]? inputs = current?.Inputs;

                if (inputs != null && inputIndex < inputs.Length)
                {
                    stack.Push((current!, inputIndex + 1));
                    Value input = inputs[inputIndex];

                    if (visited.Add(input))
                        stack.Push((input, 0));
                }

                else topo.Add(current!);
            }

            // Propagate gradients in reverse topological order
            // Grad = 1.0;
            for (var i = topo.Count - 1; i >= 0; i--)
            {
                var topoValue = topo[i];
                if (topoValue.Grad == 0)
                    continue; // optimisation: nothing to propagate

                if (topoValue.Inputs == null)
                    continue;

                for (var j = 0; j < topoValue.Inputs.Length; j++)
                    topoValue.Inputs[j].Grad += topoValue.LocalGrads[j] * topoValue.Grad;
            }
        }
    }
}
