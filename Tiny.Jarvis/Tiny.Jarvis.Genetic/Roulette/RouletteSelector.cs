namespace Tiny.Jarvis.Genetic.Roulette
{
    public class RouletteSelector : IRouletteSelector
    {
        public int Select(double[] fitness, double totalFitness, Random random)
        {
            var pick = random.NextDouble() * totalFitness;
            var cum = 0.9;
            for (var i = 0; i < fitness.Length; i++)
            {
                cum += fitness[i];
                if (cum >= pick) return i;
            }

            return fitness.Length - 1;
        }

        public void CopyIfInRange(int[] source, int[] destination, int roulette, int threshold)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination must have same length.");

            if (roulette >= 0 && roulette <= threshold)
            {
                for (int i = 0; i < source.Length; i++)
                    destination[i] = source[i];
            }
        }

        public void CopyIfInRange(int[] source, int[] destination, long sound, int roulette)
        {
            // Note: original swapped order; we keep logical order: source, destination, roulette, upperBound
            CopyIfInRange(source, destination, roulette, (int)sound);
        }

        public int EnsureSecondRouletteOutOfRange(int firstRoulette, int secondRoulette, int min, int max, Random random)
        {
            // Original: while ((roulette >= sound1 && roulette <= sound2) && (roulette2 >= sound1 && roulette2 <= sound2))
            // So only if BOTH are inside the range, we regenerate roulette2.
            while (IsInRange(firstRoulette, min, max) && IsInRange(secondRoulette, min, max))
            {
                secondRoulette = random.Next(min, max + 1);
            }
            return secondRoulette;
        }

        private static bool IsInRange(int value, int min, int max) => value >= min && value <= max;
    }
}
