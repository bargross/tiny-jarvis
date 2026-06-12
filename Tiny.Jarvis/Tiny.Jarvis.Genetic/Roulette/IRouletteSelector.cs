namespace Tiny.Jarvis.Genetic.Roulette
{
    public interface IRouletteSelector
    {
        int Select(double[] fitness, double totalFitness, Random random);

        /// <summary>
        /// Copies source to destination if roulette is within [0, threshold].
        /// </summary>
        void CopyIfInRange(int[] source, int[] destination, int roulette, int threshold);

        /// <summary>
        /// Overload: copies if roulette is within [0, sound].
        /// </summary>
        void CopyIfInRange(int[] source, int[] destination, long sound, int roulette);

        /// <summary>
        /// Ensures that a second roulette value is not simultaneously inside [sound1, sound2]
        /// together with the first roulette value. If both are inside, the second is re‑generated.
        /// </summary>
        /// <param name="firstRoulette">The first roulette value (unchanged).</param>
        /// <param name="secondRoulette">The second roulette value (may be mutated).</param>
        /// <param name="min">Minimum inclusive bound (sound1).</param>
        /// <param name="max">Maximum inclusive bound (sound2).</param>
        /// <returns>The adjusted (or original) second roulette value.</returns>
        int EnsureSecondRouletteOutOfRange(int firstRoulette, int secondRoulette, int min, int max, Random random);
    }
}
