namespace Tiny.Jarvis.Genetic.Helpers
{
    public static class ParentComparer
    {
        public static bool AreEqual(int[] parentA, int[] parentB)
        {
            if (parentA.Length != parentB.Length) return false;
            for (var geneIndex = 0; geneIndex < parentA.Length; geneIndex++)
                if (parentA[geneIndex] != parentB[geneIndex]) return false;

            return true;
        }
    }
}
