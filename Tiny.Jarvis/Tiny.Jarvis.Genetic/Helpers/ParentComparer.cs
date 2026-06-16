namespace Tiny.Jarvis.Genetic.Helpers
{
    public static class ParentComparer
    {
        public static bool AreEqual<IPopulationType>(IPopulationType[] parentA, IPopulationType[] parentB) where IPopulationType: struct
        {
            if (parentA.Length != parentB.Length) return false;
            for (var geneIndex = 0; geneIndex < parentA.Length; geneIndex++)
                if (!EqualityComparer<IPopulationType>.Default.Equals(parentA[geneIndex], parentB[geneIndex])) return false;

            return true;
        }
    }
}
