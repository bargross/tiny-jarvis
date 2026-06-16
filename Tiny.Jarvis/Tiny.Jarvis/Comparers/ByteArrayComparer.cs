namespace Tiny.Jarvis.Training.Comparers
{
    public sealed class ByteArrayComparer : IEqualityComparer<byte[]>
    {
        public bool Equals(byte[] x, byte[] y)
        {
            if (ReferenceEquals(x, y)) return true;

            if (x == null || y == null) return false;

            if (x.Length != y.Length) return false;

            for (int i = 0; i < x.Length; i++)
                if (x[i] != y[i]) return false;

            return true;
        }

        public int GetHashCode(byte[] obj)
        {
            if (obj == null) return 0;

            var hash = 17;

            foreach (byte b in obj)
                hash = hash * 31 + b;

            return hash;
        }
    }
}
