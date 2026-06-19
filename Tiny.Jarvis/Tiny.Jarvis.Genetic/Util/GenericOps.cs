using Tiny.Jarvis.Genetic.Enums;

namespace Tiny.Jarvis.Genetic.Util
{
    public static class GenericOps
    {
        public static TPopulation GetNextByType<TPopulation>(Random random, TPopulation? minGeneValue = default, TPopulation? maxGeneValue = default)
        {
            return minGeneValue switch
            {
                int i => (TPopulation)(object)random.Next((int)(object)minGeneValue, (int)(object)maxGeneValue + 1),
                float i => (TPopulation)(object)random.NextDouble(),
                double i => (TPopulation)(object)random.NextDouble(),
                decimal i => (TPopulation)(object)random.NextDouble(),
                long i => (TPopulation)(object)random.NextInt64((int)(object)minGeneValue, (int)(object)maxGeneValue + 1),
                _ => throw new ArgumentException("Unknown Type")
            };
        }

        public static (TPopulation min, TPopulation max) GetGeneMinMaxValues<TPopulation>()
        {
            var populationType = typeof(TPopulation);
            if (populationType == typeof(int)) return (ConvertToGenericPopValue<TPopulation, int>(1), ConvertToGenericPopValue<TPopulation, int>(100));
            if (populationType == typeof(long)) return (ConvertToGenericPopValue<TPopulation, long>(1), ConvertToGenericPopValue<TPopulation, long>(100));
            if (populationType == typeof(float)) return (ConvertToGenericPopValue<TPopulation, float>(0.1f), ConvertToGenericPopValue<TPopulation, float>(0.9f));
            if (populationType == typeof(double)) return (ConvertToGenericPopValue<TPopulation, double>(0.1), ConvertToGenericPopValue<TPopulation, double>(0.9));
            if (populationType == typeof(decimal)) return (ConvertToGenericPopValue<TPopulation, decimal>(0.1m), ConvertToGenericPopValue<TPopulation, decimal>(0.9m));

            throw new ArgumentException("Uknown population type");
        }

        public static TResult ConvertTo<TPopulation, TResult>(TPopulation value)
        {
            if (typeof(TPopulation) != typeof(TResult)) throw new ArgumentException("Types do not match!");

            return (TResult)(object)value;
        }

        public static int Compare<TPopulation>(TPopulation a, TPopulation b) where TPopulation : IComparable<TPopulation> => a.CompareTo(b); 

        public static TPopulation ConvertToGenericPopValue<TPopulation, TValue>(TValue a) => a switch
        {
            int c => (TPopulation)(object)a,
            long c => (TPopulation)(object)a,
            double c => (TPopulation)(object)a,
            float c => (TPopulation)(object)a,
            decimal c => (TPopulation)(object)a,
            _ => throw new ArgumentException("Invalid type!")
        };

        public static TPopulation PerformMathematicalOperation<TPopulation>(TPopulation valA, TPopulation valB, MathOperation op)
        {
            TPopulation SumInt(TPopulation a, TPopulation b) => (TPopulation)(object)((int)(object)valA + (int)(object)valB);
            TPopulation SumDouble(TPopulation a, TPopulation b) => (TPopulation)(object)((double)(object)valA + (double)(object)valB);
            TPopulation SumLong(TPopulation a, TPopulation b) => (TPopulation)(object)((long)(object)valA + (long)(object)valB);
            TPopulation SumDecimal(TPopulation a, TPopulation b) => (TPopulation)(object)((decimal)(object)valA + (decimal)(object)valB);
            TPopulation SumFloat(TPopulation a, TPopulation b) => (TPopulation)(object)((float)(object)valA + (float)(object)valB);

            TPopulation SubtractInt(TPopulation a, TPopulation b) => (TPopulation)(object)((int)(object)valA - (int)(object)valB);
            TPopulation SubtractLong(TPopulation a, TPopulation b) => (TPopulation)(object)((long)(object)valA - (long)(object)valB);
            TPopulation SubtractDouble(TPopulation a, TPopulation b) => (TPopulation)(object)((double)(object)valA - (double)(object)valB);
            TPopulation SubtractDecimal(TPopulation a, TPopulation b) => (TPopulation)(object)((decimal)(object)valA - (decimal)(object)valB);
            TPopulation SubtractFloat(TPopulation a, TPopulation b) => (TPopulation)(object)((float)(object)valA - (float)(object)valB);

            TPopulation DividetInt(TPopulation a, TPopulation b) => (TPopulation)(object)((int)(object)valA / (int)(object)valB);
            TPopulation DividetLong(TPopulation a, TPopulation b) => (TPopulation)(object)((long)(object)valA / (long)(object)valB);
            TPopulation DivideDouble(TPopulation a, TPopulation b) => (TPopulation)(object)((double)(object)valA / (double)(object)valB);
            TPopulation DivideDecimal(TPopulation a, TPopulation b) => (TPopulation)(object)((decimal)(object)valA / (decimal)(object)valB);
            TPopulation DivideFloat(TPopulation a, TPopulation b) => (TPopulation)(object)((float)(object)valA / (float)(object)valB);

            TPopulation MultiplyInt(TPopulation a, TPopulation b) => (TPopulation)(object)((int)(object)valA * (int)(object)valB);
            TPopulation MultiplyLong(TPopulation a, TPopulation b) => (TPopulation)(object)((long)(object)valA * (long)(object)valB);
            TPopulation MultiplyDouble(TPopulation a, TPopulation b) => (TPopulation)(object)((double)(object)valA * (double)(object)valB);
            TPopulation MultiplyDecimal(TPopulation a, TPopulation b) => (TPopulation)(object)((decimal)(object)valA * (decimal)(object)valB);
            TPopulation MultiplyFloat(TPopulation a, TPopulation b) => (TPopulation)(object)((float)(object)valA * (float)(object)valB);

            TPopulation SumByType(TPopulation valA, TPopulation valB)
            {
                return valA switch
                {
                    int i => SumInt(valA, valB),
                    long r => SumLong(valA, valB),
                    double b => SumDouble(valA, valB),
                    decimal s => SumDecimal(valA, valB),
                    float d => SumFloat(valA, valB),
                    _ => throw new ArgumentException("Type is not valid")
                };
            }

            TPopulation SubtractByType(TPopulation valA, TPopulation valB)
            {
                return valA switch
                {
                    int i => SubtractInt(valA, valB),
                    long r => SubtractLong(valA, valB),
                    double b => SubtractDouble(valA, valB),
                    decimal s => SubtractDecimal(valA, valB),
                    float d => SubtractFloat(valA, valB),
                    _ => throw new ArgumentException("Type is not valid")
                };
            }

            TPopulation DivideByType(TPopulation valA, TPopulation valB)
            {
                return valA switch
                {
                    int i => DividetInt(valA, valB),
                    long r => DividetLong(valA, valB),
                    double b => DivideDouble(valA, valB),
                    decimal s => DivideDecimal(valA, valB),
                    float d => DivideFloat(valA, valB),
                    _ => throw new ArgumentException("Type is not valid")
                };
            }

            TPopulation MultiplyByType(TPopulation valA, TPopulation valB)
            {
                return valA switch
                {
                    int i => MultiplyInt(valA, valB),
                    long r => MultiplyLong(valA, valB),
                    double b => MultiplyDouble(valA, valB),
                    decimal s => MultiplyDecimal(valA, valB),
                    float d => MultiplyFloat(valA, valB),
                    _ => throw new ArgumentException("Type is not valid")
                };
            }

            return op switch
            {
                MathOperation.Addition => SumByType(valA, valB),
                MathOperation.Subtraction => SubtractByType(valA, valB),
                MathOperation.Division => DivideByType(valA, valB),
                MathOperation.Multiplication => MultiplyByType(valA, valB)
            };
        }
    }
}
