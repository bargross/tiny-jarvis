using System.Diagnostics.CodeAnalysis;

namespace Tiny.Jarvis.Training.ControlFlow
{
    public class Either<TLeft, TRight>
    {
        public TLeft? Left { get; private set; }
        public TRight? Right { get; private set; }

        public bool IsLeft { get; }
        public bool IsRight { get; }

        public Either([NotNull] TLeft left)
        {
            Left = left;
            Right = default;

            IsLeft = true;
            IsRight = false;
        }

        public Either([NotNull] TRight right)
        {
            Left = default;
            Right = right;

            IsLeft = false;
            IsRight = true;
        }

        public TResult Match<TResult>(
            Func<TLeft, TResult> leftFunc,
            Func<TRight, TResult> rightFunc) => IsLeft ? leftFunc(Left!) : rightFunc(Right!);

        public static implicit operator Either<TLeft, TRight>(TLeft left) => new Either<TLeft, TRight>(left);
        public static implicit operator Either<TLeft, TRight>(TRight right) => new Either<TLeft, TRight>(right);

        public static explicit operator TLeft(Either<TLeft, TRight> either) => either.Left;
        public static explicit operator TRight(Either<TLeft, TRight> either) => either.Right;
    }
}
