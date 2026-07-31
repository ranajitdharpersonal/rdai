"""Route a coding request, while retaining automatic provider failover."""

from rdai import AI


def main() -> None:
    ai = AI(strategy="smart")
    answer = ai.generate(
        "Write a typed Python function that retries an asynchronous operation.",
    )
    print(answer)


if __name__ == "__main__":
    main()
