"""Minimal rdai SDK example.

Run after adding at least one API key to a local .env file.
"""

from rdai import AI


def main() -> None:
    ai = AI(strategy="smart")
    print(ai.generate("Explain dependency inversion in two sentences."))


if __name__ == "__main__":
    main()

