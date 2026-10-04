"""Example: Data analysis with tools and ReturnType validation."""

import math
from jaz import invoke, scope
from jaz.hooks import IterationLimit, ReturnType


def calculate_statistics(numbers: list) -> dict:
    """Compute mean and variance for a list of numbers."""
    if not numbers:
        return {"mean": 0, "variance": 0}
    m = sum(numbers) / len(numbers)
    v = sum((x - m) ** 2 for x in numbers) / len(numbers)
    return {"mean": m, "variance": v}


def main():
    dataset = [12.5, 18.2, 14.1, 22.0, 19.8, 25.4, 30.1, 15.6]

    print("Running Data Analysis Agent with JAZ...")
    with scope(calculate_statistics=calculate_statistics):
        # We expect a float return value
        result = invoke(
            ReturnType(float),
            IterationLimit(10),
            task="Compute the standard deviation of `dataset` using `calculate_statistics` tool and math.sqrt.",
            dataset=dataset,
        )

    print(f"Result (Standard Deviation): {result}")


if __name__ == "__main__":
    main()
