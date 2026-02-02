def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values)


result = compute_mean("not a list")
print(result)
