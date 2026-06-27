class Animal:
    def __init__(self, name: str):
        self.name = name
        self.fed = False

    def describe(self) -> str:
        return f"{self.name} is an animal"

    def feed(self) -> None:
        self.fed = True


class Cat(Animal):
    def describe(self) -> str:
        return super().describe() + ", specifically a cat"

    def speak(self) -> str:
        return "Meow"


import statistics


# --- independent "cleaner" components ---
class KeepAll:
    def clean(self, values):
        return values


class DropNegatives:
    def clean(self, values):
        return [v for v in values if v >= 0]


# --- independent "statistic" components ---
class Mean:
    def compute(self, values):
        return statistics.mean(values)


class Maximum:
    def compute(self, values):
        return max(values)


class Summarizer:
    def __init__(self, cleaner, statistic):
        self.cleaner = cleaner
        self.statistic = statistic

    def run(self, values):
        cleaned = self.cleaner.clean(values)
        return self.statistic.compute(cleaned)
