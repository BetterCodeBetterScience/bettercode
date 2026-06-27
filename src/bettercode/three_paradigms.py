"""One task, three paradigms: keeping a running mean of a stream of numbers.

The task is deliberately trivial so that the only thing on display is how each
paradigm organizes *state* -- where it lives, and whether it is mutated:

  * Procedural: the data (a record) and the functions that act on it are
    separate; the functions mutate the record passed to them.
  * Object-oriented: the data and the behavior are bundled into an object that
    owns its state and mutates it through methods.
  * Functional: state is immutable; the functions are pure and return a new
    state instead of mutating anything.

All three compute the same answer; the difference is entirely structural.
"""

import statistics



# === Procedural: separate data and functions; functions mutate the data =====
def proc_new() -> dict[str, float]:
    return {"count": 0, "total": 0.0}

def proc_add(acc: dict[str, float], value: float) -> None:
    acc["count"] += 1
    acc["total"] += value

def proc_mean(acc: dict[str, float]) -> float:
    return acc["total"] / acc["count"]

def compute_procedural_result(data: list[float]) -> float:
    acc = proc_new()
    for value in data:
        proc_add(acc, value)
    return proc_mean(acc)


# === Object-oriented: state and behavior bundled; methods mutate self ========
class RunningMean:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value

    @property
    def mean(self) -> float:
        return self.total / self.count

def compute_oop_result(data: list[float]) -> float:
    running = RunningMean()
    for value in data:
        running.add(value)
    return running.mean


# === Functional: immutable state; pure functions return a new state ==========
# first define our State object as a tuple of int and float
from functools import reduce

State = tuple[int, float]  # (count, total)

def func_add(state: State, value: float) -> State:
    count, total = state
    return (count + 1, total + value)

def func_mean(state: State) -> float:
    count, total = state
    return total / count

def compute_functional_result(data: list[float]) -> float:
    final_state = reduce(func_add, data, (0, 0.0))
    return func_mean(final_state)

def _demo() -> None:

    # Procedural: build a record, then mutate it as each value arrives.
    data = [10.0, 12.0, 14.0, 11.0, 13.0]
    procedural_result = compute_procedural_result(data)
    print(f"procedural: {procedural_result}")

    # Object-oriented: the object updates its own state through method calls.
    oop_result = compute_oop_result(data)
    print(f"oop:        {oop_result}")

    # Functional: fold a pure update over the data, threading immutable state.
    functional_result = compute_functional_result(data)
    print(f"functional: {functional_result}")

    assert procedural_result == oop_result == functional_result == statistics.mean(data)

    # The functional update never mutates: applying it to a state returns a new
    # one and leaves the original untouched.
    start: State = (0, 0.0)
    print("adding creates new updated state", func_add(start, 99.0))
    print("functional start state unchanged:", start)


if __name__ == "__main__":
    _demo()
