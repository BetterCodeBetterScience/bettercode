from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True, order=True)
class Interval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Invalid interval [{self.start}, {self.end}] (end < start)")


def merge_overlapping(intervals: Iterable[Interval]) -> List[Interval]:
    """
    Merge intervals that OVERLAP.

    Definition used by this function (as intended/documented):
      Two intervals [a,b] and [c,d] overlap iff max(a,c) < min(b,d).
      (Touching at a point is NOT overlap: [1,2] and [2,3] do not overlap.)

    Returns a list of disjoint intervals sorted by start.

    BUG: The implementation incorrectly merges "touching" intervals as well.
    """
    items = sorted(intervals)
    if not items:
        return []

    out: List[Interval] = [items[0]]
    for cur in items[1:]:
        last = out[-1]

        # --- BUG HERE ---
        # This condition treats "touching" as overlap by using <= instead of <.
        if cur.start <= last.end:
            out[-1] = Interval(last.start, max(last.end, cur.end))
        else:
            out.append(cur)

    return out
