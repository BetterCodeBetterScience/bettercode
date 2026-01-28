# Memory Profiling Examples

This directory contains standalone memory profiling examples demonstrating common memory issues in scientific computing and how to fix them.

## Overview

Each example is split into separate BAD and GOOD scripts so you can run them independently with clean memory states. This makes the memory profiler output much clearer and avoids memory carryover between versions.

## Installation

```bash
pip install memory_profiler
```

## Examples

### 1. Memory Leak

**Problem**: Accumulating all intermediate results when you only need aggregates.

**Lesson**: Only store what you actually need!

Run the BAD version first:
```bash
python -m memory_profiler src/bettercode/profiling/memory_leak_bad.py
```

**What to watch for**: Memory grows ~80 MB (10 iterations × 8 MB each)

Then run the GOOD version:
```bash
python -m memory_profiler src/bettercode/profiling/memory_leak_good.py
```

**What to watch for**: Memory oscillates but stays relatively constant

---

### 2. Chunked Processing

**Problem**: Loading entire large datasets into memory at once.

**Lesson**: Process data in chunks using online/incremental algorithms.

Run the BAD version first:
```bash
python -m memory_profiler src/bettercode/profiling/chunked_processing_bad.py
```

**What to watch for**: Large spike to ~400 MB when loading full array

Then run the GOOD version:
```bash
python -m memory_profiler src/bettercode/profiling/chunked_processing_good.py
```

**What to watch for**: Lower memory (~40 MB per chunk) throughout processing

---

### 3. Unnecessary Copies

**Problem**: Making defensive copies at every processing step.

**Lesson**: Use in-place operations and avoid unnecessary copies.

Run the BAD version first:
```bash
python -m memory_profiler src/bettercode/profiling/unnecessary_copies_bad.py
```

**What to watch for**: Peak memory ~320 MB (multiple copies)

Then run the GOOD version:
```bash
python -m memory_profiler src/bettercode/profiling/unnecessary_copies_good.py
```

**What to watch for**: Peak memory ~160 MB (minimal copying)

---

### 4. Array Growth

**Problem**: Growing arrays incrementally with append/concatenate.

**Lesson**: Preallocate arrays when you know the final size.

Run the BAD version first:
```bash
python -m memory_profiler src/bettercode/profiling/array_growth_bad.py
```

**What to watch for**: Memory spikes during concatenation

Then run the GOOD version:
```bash
python -m memory_profiler src/bettercode/profiling/array_growth_good.py
```

**What to watch for**: Memory allocated once and filled in-place

---

## Pandas Categorical Example

The `memory_profiling_pandas.py` script demonstrates a different kind of memory optimization specific to pandas DataFrames with categorical data. This uses pandas' `.memory_usage(deep=True)` method rather than line-by-line profiling because Python's string interning makes the line-by-line output less informative.

```bash
python src/bettercode/profiling/memory_profiling_pandas.py
```

**Key insight**: Converting repeated string columns to categorical dtype can reduce memory by 60-90%!

---

## How to Read memory_profiler Output

When you run `python -m memory_profiler script.py`, you'll see output like:

```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    10    50.2 MiB     50.2 MiB           1   @profile
    11                                         def my_function():
    12    58.3 MiB      8.1 MiB           1       data = np.random.randn(1000000)
    13    58.3 MiB      0.0 MiB           1       result = data.mean()
    14    58.3 MiB      0.0 MiB           1       return result
```

- **Mem usage**: Total memory used by the process at that line
- **Increment**: Memory added (or freed with negative) by that line
- **Occurrences**: How many times that line was executed (useful for loops)

---

## Common Patterns

### Memory Leaks
**Symptom**: Memory continuously grows in a loop
**Fix**: Only accumulate aggregates, not all intermediate results

### Out of Memory Errors
**Symptom**: Script crashes with "MemoryError" or "Killed"
**Fix**: Process data in chunks, use online algorithms

### Slow Processing
**Symptom**: Operations take longer than expected
**Fix**: Check for unnecessary copies, use in-place operations

### Inefficient Data Structures
**Symptom**: More memory used than data size suggests
**Fix**: Use appropriate dtypes (e.g., categorical for strings, smaller numeric types)

---

## Tips for Memory Profiling

1. **Run scripts independently**: Each script should start with a clean memory state
2. **Compare before/after**: Always profile both bad and good versions to see the difference
3. **Use realistic data sizes**: Too small and you won't see issues; too large and profiling is slow
4. **Check peak memory**: The maximum memory used matters more than average
5. **Combine with time profiling**: Memory optimization often improves speed too

---

## Additional Resources

- [memory_profiler documentation](https://pypi.org/project/memory-profiler/)
- [Pandas memory optimization guide](https://pandas.pydata.org/docs/user_guide/scale.html)
- [NumPy memory layout](https://numpy.org/doc/stable/reference/arrays.ndarray.html)
