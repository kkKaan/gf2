# Benchmarks

Two things live here: a measurement harness with rules, and the scripts that use it.

```bash
python benchmarks/bench_gf2.py      # measure -> benchmarks/results.json
python benchmarks/make_report.py    # results.json -> BENCHMARK_RESULTS.md
```

`BENCHMARK_RESULTS.md` is **generated**. Never edit it by hand.

## Files

| file | what it is |
|---|---|
| `harness.py` | measurement primitives: timing, peak allocation, equivalence checking |
| `bench_gf2.py` | the canonical suite - gf2 vs packed NumPy vs naive NumPy vs galois |
| `make_report.py` | turns `results.json` into `BENCHMARK_RESULTS.md` |
| `results.json` | raw measurements from the last run |
| `benchmark_vs_numpy.py` | older chart-producing script, kept for the plots |
| `benchmark_gf2_libraries.py` | older GF(2)-library comparison, kept for the plots |
| `memory_profile.py` | memory-only profiling |

## The rules, and why each one exists

Each of these is a mistake this repository actually made. They are written down
so they are not made again.

### 1. Never time code while `tracemalloc` is running

The original harness called `tracemalloc.start()` and then `time.perf_counter()`,
so every published timing was taken under an allocation tracer.

Measured cost of that mistake on this codebase:

| operation | clean | under tracemalloc | inflation |
|---|---:|---:|---:|
| multiply 100x100 | 37.1 ms | 960.6 ms | 25.9x |
| multiply 200x200 | 281.3 ms | 7723.0 ms | 27.5x |
| rank 100x100 | 1.08 ms | 16.4 ms | 15.2x |
| rank 200x200 | 3.01 ms | 41.6 ms | 13.8x |

The inflation is not a constant, so it does not cancel out of a ratio: it
scales with how many Python-level allocations the implementation makes, which
penalised gf2's big-integer paths far more than NumPy's array paths. The old
`BENCHMARK_RESULTS.md` reported "multiply 200x200: 7612 ms", a figure that was
roughly 27x the real cost of the code it was measuring.

Time and memory are now measured in separate passes.

### 2. Report the minimum, show the median

For deterministic CPU work, run-to-run variation is almost entirely
interference from the rest of the machine, and it only ever makes a run
*slower*. The minimum of N runs is therefore the best estimate of the true
cost. The mean folds the noise back in.

The median is printed next to it. If they diverge, the measurement is unstable
and should not be quoted.

### 3. Warm up first

The first call to any gf2 routine may build the matrix's packed-row cache,
import NumPy machinery, or trigger a JIT inside `galois`. One untimed warm-up
call keeps that out of the numbers.

### 4. Check the outputs agree before comparing the speeds

Every contender's result is normalised and compared against a reference; the
`=?` column in the report is that check. A fast wrong answer is not a result.

Nullspace is checked by verifying `A @ x == 0` and `x != 0`, not by comparing
vectors, because different algorithms legitimately pick different free
variables and both answers are correct.

### 5. Compare the same amount of work, and say so when you cannot

`galois.null_space()` returns the entire basis; gf2's `nullspace_fast`
returns one vector. Those are not the same task. The suite still reports the
pair because it is the comparison users ask about, but the report labels the
galois row "computes FULL basis, then takes row 0" so nobody reads it as a
like-for-like win.

### 6. Time setup separately

Building each library's native matrix from the shared input is its own
measurement. Folding it into one side's operation is how a comparison silently
becomes a comparison of constructors.

### 7. Pick a baseline that is trying

The original suite's "NumPy" rank was a Python loop indexing single `uint8`
elements. Most of its time went on allocating NumPy scalar objects, not on
arithmetic, and it is 9-13x slower than gf2 - a number that says nothing.

The suite now reports both:

- **`numpy-packed`** - bit-packed `uint64` rows, pivot search and elimination
  vectorised with `flatnonzero`. This is what a competent NumPy user writes
  for GF(2), and **the only baseline a speed claim should be made against.**
- **`numpy-naive`** - the element-wise version, kept solely to show the size of
  the gap between the two NumPy styles.

At n=512 the packed version is 10.9x faster than the naive one. Choosing which
of them to call "NumPy" decides the headline before any measurement happens.

### 8. Generate the report from the data

The old `BENCHMARK_RESULTS.md` was maintained by hand and drifted away from the
code that was meant to produce it. It:

- attributed NumPy's rank to "highly optimized LAPACK routines (QR
  decomposition)" when the benchmark ran a hand-written Python loop;
- quoted `np.linalg.svd` timings of "50-70ms" that no script here produces, and
  compared a real-valued SVD nullspace to a GF(2) nullspace as if they were the
  same computation;
- claimed "gf2 uses 75-90% less memory" four lines above its own table
  showing gf2 using 60% *more* at 500x500;
- claimed "NumPy would be 3-4x slower" for Simon's algorithm with no
  measurement behind it;
- signed off "Auto-generated by benchmark_vs_numpy.py", which generates no
  markdown.

`make_report.py` now writes the file from `results.json`, so a claim that is
not in the measurements cannot appear in the report.
