# Benchmark report: batching and CPU threads (Stage 1)

How does batch size affect inference time for the main model, and how many CPU threads should a
worker use? This report records every run, the code used for each, what changed between runs
and why, and the design decisions the numbers led to.

All numbers below come from runs on the hardware listed in the setup section. The raw output of
each run, including hardware and library versions, is committed as JSON in `bench/results/`.

## Summary

| Finding | Evidence |
|---|---|
| Batching 16 requests makes each one **4.2× cheaper** (13.6 → 3.2 ms per item), not the ~10× the design assumed. | Every run, table 1 |
| On Apple M1, using all 8 cores is **slower** than using only the 4 performance cores, and it makes worst-case (p99) batch times **about 2× worse**. | Runs B vs C, table 2 |
| With 4 threads, a batch of 16 takes 51.8 ms typically and 54.6 ms at p99, so its timing is tight and predictable. | Run B |
| The neural network itself (forward pass) is **98%** of batch time. Tokenization is under 2%. | Run B, table 3 |
| At batch size 16, the worst case at the backpressure threshold (~164 ms) **misses the 150 ms target**, even before network overhead. Batch size 8 fits (~100 ms). | Table 4 |
| The sklearn model costs a flat **0.12 ms** per call whatever the batch size, so all of it is fixed per-call cost. | Run S |

## Setup

- **Machine:** Apple M1, 8 cores (**4 performance + 4 efficiency**), macOS
- **Python:** 3.12.11. torch/transformers versions are recorded in each JSON file.
- **Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, CPU, eval mode,
  `torch.inference_mode()`
- **Inputs:** random 24-word sentences, about 26 tokens after tokenization, so padding is about
  the same in every batch. Each timed batch gets new inputs.
- **Method:** for each batch size, 5 warm-up batches (not recorded), then N timed batches.
  Timing uses `time.perf_counter_ns()` and covers tokenize + forward + postprocess.
- **Scope:** this is the model alone. There's no Redis, HTTP or JSON, so these numbers are a
  lower bound on server latency. The full request path gets measured in Stage 11.

How to read the columns: **p50** is the median batch time, and **p99** means 99% of batches
finished faster. **ms/item** = p50 ÷ batch size. **speedup** = ms/item at batch size 1 ÷ ms/item
at this size. With N timed batches, p99 is roughly the ⌈N/100⌉-th slowest sample, so it's noisy
at N=100.

## The runs

| Run | Script | Model | Threads | Iterations | Result file (`bench/results/`) |
|---|---|---|---|---|---|
| A | `scripts/bench_batching.py` | DistilBERT | 8 | 100 | `stage1-transformer_classifier-20261106T150229Z.json` |
| S | `scripts/bench_batching.py` | sklearn (iris) | n/a | 100 | `stage1-sklearn_classifier-20261106T150255Z.json` |
| B | `scripts/bench_batching_1.py` | DistilBERT | **4** | 300 | `stage1-transformer_classifier-20261106T151131Z.json` |
| C | `scripts/bench_batching_1.py` | DistilBERT | 8 | 300 | `stage1-transformer_classifier-20261106T151327Z.json` |

### Run A: first run, all 8 threads

```
batch    p50 ms    p90 ms    p99 ms   ms/item   items/s  speedup   tok ms   fwd ms  post ms  tokens
    1     14.22     14.58     16.11    14.216        70     1.0x     0.17    14.03     0.03    26.0
    2     18.04     18.37     20.70     9.020       111     1.6x     0.22    17.72     0.09    26.0
    4     24.99     26.51     32.45     6.247       160     2.3x     0.34    24.53     0.10    26.0
    8     34.41     38.77     67.28     4.302       232     3.3x     0.54    33.80     0.10    26.0
   16     54.59     55.05     88.31     3.412       293     4.2x     0.96    53.51     0.10    26.0
   32     97.60     97.96    100.05     3.050       328     4.7x     1.80    95.68     0.11    26.0

Latency budget (model only), max_batch_size=16, max_wait=15 ms, full batch p99=88.3 ms
  light_load                     103.3 ms   ok
  saturated                      176.6 ms   OVER
  at_backpressure_threshold      264.9 ms   OVER
```

**Observation that led to runs B and C:** at batch sizes 8 and 16, p90 is close to p50 but p99
jumps 60–75% higher (16: 55.1 → 88.3 ms). A tight distribution with a few big outliers suggests a
systematic cause, not random noise. The guess: the M1 has 4 fast and 4 slow cores, and
`threads=8` puts work on the slow ones. PyTorch splits each operation evenly across its threads,
so a batch finishes only when the slowest thread does. Testing this needed two things: knowing
the core layout, and running with threads limited to the performance cores.

### Change made between runs: `bench_batching.py` → `bench_batching_1.py`

The original script printed the CPU as just `arm`. It reads the CPU model from `/proc/cpuinfo`,
which exists only on Linux, and `platform.processor()` on macOS returns only the architecture.
So it couldn't show the core layout the guess depended on.

`scripts/bench_batching_1.py` is a copy with **reporting changes only**:

- reads the chip name with `sysctl -n machdep.cpu.brand_string` (e.g. `Apple M1`)
- reads the performance/efficiency core counts from `hw.perflevel0.physicalcpu` /
  `hw.perflevel1.physicalcpu`, prints them (`4P + 4E`), and saves them in the JSON

**Measurement code is unchanged**: warm-up, timing, percentiles, input generation and the budget
formulas are identical, so results from the two scripts are directly comparable. Both files are
kept so run A can be reproduced exactly; `diff scripts/bench_batching.py
scripts/bench_batching_1.py` shows the full change (about 25 lines).

Runs B and C also raise iterations from 100 to 300, so p99 rests on the 3rd-slowest sample
instead of the 1st.

### Run B: 4 threads (performance cores only)

```
Apple M1, 8 logical cores (4P + 4E), threads=4
batch    p50 ms    p90 ms    p99 ms   ms/item   items/s  speedup   tok ms   fwd ms  post ms  tokens
    1     13.56     13.94     15.38    13.563        74     1.0x     0.16    13.38     0.02    26.0
    2     16.99     19.89     32.84     8.494       118     1.6x     0.22    16.70     0.06    26.0
    4     22.04     22.19     24.50     5.511       181     2.5x     0.32    21.66     0.06    26.0
    8     31.62     31.80     33.44     3.952       253     3.4x     0.54    31.02     0.06    26.0
   16     51.76     52.04     54.61     3.235       309     4.2x     0.96    50.73     0.06    26.0
   32     93.52     95.41    156.08     2.922       342     4.6x     1.80    91.64     0.07    26.0

Latency budget (model only), max_batch_size=16, max_wait=15 ms, full batch p99=54.6 ms
  light_load                      69.6 ms   ok
  saturated                      109.2 ms   ok
  at_backpressure_threshold      163.8 ms   OVER
```

### Run C: 8 threads again, same method as B

```
Apple M1, 8 logical cores (4P + 4E), threads=8
batch    p50 ms    p90 ms    p99 ms   ms/item   items/s  speedup   tok ms   fwd ms  post ms  tokens
    1     14.28     15.20     17.69    14.278        70     1.0x     0.16    14.09     0.03    26.0
    2     18.27     19.97     30.64     9.136       109     1.6x     0.22    17.96     0.09    26.0
    4     24.11     24.85     26.20     6.027       166     2.4x     0.33    23.69     0.09    26.0
    8     34.05     34.75     36.97     4.257       235     3.4x     0.55    33.41     0.09    26.0
   16     54.62     55.08    107.75     3.414       293     4.2x     0.97    53.56     0.10    26.0
   32     97.67     98.13    102.98     3.052       328     4.7x     1.80    95.76     0.11    26.0

Latency budget (model only), max_batch_size=16, max_wait=15 ms, full batch p99=107.8 ms
  light_load                     122.8 ms   ok
  saturated                      215.5 ms   OVER
  at_backpressure_threshold      323.3 ms   OVER
```

### Run S: sklearn contrast model

```
batch    p50 ms    p90 ms    p99 ms   ms/item   items/s  speedup
    1      0.12      0.17      0.28     0.121      8287     1.0x
    2      0.12      0.12      0.15     0.058     17285     2.1x
    4      0.12      0.12      0.14     0.030     33862     4.1x
    8      0.12      0.12      0.14     0.015     67168     8.1x
   16      0.12      0.13      0.15     0.008    128385    15.5x
   32      0.13      0.13      0.14     0.004    247104    29.8x
```

## Analysis

### Table 1: how much batching helps (run B, 4 threads)

| Batch size | ms per item | Speedup vs 1 | Throughput (items/s, one worker) |
|---|---|---|---|
| 1 | 13.56 | 1.0× | 74 |
| 8 | 3.95 | 3.4× | 253 |
| 16 | 3.24 | 4.2× | 309 |
| 32 | 2.92 | 4.6× | 342 |

Gains flatten fast: going from 16 to 32 adds only 11% throughput and nearly doubles batch time
(51.8 → 93.5 ms). The ~10× in the original design was an illustrative guess; the measured
figure on this machine is 4.2× at batch size 16. Runs A and C (8 threads) give the same 4.2×.

### Table 2: 4 threads vs 8 threads (runs B and C, 300 iterations each)

| Batch | 4 threads p50 / p99 (ms) | 8 threads p50 / p99 (ms) | p99 ÷ p50, 4 vs 8 threads |
|---|---|---|---|
| 1 | 13.56 / 15.38 | 14.28 / 17.69 | 1.13 vs 1.24 |
| 8 | 31.62 / 33.44 | 34.05 / 36.97 | 1.06 vs 1.09 |
| 16 | 51.76 / 54.61 | 54.62 / **107.75** | 1.06 vs **1.97** |
| 32 | 93.52 / **156.08** | 97.67 / 102.98 | **1.67** vs 1.05 |

- **The median is lower with 4 threads at every batch size** (4–7%). Doubling the threads to
  include the efficiency cores makes the typical batch *slower*, not faster.
- **The tail at batch size 16, the size we'd deploy, collapses with 4 threads:** p99 is 1.06× the
  median instead of 1.97×. Run A, an independent 8-thread run, showed the same pattern (88.3 ms),
  so this is reproducible, not a one-off.
- **Not every spike is explained.** Run B has an outlier at batch size 32 (p99 156 ms), and both
  300-iteration runs show one at batch size 2 (~31–33 ms). With 300 samples, p99 is the
  3rd-slowest batch, so three slow batches are enough to cause this. Likely causes are other
  processes, the OS scheduler or thermal limits on a laptop. The claim backed by this data is
  narrower: 8 threads adds a *systematic* tail at batch size 16, and 4 threads removes it.

**Why:** PyTorch splits each matrix multiply into equal chunks, one per thread. On the M1, chunks
given to efficiency cores take longer, and the operation can't finish until the last chunk
does, so every batch waits for the slowest core. Threads beyond the performance-core count add
waiting, not speed.

### Table 3: where batch time goes (run B, batch size 16)

| Phase | ms | Share |
|---|---|---|
| Tokenize | 0.96 | 1.9% |
| Forward (neural network) | 50.73 | 98.0% |
| Postprocess | 0.06 | 0.1% |
| **Total** | 51.76 | |

Tokenization and postprocessing are the Python-side work, which is where Python's GIL matters.
They're about 2% of batch time. PyTorch releases the GIL during the forward pass. This supports
design decision D7: one worker process with multithreaded PyTorch, plus a separate collector
thread, loses almost nothing to the GIL.

### Table 4: does the 150 ms p99 target hold? (4 threads)

The worker overlaps collecting the next batch with computing the current one. The worst-case
model-side wait for a request depends on load, with T = p99 time of a full batch:

| Situation | Formula | Batch size 16 (T = 54.6) | Batch size 8 (T = 33.4) |
|---|---|---|---|
| Light load | max_wait + T | 69.6 ms | 48.4 ms |
| Saturated | 2T | 109.2 ms | 66.9 ms |
| At backpressure threshold (one full batch queued per worker) | 3T | **163.8 ms** | 100.3 ms |

Even using the median instead of p99, batch size 16 at the threshold is 3 × 51.8 = 155 ms. That
misses 150 ms with no room left for Redis, JSON or HTTP. Batch size 8 leaves about 50 ms for that
overhead, and costs 18% of peak throughput (253 vs 309 items/s).

(Light load only reaches max_wait + T if the batch fills up. At light load, batches are usually
smaller and faster, so these are upper bounds.)

### Sklearn

Batch time is flat at about 0.12 ms for 1 to 32 items, so all of it is fixed per-call
overhead (Python, input validation). Strictly, batching does lower the per-item cost (15× at
16). But the cost is so small that waiting up to `max_wait` (15 ms) to fill a batch would cost
far more than it saves. So batching stays off for this model, for that reason rather than the
original one ("batching doesn't help").

## Decisions

Status: **proposed, pending confirmation.** This section will be updated when confirmed.

1. **Inference threads = number of performance cores** (4 on this M1). On machines where all
   cores are the same (typical Linux servers), use the physical core count. The Stage 11 layout
   comparison becomes 1×4, 2×2 and 4×1 on this machine, plus one 8-thread layout to show the
   effect of the efficiency cores under load.
2. **Main model `max_batch_size`: 16 → 8**, so the worst case at the backpressure threshold fits
   the 150 ms target with headroom for network overhead. Trade-off: −18% peak throughput per
   worker. Stage 11 re-measures this end to end.
3. **Backpressure threshold = `max_batch_size × workers`** (was the fixed `16 × workers`), so it
   stays one full batch per worker if the batch size changes.
4. **"Target load" = at most 70% of a worker's measured throughput.** The p99 < 150 ms target
   applies there. The saturated and threshold cases are overload, where 503s take over.
5. **`max_wait` stays at 15 ms** until the Stage 11 sweep.
6. **sklearn stays at `max_batch_size = 1`**, with the reason corrected as above.

## Reproduce

```bash
uv sync --extra worker
uv run python -m scripts.fetch_sentiment_model
uv run python -m scripts.train_iris

uv run python -m scripts.bench_batching   --model-dir artifacts/sentiment-v1                       # run A
uv run python -m scripts.bench_batching   --kind sklearn_classifier --model-dir artifacts/iris-v1  # run S
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 300  # run B
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 8 --iters 300  # run C
```

Expect different absolute numbers on other hardware. The patterns are what should carry over:
gains that flatten past 16, forward-pass dominance, and the effect of oversubscribing threads.
