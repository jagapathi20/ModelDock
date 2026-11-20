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
| At batch size 16, a saturated worker's worst case is ~164 ms, which **misses the 150 ms target** even with an empty queue. Batch size 8 gives ~100 ms. **Decision: batch size 8.** | Table 4 |
| The queue limit (503 threshold) is derived from the latency budget and set **per model version**: 8 requests per worker for the transformer, hundreds for sklearn. | Queue limit section |
| The sklearn model costs a flat **0.12 ms** per call whatever the batch size, so all of it is fixed per-call cost. | Run S |
| **Follow-up (2026-11-10):** batch time tracks total padded tokens, not request count, so batching helps less as inputs get longer: 10.6× per item at batch 32 for 10-token inputs, 1.8× at 128 tokens. | Runs D–H, table 5 |
| **Follow-up:** with the worker pulling a batch only when compute is free (no collector thread), the saturated worst case is 2T. At 48-token inputs batch 8 misses 150 ms (2T + 30 = 159 ms) and batch 4 fits (108 ms). **Decisions: `max_length = 48`, `max_batch_size = 4`, queue limit 8 per worker.** | Run I, table 6 |

The first six rows describe the 2026-11-06 runs on ~26-token inputs. Where the follow-up rows
disagree with them, the follow-up rows are current.

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
| D | `scripts/bench_batching_1.py` | DistilBERT, 10 tokens | 4 | 300 | `stage1-transformer_classifier-20261110T093840Z.json` |
| E | `scripts/bench_batching_1.py` | DistilBERT, 98 tokens | 4 | 300 | `stage1-transformer_classifier-20261110T094516Z.json` |
| F | `scripts/bench_batching_1.py` | DistilBERT, 128 tokens | 4 | 300 | `stage1-transformer_classifier-20261110T095534Z.json` |
| G | `scripts/bench_batching_1.py` | DistilBERT, 48 tokens | 4 | 300 | `stage1-transformer_classifier-20261110T103413Z.json` |
| H | `scripts/bench_batching_1.py` | DistilBERT, 64 tokens | 4 | 300 | `stage1-transformer_classifier-20261110T104036Z.json` |
| I | `scripts/bench_batching_1.py` | DistilBERT, 48 tokens, batch 4 and 8 only | 4 | 1000 | `stage1-transformer_classifier-20261110T105742Z.json` |
| J | `scripts/bench_batching.py`, in a Linux container | DistilBERT, 48 tokens, batch 4 and 8 only | 4 | 1000 | `stage1-transformer_classifier-20261110T121123Z.json` |
| K | `scripts/bench_batching.py`, in a Linux container | DistilBERT, 48 tokens, batch 4 and 8 only | 8 | 1000 | `stage1-transformer_classifier-20261110T121918Z.json` |

Runs A–C and S use ~26-token inputs (24 words). Runs D–I are the input-length follow-up, described
in [the follow-up section](#follow-up-2026-11-10-input-length-and-the-worker-loop).

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

The worker has two threads (design decision D7). A **compute thread** runs one batch at a time,
and a **collector thread** builds the next batch from the Redis queue while the current one runs.
The collector holds **at most one formed batch** ahead of the running one. Stage 3 must enforce
this; otherwise more batches pile up inside the worker, where the queue limit can't see them.

T = p99 time of a full batch. The worst-case model-side wait for one request:

| Situation | What the request waits for | Formula | Batch 16 (T = 54.6) | Batch 8 (T = 33.4) |
|---|---|---|---|---|
| Light load | its batch fills or `max_wait` passes, then runs | max_wait + T | 69.6 ms | 48.4 ms |
| Busy (a batch is running, the request's batch is forming) | the running batch, then its own | 2T | 109.2 ms | 66.9 ms |
| Saturated (a batch is running, the next is already formed) | the running batch, the formed one, then its own | **3T** | **163.8 ms** | **100.3 ms** |
| Each additional full batch queued ahead in Redis | one more batch | +T | +54.6 ms | +33.4 ms |

**Batch size 16 can't meet the target under saturation, whatever the queue limit.** Even with an
empty queue, a request can wait for three batches: 164 ms at p99, and 155 ms at the median. That's
over 150 ms before any Redis, JSON or HTTP overhead. Batch size 8 gives 100 ms, leaving room for
overhead, at a cost of 18% of peak throughput (253 vs 309 items/s).

(Light load only reaches max_wait + T if the batch fills up; at light load, batches are usually
smaller and faster, so these are upper bounds.)

**Correction to the printed output of runs A–C.** The "Latency budget" block the scripts print
used an earlier version of this model. It listed "saturated" as 2T and "at backpressure threshold"
(one batch queued ahead) as 3T. It missed that, under saturation, the collector has usually
already formed the next batch, so 3T is reached even with an empty queue, and a queued batch adds a
4th T. The table above is the corrected model. The script output is kept as recorded. The script's
budget function is updated in Stage 3, together with the collector rule it depends on.

> **Superseded (2026-11-10):** the collector thread was removed, so the saturated case is 2T. See
> [the follow-up section](#follow-up-2026-11-10-input-length-and-the-worker-loop).

### Sklearn

Batch time is flat at about 0.12 ms for 1 to 32 items, so all of it is fixed per-call
overhead (Python, input validation). Strictly, batching does lower the per-item cost (15× at
16). But the cost is so small that waiting up to `max_wait` (15 ms) to fill a batch would cost
far more than it saves. So batching stays off for this model, for that reason rather than the
original one ("batching doesn't help").

### The queue limit (backpressure threshold)

Before enqueueing, the API rejects a request with 503 if the queue is already at its limit
(design decision D6). That limit should come from the latency budget: how many requests can
wait in the queue and still be served within 150 ms.

**First proposal, rejected in review:** `limit = max_batch_size × workers`. It works for the
transformer, but the sklearn model runs with `max_batch_size = 1`, so its limit would be 1 per
worker. With one worker, any burst of a few requests would get 503s, even though each one
takes 0.12 ms and the queue would clear almost instantly. The formula mixed up two different
things: **batch size** is how much work runs together, and the **queue limit** is how much waiting
is acceptable.

**Derivation from the budget.** With `d` requests ahead of it in Redis, a request in a
saturated worker waits for the running batch, the formed batch, and `ceil((d + 1) / B)` batches
from the queue, the last of which is its own (B = `max_batch_size`):

```
worst case  = T × (2 + ceil((d + 1) / B))  ≤  target − overhead

m = floor((target − overhead) / T) − 2        # batches from the queue that fit, incl. its own
max_queue_depth = m × B × workers             # reject with 503 when queue length ≥ this
                                              # m < 1 means the target is infeasible at this B
```

With multiple workers on one queue, the queue drains `workers` times faster, hence the
`× workers`. Values, with target = 150 ms and overhead = 30 ms. The overhead is an
**assumption**: Redis, JSON and HTTP costs are measured in Stage 2 and the values recomputed:

| Model | B | T (p99 of a full batch) | m | max_queue_depth per worker | Worst case (+ overhead) |
|---|---|---|---|---|---|
| Transformer | 8 | 33.4 ms | floor(120 / 33.4) − 2 = 1 | **8** | 100.3 + 30 = 130 ms |
| Transformer | 16 | 54.6 ms | floor(120 / 54.6) − 2 = 0 | infeasible | 163.8 + 30 = 194 ms |
| sklearn | 1 | 0.28 ms | floor(120 / 0.28) − 2 = 426 | **426** | ≈ 120 ms |

The sklearn figure uses model-only time. In the real server, each request also costs Redis round
trips, which raise its effective T and lower its limit. It will still be in the hundreds, not 1.

> **Superseded (2026-11-10):** with the collector removed, `− 2` becomes `− 1`, and the
> transformer values are recomputed for 48-token inputs in table 6.

**Where the limit lives (options considered):**

| Option | Verdict |
|---|---|
| A. A fixed floor: `max(B, floor) × workers` | Rejected: patches the symptom with a magic number and keeps the wrong formula. |
| **B. A per-version setting, `max_queue_depth`, defaulting to the derived value** | **Chosen.** Each version already stores its own `max_batch_size` and `max_wait_ms`; the queue limit belongs next to them. TF Serving (`max_enqueued_batches`) and Triton (`max_queue_size` in its queue policy) both configure it per model. |
| C. Keep `B × workers` and accept it | Rejected: knowingly ships needless 503s for the cheapest model. |

## Follow-up (2026-11-10): input length and the worker loop

Two things changed the conclusions above. A design review pointed out that every run so far used
~26-token inputs, while the runner allowed 128, and a batch pads to its longest input. It also
showed that the collector thread itself causes the middle T of the 3T worst case. Runs D–I measure
input length; all use 4 threads.

### Table 5: input length (batch size 8 unless stated)

| Tokens | p50 (ms) | p90 (ms) | p99 (ms) | Per-item speedup at batch 32 | Throughput at batch 8 (items/s) | Run (file) |
|---|---|---|---|---|---|---|
| 10 | 19.5 | 19.7 | 21.5 | 10.6× | 411 | D (`…20261110T093840Z.json`) |
| 26 | 31.6 | 31.8 | 33.4 | 4.6× | 253 | B (`…20261106T151131Z.json`) |
| 48 | 48.3 | 51.6 | 89.8 | 3.0× | 166 | G (`…20261110T103413Z.json`) |
| 48 (1,000 iters) | 48.0 | 50.4 | 64.5 | n/a | 167 | I (`…20261110T105742Z.json`) |
| 64 | 62.0 | 66.4 | 76.0 | 2.3× | 129 | H (`…20261110T104036Z.json`) |
| 98 | 92.5 | 95.4 | 109.2 | 1.8× | 86 | E (`…20261110T094516Z.json`) |
| 128 | 122.3 | 125.6 | 162.6 | 1.8× | 65 | F (`…20261110T095534Z.json`) |

**Batch time tracks total padded tokens, not request count.** Batches with similar totals take
similar time (p50): 32 × 10 tokens = 320 → 42.4 ms (D), 16 × 26 = 416 → 51.8 ms (B), 4 × 98 = 392
→ 51.7 ms (E). Past roughly 400 tokens per batch the CPU is fully used, and each extra token costs
about 0.12 ms (8 × 128 = 1,024 tokens → 122.3 ms, F). So short inputs leave the CPU underused and
batching fills it, while long inputs fill it alone: at 128 tokens throughput is flat at about
65 items/s from batch 4 upward (F).

**Noise at 48 tokens.** Run G's batch-8 p99 (89.8 ms) is higher than at 64 tokens (76.0 ms, H),
which can't reflect real cost. With 300 samples p99 is the 3rd-slowest batch. Run I repeated
batch 4 and 8 at 1,000 iterations; its 64.5 ms is the figure used.

**Unexplained:** at 98 tokens, batch 16 has p99 = 550.9 ms and a worse per-item cost than batch 8
(E). Possibly cache effects on larger tensors, or thermal throttling after several runs. It's not
a configuration we would deploy, so it is noted rather than investigated.

### Table 6: the budget at 48 tokens, with the worker pulling a batch only when compute is free

Without the collector thread, a saturated request waits for the running batch and then its own:
`worst case = T × (1 + ceil((d + 1) / B))`, so `m = floor((target − overhead) / T) − 1`.
Overhead is still the assumed 30 ms.

| | Batch 4 | Batch 8 |
|---|---|---|
| p50 / p99 batch time at 48 tokens (run I) | 29.2 / 38.9 ms | 48.0 / 64.5 ms |
| Saturated, empty queue: 2T + 30 | **108 ms** | 159 ms (over) |
| m = floor(120 / T) − 1 | 2 | 0 |
| `max_queue_depth` per worker = m × B | **8** | infeasible |
| Worst case at the limit (7 ahead): 3T + 30 | 147 ms | |
| Throughput, one worker (run I) | 137 items/s | 167 items/s |

64 tokens doesn't fit even at the median (62.0 ms at batch 8, run H), so 48 is the longest
measured length that works.

### Table 7: native vs Docker container (48 tokens, 1,000 iterations)

Runs J and K use the original `bench_batching.py` (the `_1` copy relies on macOS `sysctl`;
measurement code is identical) inside `python:3.12-slim` on Docker Desktop, 8 vCPUs. The container
is native ARM (`uname -m` → `aarch64`), not emulated.

| | Native, 4 threads (I) | Container, 4 threads (J) | Container, 8 threads (K) |
|---|---|---|---|
| Batch 4: p50 / p99 | 29.2 / 38.9 ms | 80.8 / 159.3 ms | 77.4 / 151.4 ms |
| Batch 8: p50 / p99 | 48.0 / 64.5 ms | 141.0 / 149.4 ms | 131.6 / 223.9 ms |
| Batch 4 throughput | 137 items/s | 49 items/s | 52 items/s |

The container is about 2.8× slower, and even a batch of 4 has p99 over 150 ms. Likely cause, not
separated by experiment: macOS PyTorch's matrix math most likely goes through Apple's Accelerate
library, which can use M1 matrix hardware a Linux guest can't reach, and the VM hides the P/E
core split. So all later performance numbers are measured natively (decision 6 below).

## Decisions

### Current decisions (2026-11-10)

These replace decisions 2, 3, 4 and 7 below. Full entries: `docs/DESIGN.md`, C1.11 to C1.14.

1. **Worker loop: no collector thread.** The worker pulls a batch from Redis only when compute is
   free, so every waiting request stays where `LLEN` can see it. Saturated worst case: 2T.
   (C1.11; reasoning, applied to T from runs B and I.)
2. **`max_length = 48` tokens, per model version and immutable.** The latency target is stated for
   inputs up to that length. (C1.13; runs B, D–I.)
3. **Main model `max_batch_size = 4`.** Batch 8 misses 150 ms at 48 tokens. Cost: 137 vs
   167 items/s. Revisited once Stage 2 measures overhead. (C1.14; run I.)
4. **`max_queue_depth` = 8 per worker** for the transformer (m = 2). (C1.14; run I.)
5. **Latency target as two guarantees:** p99 < 150 ms at ≤ 70% load, and accepted requests under
   overload still finish within 150 ms. (C1.12.)
6. **Benchmarks run natively** (API and worker with `uv run`, Redis in Docker), with open-loop
   load. Containers are built but are not where numbers come from. (C1.15; runs I, J, K.)

### Original decisions (2026-11-07)

Status: **confirmed 2026-11-07. Decisions 2, 3, 4 and 7 are superseded by the list above.**

1. **Inference threads = number of performance cores** (4 on this M1). On machines where all
   cores are the same (typical Linux servers), use the physical core count. The Stage 11 layout
   comparison becomes 1×4, 2×2 and 4×1 on this machine, plus one 8-thread layout to show the
   effect of the efficiency cores under load.
2. **Main model `max_batch_size`: 16 → 8.** At 16, a saturated worker can't meet 150 ms even with
   an empty queue (3T = 164 ms). At 8 it's 100 ms. Trade-off: −18% peak throughput per worker.
   Stage 11 re-measures this end to end.
3. **Queue limit: a new per-version setting, `max_queue_depth`** (option B above), defaulting to
   `m × max_batch_size × workers` from the budget derivation. This is a new column on
   `model_versions`, added in Stage 5; until then it comes from config. Current values: 8 per worker
   (transformer), recomputed for both models once Stage 2 measures overhead.
4. **"Target load" = at most 70% of a worker's measured throughput.** The p99 < 150 ms target
   applies there. Saturation is overload, where 503s take over.
5. **`max_wait` stays at 15 ms** until the Stage 11 sweep.
6. **sklearn stays at `max_batch_size = 1`**, because waiting up to 15 ms to fill a batch costs far
   more than its 0.12 ms inference, not because batching has no effect.
7. **Stage 3 requirement: the collector holds at most one formed batch ahead of the running
   one.** The latency model and the queue limit both depend on this.

## Reproduce

```bash
uv sync --extra worker
uv run python -m scripts.fetch_sentiment_model
uv run python -m scripts.train_iris

uv run python -m scripts.bench_batching   --model-dir artifacts/sentiment-v1                       # run A
uv run python -m scripts.bench_batching   --kind sklearn_classifier --model-dir artifacts/iris-v1  # run S
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 300  # run B
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 8 --iters 300  # run C

# follow-up runs, 4 threads
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 300 --words 8    # run D
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 300 --words 96   # run E
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 300 --words 140  # run F
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 300 --words 46   # run G
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 300 --words 62   # run H
uv run python -m scripts.bench_batching_1 --model-dir artifacts/sentiment-v1 --threads 4 --iters 1000 --words 46 --batch-sizes 4,8  # run I

# runs J and K: same measurement inside a Linux container (keeps the Mac's .venv and uv.lock untouched)
docker run --rm -it -v "$PWD":/app -w /app -e UV_PROJECT_ENVIRONMENT=/tmp/venv python:3.12-slim \
  sh -c "pip install -q uv && uv sync --extra worker --frozen && \
    uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 --threads 4 --iters 1000 --words 46 --batch-sizes 4,8 && \
    uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 --threads 8 --iters 1000 --words 46 --batch-sizes 4,8"
```

Expect different absolute numbers on other hardware. The patterns are what should carry over:
gains that flatten past 16, forward-pass dominance, and the effect of oversubscribing threads.
