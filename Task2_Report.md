# Task 2: Implementation and Comparative Analysis (SPMiner)

## 1. Experiment Setup
- Project: `neural-subgraph-matcher-miner`
- Mining method: SPMiner (with pre-trained model checkpoints in `ckpt/`)
- Dataset: SNAP Facebook social network (`facebook_combined.txt`)
- Environment: Linux, Python via `uv`
- Analysis script: `graph.py`
- Plot output directory: `metrics_plots/`

### Search Strategies Evaluated
- `Greedy`
- `MCTS`
- `Beam`

### Hyperparameter Sweeps Evaluated
- Neighborhood range: `(min_hops, max_hops)` with two runs per config
  - `(2,3), (2,5), (3,5), (5,8)`
- Number of neighborhoods (`n_neighbors`): `50, 100, 200, 300, 500, 1000`
- Number of trials (`n_trials`): `50, 100, 200, 300, 400, 500`

Note: for neighborhood range experiments, pairs in `neighborhood_range_patterns` and `neighborhood_range_runtimes` are interpreted as `(try1, try2)` and averaged for reporting.

---

## 2. Results

### A) Search Strategy vs Runtime
| Strategy | Runtime (s) |
|---|---:|
| Greedy | 77 |
| MCTS | 39 |
| Beam | 300 |

Observation: `MCTS` is the fastest strategy by a large margin.

### B) Configuration Tuning vs Number of Patterns Found

#### Neighborhood range (2 tries each, averaged)
| Range | Patterns Try1 | Patterns Try2 | Patterns Avg | Runtime Avg (s) | Patterns/sec Avg |
|---|---:|---:|---:|---:|---:|
| (2, 3) | 82 | 85 | 83.50 | 34.00 | 2.456 |
| (2, 5) | 77 | 80 | 78.50 | 35.50 | 2.211 |
| (3, 5) | 81 | 92 | 86.50 | 35.00 | 2.471 |
| (5, 8) | 78 | 84 | 81.00 | 32.50 | 2.492 |

#### `n_neighbors` sweep
| n_neighbors | Patterns | Runtime (s) | Patterns/sec |
|---:|---:|---:|---:|
| 50 | 90 | 36 | 2.500 |
| 100 | 82 | 28 | 2.930 |
| 200 | 77 | 31 | 2.480 |
| 300 | 90 | 31 | 2.900 |
| 500 | 83 | 32 | 2.590 |
| 1000 | 116 | 63 | 1.840 |

#### `n_trials` sweep
| n_trials | Patterns | Runtime (s) | Patterns/sec |
|---:|---:|---:|---:|
| 50 | 40 | 22 | 1.810 |
| 100 | 78 | 28 | 2.780 |
| 200 | 155 | 51 | 3.030 |
| 300 | 237 | 77 | 3.070 |
| 400 | 280 | 112 | 2.500 |
| 500 | 348 | 134 | 2.590 |

---

## 3. Visualizations Produced
The following plots were generated with Matplotlib:
- `metrics_plots/search_strategy_vs_runtime.png`
- `metrics_plots/config_tuning_vs_patterns_found.png`
- `metrics_plots/patterns_per_sec_comparison.png`
- `metrics_plots/config_sigmoid_vs_patterns.png`

`config_tuning_vs_patterns_found.png` includes:
- Separate lines for neighborhood range min-hops and max-hops
- Continuous point indexing with labeled markers and key

`config_sigmoid_vs_patterns.png` additionally normalizes config values to `[0,1]` (sigmoid) for cross-config comparability.

---

## 4. Best Config and Best Algorithm

### Best Config (observed)
- Highest pattern count: `n_trials = 500` with **348 patterns**
- Best throughput among configs: `n_trials = 300` with **3.070 patterns/sec**
- Best neighborhood range by patterns (avg): `(3,5)` with **86.50**
- Best neighborhood range by throughput (avg): `(5,8)` with **2.492 patterns/sec**
- Best `n_neighbors` by patterns: `1000` with **116**
- Best `n_neighbors` by throughput: `100` with **2.930 patterns/sec**

### Best Algorithm (observed)
- Fastest strategy: **MCTS** (`39 s`)
- Best strategy throughput proxy: **MCTS** (`2.000 patterns/sec`)

Important note: per-strategy pattern counts were not directly logged, so strategy throughput was derived with a fixed baseline of 78 patterns for comparability.

---

## 5. Discussion and Conclusion
- `MCTS` clearly outperformed `Greedy` and `Beam` in runtime.
- Increasing `n_trials` improves total patterns found, but with diminishing throughput after around `n_trials=300`.
- For neighborhood range, `(3,5)` gives the highest average pattern count, while `(5,8)` gives the best average efficiency.
- For `n_neighbors`, very large values (`1000`) increase total patterns but reduce patterns/sec, indicating a quality-speed tradeoff.

Final recommendation:
- **Best Algorithm**: `MCTS`
- **Best Config for max patterns**: `n_trials=500`
- **Best Config for efficiency**: `n_trials=300`

This tradeoff-based selection is suitable for practical deployment: use `n_trials=300` for faster runs, and `n_trials=500` when maximizing discovered patterns is the top priority.
