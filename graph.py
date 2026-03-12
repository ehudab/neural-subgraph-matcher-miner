
"""Experiment metrics analysis and plotting.

This script creates:
1) Search Strategy vs Runtime
2) Configuration Tuning vs Number of Patterns Found
3) Unified patterns/sec comparison across n_neighbors, n_trials, strategy
4) Sigmoid-normalized config values (0-1) vs patterns

Interpretation notes used in this script:
- neighborhood_range entries are true ranges: (min_hops, max_hops)
- neighborhood_range_patterns and neighborhood_range_runtimes contain
  (first_try, second_try)
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _index_to_label(idx: int) -> str:
	# Excel-like labels: 0->A, 25->Z, 26->AA
	label = ""
	n = idx + 1
	while n > 0:
		n, rem = divmod(n - 1, 26)
		label = chr(ord("A") + rem) + label
	return label


def _sigmoid_normalize(values: list[float]) -> list[float]:
	if not values:
		return []
	mu = mean(values)
	variance = sum((v - mu) ** 2 for v in values) / len(values)
	std = variance**0.5
	if std == 0:
		return [0.5 for _ in values]
	# Logistic squashing to [0, 1] for scale-invariant comparison across config types.
	return [1.0 / (1.0 + pow(2.718281828459045, -((v - mu) / std))) for v in values]


neighborhood_range = ((2, 3), (2, 5), (3, 5), (5, 8))
neighborhood_range_patterns = ((82, 85), (77, 80), (81, 92), (78, 84))
neighborhood_range_runtimes = ((32, 36), (40, 31), (39, 31), (35, 30))

# One duplicate found on 300.
n_neighbors = (50, 100, 200, 300, 500, 1000)
n_neighbors_patterns = (90, 82, 77, 90, 83, 116)
n_neighbors_runtimes = (36, 28, 31, 31, 32, 63)
n_neighbors_patternspersec = (2.5, 2.93, 2.48, 2.90, 2.59, 1.84)

n_trials = (50, 100, 200, 300, 400, 500)
n_trials_patterns = (40, 78, 155, 237, 280, 348)
n_trials_runtimes = (22, 28, 51, 77, 112, 134)
n_trials_patternspersec = (1.81, 2.78, 3.03, 3.07, 2.50, 2.59)

strategy = ("Greedy", "MCTS", "Beam")
strategy_runtimes = (77, 39, 300)


def _fmt_row(cols: list[str], widths: list[int]) -> str:
	return " | ".join(col.ljust(w) for col, w in zip(cols, widths))


def print_neighborhood_range_table() -> list[dict[str, float]]:
	rows = []
	for r, patt_pair, rt_pair in zip(
		neighborhood_range,
		neighborhood_range_patterns,
		neighborhood_range_runtimes,
	):
		p_avg = mean(patt_pair)
		rt_avg = mean(rt_pair)
		pps = p_avg / rt_avg if rt_avg > 0 else 0.0
		rows.append(
			{
				"range": f"({r[0]}, {r[1]})",
				"patterns_try1": patt_pair[0],
				"patterns_try2": patt_pair[1],
				"runtime_try1": rt_pair[0],
				"runtime_try2": rt_pair[1],
				"patterns_avg": p_avg,
				"runtime_avg": rt_avg,
				"patterns_per_sec_avg": pps,
			}
		)

	header = [
		"Range",
		"P1",
		"P2",
		"R1",
		"R2",
		"Patterns(avg)",
		"Runtime(avg)",
		"Patterns/sec(avg)",
	]
	lines = [
		[
			row["range"],
			str(row["patterns_try1"]),
			str(row["patterns_try2"]),
			str(row["runtime_try1"]),
			str(row["runtime_try2"]),
			f"{row['patterns_avg']:.2f}",
			f"{row['runtime_avg']:.2f}",
			f"{row['patterns_per_sec_avg']:.3f}",
		]
		for row in rows
	]
	widths = [max(len(h), max(len(line[i]) for line in lines)) for i, h in enumerate(header)]

	print("\nNeighborhood Range (two tries each):")
	print(_fmt_row(header, widths))
	print("-+-".join("-" * w for w in widths))
	for line in lines:
		print(_fmt_row(line, widths))

	return rows


def print_simple_table(name: str, x_name: str, xs, patterns, runtimes, pps) -> None:
	header = [x_name, "Patterns", "Runtime", "Patterns/sec"]
	lines = [
		[str(x), str(p), str(r), f"{rate:.3f}"]
		for x, p, r, rate in zip(xs, patterns, runtimes, pps)
	]
	widths = [max(len(h), max(len(line[i]) for line in lines)) for i, h in enumerate(header)]

	print(f"\n{name}:")
	print(_fmt_row(header, widths))
	print("-+-".join("-" * w for w in widths))
	for line in lines:
		print(_fmt_row(line, widths))


def compute_strategy_patterns_per_sec(pattern_baseline: int = 78) -> list[float]:
	# Strategy-level pattern counts were not logged with runtime.
	# We derive a comparable throughput proxy with a constant baseline.
	return [pattern_baseline / rt if rt > 0 else 0.0 for rt in strategy_runtimes]


def make_plots(out_dir: Path, strategy_pps: list[float], range_rows: list[dict[str, float]]) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)

	# 1) Search Strategy vs Runtime
	fig, ax = plt.subplots(figsize=(8, 4.5))
	bars = ax.bar(strategy, strategy_runtimes, color=["#4E79A7", "#F28E2B", "#E15759"])
	ax.set_title("Search Strategy vs Runtime")
	ax.set_xlabel("Search Strategy")
	ax.set_ylabel("Runtime (seconds)")
	ax.grid(axis="y", alpha=0.3)
	for bar, value in zip(bars, strategy_runtimes):
		ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value}", ha="center", va="bottom", fontsize=9)
	fig.tight_layout()
	fig.savefig(out_dir / "search_strategy_vs_runtime.png", dpi=160)
	plt.close(fig)

	# 2) Configuration Tuning vs Number of Patterns Found (continuous x with point-key)
	fig, ax = plt.subplots(figsize=(12, 5.8))
	range_patterns_avg = [row["patterns_avg"] for row in range_rows]
	range_min_values = [r[0] for r in neighborhood_range]
	range_max_values = [r[1] for r in neighborhood_range]

	offset = 1
	x_range_min = list(range(offset, offset + len(range_patterns_avg)))
	offset += len(range_patterns_avg)
	x_range_max = list(range(offset, offset + len(range_patterns_avg)))
	offset += len(range_patterns_avg)
	x_neighbors = list(range(offset, offset + len(n_neighbors)))
	offset += len(n_neighbors)
	x_trials = list(range(offset, offset + len(n_trials)))

	all_x = x_range_min + x_range_max + x_neighbors + x_trials
	all_labels = [_index_to_label(i) for i in range(len(all_x))]
	label_by_x = {x: lbl for x, lbl in zip(all_x, all_labels)}

	ax.plot(
		x_range_min,
		range_patterns_avg,
		marker="o",
		linewidth=2,
		label="Neighborhood range min-hops",
	)
	ax.plot(
		x_range_max,
		range_patterns_avg,
		marker="d",
		linewidth=2,
		label="Neighborhood range max-hops",
	)
	ax.plot(x_neighbors, n_neighbors_patterns, marker="s", linewidth=2, label="n_neighbors")
	ax.plot(x_trials, n_trials_patterns, marker="^", linewidth=2, label="n_trials")

	for x, y in zip(x_range_min, range_patterns_avg):
		ax.annotate(label_by_x[x], (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
	for x, y in zip(x_range_max, range_patterns_avg):
		ax.annotate(label_by_x[x], (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
	for x, y in zip(x_neighbors, n_neighbors_patterns):
		ax.annotate(label_by_x[x], (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
	for x, y in zip(x_trials, n_trials_patterns):
		ax.annotate(label_by_x[x], (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

	ax.set_title("Configuration Tuning vs Number of Patterns Found")
	ax.set_xlabel("Continuous Point Index (see letter key)")
	ax.set_ylabel("Number of Patterns Found")
	ax.set_xticks(all_x)
	ax.set_xticklabels(all_labels)
	ax.grid(alpha=0.3)
	ax.legend(loc="upper left")

	range_min_span = f"{label_by_x[x_range_min[0]]}-{label_by_x[x_range_min[-1]]}"
	range_max_span = f"{label_by_x[x_range_max[0]]}-{label_by_x[x_range_max[-1]]}"
	neighbors_span = f"{label_by_x[x_neighbors[0]]}-{label_by_x[x_neighbors[-1]]}"
	trials_span = f"{label_by_x[x_trials[0]]}-{label_by_x[x_trials[-1]]}"

	point_key_text = (
		"Point Key\n"
		f"{range_min_span}: range_min={range_min_values}\n"
		f"{range_max_span}: range_max={range_max_values}\n"
		f"{neighbors_span}: n_neighbors={list(n_neighbors)}\n"
		f"{trials_span}: n_trials={list(n_trials)}"
	)
	ax.text(
		1.01,
		0.5,
		point_key_text,
		transform=ax.transAxes,
		fontsize=8,
		va="center",
		ha="left",
		bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
	)

	fig.subplots_adjust(right=0.72)
	fig.tight_layout()
	fig.savefig(out_dir / "config_tuning_vs_patterns_found.png", dpi=160)
	plt.close(fig)

	# 3) Unified patterns/sec comparison for n_neighbors, n_trials, strategy
	fig, ax = plt.subplots(figsize=(11, 5))

	x_nei = list(range(len(n_neighbors)))
	x_tri = list(range(len(n_neighbors), len(n_neighbors) + len(n_trials)))
	x_str = list(range(len(n_neighbors) + len(n_trials), len(n_neighbors) + len(n_trials) + len(strategy)))

	ax.plot(x_nei, n_neighbors_patternspersec, marker="o", linewidth=2, label="n_neighbors")
	ax.plot(x_tri, n_trials_patternspersec, marker="s", linewidth=2, label="n_trials")
	ax.plot(x_str, strategy_pps, marker="^", linewidth=2, label="strategy (derived)")

	ticks = x_nei + x_tri + x_str
	tick_labels = [f"N={x}" for x in n_neighbors] + [f"T={x}" for x in n_trials] + [f"S={x}" for x in strategy]
	ax.set_xticks(ticks)
	ax.set_xticklabels(tick_labels, rotation=45, ha="right")
	ax.set_title("Patterns per Second Comparison (n_neighbors, n_trials, strategy)")
	ax.set_xlabel("Experiment Setting")
	ax.set_ylabel("Patterns per second")
	ax.grid(alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_dir / "patterns_per_sec_comparison.png", dpi=160)
	plt.close(fig)

	# 4) Config vs Patterns with sigmoid-normalized config values in [0, 1]
	fig, ax = plt.subplots(figsize=(10, 5.2))

	range_patterns_avg = [row["patterns_avg"] for row in range_rows]
	range_min_values = [float(r[0]) for r in neighborhood_range]
	range_max_values = [float(r[1]) for r in neighborhood_range]

	x_range_min_norm = _sigmoid_normalize(range_min_values)
	x_range_max_norm = _sigmoid_normalize(range_max_values)
	x_neighbors_norm = _sigmoid_normalize([float(v) for v in n_neighbors])
	x_trials_norm = _sigmoid_normalize([float(v) for v in n_trials])

	ax.plot(
		x_range_min_norm,
		range_patterns_avg,
		marker="o",
		linewidth=2,
		label="Neighborhood range min-hops",
	)
	ax.plot(
		x_range_max_norm,
		range_patterns_avg,
		marker="d",
		linewidth=2,
		label="Neighborhood range max-hops",
	)
	ax.plot(
		x_neighbors_norm,
		n_neighbors_patterns,
		marker="s",
		linewidth=2,
		label="n_neighbors",
	)
	ax.plot(
		x_trials_norm,
		n_trials_patterns,
		marker="^",
		linewidth=2,
		label="n_trials",
	)

	ax.set_title("Config vs Patterns (Sigmoid-Normalized Config Values)")
	ax.set_xlabel("Normalized Config Value (0-1 via sigmoid)")
	ax.set_ylabel("Number of Patterns Found")
	ax.set_xlim(-0.02, 1.02)
	ax.grid(alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_dir / "config_sigmoid_vs_patterns.png", dpi=160)
	plt.close(fig)


def identify_best(range_rows: list[dict[str, float]], strategy_pps: list[float]) -> None:
	best_range_by_patterns = max(range_rows, key=lambda r: r["patterns_avg"])
	best_range_by_pps = max(range_rows, key=lambda r: r["patterns_per_sec_avg"])

	best_neighbors_by_patterns = max(
		zip(n_neighbors, n_neighbors_patterns), key=lambda t: t[1]
	)
	best_neighbors_by_pps = max(
		zip(n_neighbors, n_neighbors_patternspersec), key=lambda t: t[1]
	)

	best_trials_by_patterns = max(zip(n_trials, n_trials_patterns), key=lambda t: t[1])
	best_trials_by_pps = max(zip(n_trials, n_trials_patternspersec), key=lambda t: t[1])

	best_strategy_runtime = min(zip(strategy, strategy_runtimes), key=lambda t: t[1])
	best_strategy_pps = max(zip(strategy, strategy_pps), key=lambda t: t[1])

	print("\nBest Config (observed):")
	print(
		f"- Highest pattern count: n_trials={best_trials_by_patterns[0]} "
		f"with {best_trials_by_patterns[1]} patterns"
	)
	print(
		f"- Best throughput among configs: n_trials={best_trials_by_pps[0]} "
		f"with {best_trials_by_pps[1]:.3f} patterns/sec"
	)
	print(
		f"- Best neighborhood range by patterns(avg): {best_range_by_patterns['range']} "
		f"with {best_range_by_patterns['patterns_avg']:.2f}"
	)
	print(
		f"- Best neighborhood range by throughput(avg): {best_range_by_pps['range']} "
		f"with {best_range_by_pps['patterns_per_sec_avg']:.3f} patterns/sec"
	)
	print(
		f"- Best n_neighbors by patterns: {best_neighbors_by_patterns[0]} "
		f"with {best_neighbors_by_patterns[1]}"
	)
	print(
		f"- Best n_neighbors by patterns/sec: {best_neighbors_by_pps[0]} "
		f"with {best_neighbors_by_pps[1]:.3f}"
	)

	print("\nBest Algorithm (observed):")
	print(
		f"- Fastest runtime: {best_strategy_runtime[0]} ({best_strategy_runtime[1]} sec)"
	)
	print(
		f"- Best throughput (derived): {best_strategy_pps[0]} "
		f"({best_strategy_pps[1]:.3f} patterns/sec)"
	)
	print(
		"  Note: strategy throughput is derived using a fixed pattern baseline "
		"(78 patterns) because per-strategy pattern counts were not logged."
	)


def main() -> None:
	range_rows = print_neighborhood_range_table()
	print_simple_table(
		"n_neighbors Sweep",
		"n_neighbors",
		n_neighbors,
		n_neighbors_patterns,
		n_neighbors_runtimes,
		n_neighbors_patternspersec,
	)
	print_simple_table(
		"n_trials Sweep",
		"n_trials",
		n_trials,
		n_trials_patterns,
		n_trials_runtimes,
		n_trials_patternspersec,
	)

	strategy_pps = compute_strategy_patterns_per_sec(pattern_baseline=78)
	print_simple_table(
		"Search Strategy",
		"strategy",
		strategy,
		(78, 78, 78),
		strategy_runtimes,
		strategy_pps,
	)

	out_dir = Path("metrics_plots")
	make_plots(out_dir, strategy_pps, range_rows)
	identify_best(range_rows, strategy_pps)

	print("\nSaved plots:")
	print(f"- {out_dir / 'search_strategy_vs_runtime.png'}")
	print(f"- {out_dir / 'config_tuning_vs_patterns_found.png'}")
	print(f"- {out_dir / 'patterns_per_sec_comparison.png'}")
	print(f"- {out_dir / 'config_sigmoid_vs_patterns.png'}")


if __name__ == "__main__":
	main()