import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def _ensure_plots_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


COLORS = {
    'custom_mm': '#E74C3C',
    'cublas': '#27AE60',
    'custom_mv': '#9B59B6',
    'sparse': '#E67E22',
    'dense_gemm': '#9B59B6',
    'dense_gemv': '#2ECC71',
    'cusparse': '#27AE60',
    'spmm_faster': '#27AE60',
    'spmv_faster': '#007830',
    'spmm_slower': '#E74C3C',
    'spmv_slower': '#FF99CC',
    'neutral': '#F39C12',
}

PLOT_CONFIG = {
    'dpi': 200,
    'tight_layout': True,
    'grid_alpha': 0.3,
    'annotation_fontsize': 8,
    'title_fontsize': 14,
    'label_fontsize': 12,
    'legend_fontsize': 9,
}

def extract_times_from_results(results: Dict[str, Dict], key: str, sizes: List) -> List[float]:
    return [results[key].get(size, 0) for size in sizes]


def filter_common_sizes(*result_dicts: Dict) -> List[str]:
    if not result_dicts:
        return []

    common = set(result_dicts[0].keys())
    for d in result_dicts[1:]:
        common &= set(d.keys())

    return sorted(common, key=lambda x: tuple(int(d) for d in x.split('x')) if 'x' in x else int(x))


def calculate_speedups(baseline_times: List[float], compare_times: List[float]) -> List[float]:
    return [b/c if c > 0 else 0 for b, c in zip(baseline_times, compare_times)]


def setup_comparison_plot(n_cols: int = 2, figsize: Tuple[int, int] = (14, 5)) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    return fig, axes


def configure_bar_plot(ax, x_positions: List[float], values: List[float],
                      labels: List[str], xlabel: str, ylabel: str, title: str,
                      color: str = None, log_scale: bool = False) -> None:
    ax.bar(x_positions, values, color=color, edgecolor='white', width=0.6)
    ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['label_fontsize'])
    ax.set_title(title, fontsize=PLOT_CONFIG['title_fontsize']-2, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])

    if log_scale:
        ax.set_yscale('log')


def add_value_annotations(ax, bars, values: List[float], precision: int = 2) -> None:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.{precision}f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center',
                   va='bottom', fontsize=PLOT_CONFIG['annotation_fontsize'])


def add_speedup_annotations(ax, x_positions: List[float], speedups: List[float]) -> None:
    for x, spd in zip(x_positions, speedups):
        ax.annotate(f'{spd:.2f}x', xy=(x, spd), xytext=(0, 3),
                    textcoords="offset points", ha='center', fontsize=10, fontweight='bold')


def configure_speedup_subplot(ax, x_positions: List[float], labels: List[str],
                             xlabel: str, ylabel: str, title: str,
                             speedups: List[float], target_line: float = 1.0) -> None:
    colors = [COLORS['spmm_faster'] if s >= 1.0 else COLORS['spmm_slower'] for s in speedups]
    bars = ax.bar(x_positions, speedups, color=colors, edgecolor='white', width=0.6)

    if target_line > 0:
        ax.axhline(y=target_line, color='black', linestyle='--', linewidth=1.5,
                  label=f'Parity ({target_line:.1f}x)')

    ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['label_fontsize'])
    ax.set_title(title, fontsize=PLOT_CONFIG['title_fontsize']-2, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])

    import matplotlib.patches as mpatches
    faster_patch = mpatches.Patch(color=COLORS['spmm_faster'], label='≥1.0x (Faster)')
    slower_patch = mpatches.Patch(color=COLORS['spmm_slower'], label='<1.0x (Slower)')
    ax.legend(handles=[faster_patch, slower_patch], loc='upper right', fontsize=PLOT_CONFIG['legend_fontsize'])

    add_speedup_annotations(ax, x_positions, speedups)


def save_plot(fig: plt.Figure, output_path: Path, title: str = None) -> None:
    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], y=1.02, fontweight='bold')

    _ensure_plots_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    #print(f"Saved: {output_path}")



def _load_benchmark_data(json_path: Path) -> List[Dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    return content.get("benchmarks", [])


def _get_summary_value(summaries: List[Dict], tag: str) -> Optional[float]:
    for summ in summaries:
        if summ.get('tag') == tag:
            data = summ.get('data', [])
            for d in data:
                if d.get('name') == 'value':
                    try:
                        return float(d.get('value', 0))
                    except (ValueError, TypeError):
                        return None
    return None


def _parse_nvbench_json(json_path: Path) -> Dict[str, Dict[str, float]]:
    results = {"GEMM_Custom": {}, "GEMM_cuBLAS": {}, "GEMV_Custom": {}, "GEMV_cuBLAS": {}}

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])

    for bench in benchmarks:
        bench_name = bench.get("name", "")

        bench_type = None
        for bt in results.keys():
            if bench_name == bt:
                bench_type = bt
                break

        if bench_type is None:
            continue

        for state in bench.get("states", []):
            if state.get("is_skipped", False):
                continue

            axis_values = state.get("axis_values", [])
            dims = {}
            for av in axis_values:
                name = av.get("name", "")
                try:
                    value = int(av.get("value", 0))
                    dims[name] = value
                except (ValueError, TypeError):
                    continue

            if not dims:
                continue

            m = dims.get("m", 0)
            n = dims.get("n", 0)
            k = dims.get("k", 0)

            if "GEMM" in bench_type:
                size_label = f"{m}x{n}x{k}"
            else:
                size_label = f"{m}x{n}"

            summaries = state.get("summaries", [])

            gpu_time = _get_summary_value(summaries, "nv/cold/time/gpu/mean")
            if gpu_time is None:
                gpu_time = _get_summary_value(summaries, "nv/cold/time/gpu/min")

            if gpu_time is not None and gpu_time > 0:
                gpu_time_ms = gpu_time * 1000
                results[bench_type][size_label] = gpu_time_ms

    return results


def plot_gpu_gemm_comparison(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    ours = results.get("GEMM_Custom", {})
    cublas = results.get("GEMM_cuBLAS", {})

    common_sizes = filter_common_sizes(ours, cublas)
    if not common_sizes:
        #print("No common GPU GEMM sizes found for comparison")
        return

    square_sizes = [s for s in common_sizes if len(set(s.split('x'))) == 1]
    plot_sizes = square_sizes if len(square_sizes) >= 3 else common_sizes[:12]

    ours_times = extract_times_from_results(results, "GEMM_Custom", plot_sizes)
    cublas_times = extract_times_from_results(results, "GEMM_cuBLAS", plot_sizes)
    speedups = calculate_speedups(cublas_times, ours_times)

    fig, (ax1, ax2) = setup_comparison_plot(2)

    x = np.arange(len(plot_sizes))
    width = 0.35
    labels = [s.split('x')[0] for s in plot_sizes]

    ax1.bar(x - width/2, ours_times, width, label="Custom MM", color=COLORS['custom_mm'], edgecolor='white')
    ax1.bar(x + width/2, cublas_times, width, label="cuBLAS", color=COLORS['cublas'], edgecolor='white')

    ax1.set_xlabel("Matrix Size (M=N=K)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.set_title("Execution Time Comparison", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(loc='upper left', fontsize=PLOT_CONFIG['legend_fontsize'])
    ax1.grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])

    configure_speedup_subplot(ax2, x, labels, "Matrix Size (M=N=K)", "Speedup (cuBLAS / Custom)",
                             "Speedup Ratio (>1 = We're Faster)", speedups)

    save_plot(fig, output_path, "GPU GEMM: Custom MM Implementation vs cuBLAS")


def plot_gpu_gemv_comparison(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    ours = results.get("GEMV_Custom", {})
    cublas = results.get("GEMV_cuBLAS", {})

    common_sizes = filter_common_sizes(ours, cublas)
    if not common_sizes:
        #print("No common GPU GEMV sizes found for comparison")
        return

    square_sizes = [s for s in common_sizes if len(set(s.split('x'))) == 1]
    plot_sizes = square_sizes if len(square_sizes) >= 3 else common_sizes[:12]

    ours_times = extract_times_from_results(results, "GEMV_Custom", plot_sizes)
    cublas_times = extract_times_from_results(results, "GEMV_cuBLAS", plot_sizes)
    speedups = calculate_speedups(cublas_times, ours_times)

    fig, (ax1, ax2) = setup_comparison_plot(2)

    x = np.arange(len(plot_sizes))
    width = 0.35
    labels = [s.replace('x', '×') for s in plot_sizes]

    ax1.bar(x - width/2, ours_times, width, label="Custom MV", color=COLORS['custom_mv'], edgecolor='white')
    ax1.bar(x + width/2, cublas_times, width, label="cuBLAS", color=COLORS['cublas'], edgecolor='white')

    ax1.set_xlabel("Matrix Size (MxN)", fontsize=11)
    ax1.set_ylabel("Execution Time (ms)", fontsize=11)
    ax1.set_title("Execution Time Comparison", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(loc='upper left', fontsize=PLOT_CONFIG['legend_fontsize'])
    ax1.grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])

    configure_speedup_subplot(ax2, x, labels, "Matrix Size (MxN)", "Speedup (cuBLAS / Custom)",
                             "Speedup Ratio (>1 = We're Faster)", speedups)

    save_plot(fig, output_path, "GPU GEMV: Custom MV Implementation vs cuBLAS")


def _parse_nn_nvbench_json(json_path: Path) -> Dict[str, Dict]:
    results = {
        "DenseNN_GEMM": {},
        "DenseNN_GEMV": {},
        "DenseNN_cuBLAS": {},
        "SparseNN_SpMM": {},
        "SparseNN_SpMV": {},
        "SparseNN_cuSPARSE": {},
        "SparseNN_SpMM_Sparsity": {},
        "SparseNN_SpMV_Sparsity": {},
    }

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])

    name_mapping = {
        "GPU_DenseNN_GEMM": "DenseNN_GEMM",
        "GPU_DenseNN_GEMV": "DenseNN_GEMV",
        "GPU_DenseNN_cuBLAS": "DenseNN_cuBLAS",
        "GPU_SparseNN_SpMM": "SparseNN_SpMM",
        "GPU_SparseNN_SpMV": "SparseNN_SpMV",
        "GPU_SparseNN_cuSPARSE": "SparseNN_cuSPARSE",
        "GPU_SparseNN_SpMM_Sparsity": "SparseNN_SpMM_Sparsity",
        "GPU_SparseNN_SpMV_Sparsity": "SparseNN_SpMV_Sparsity",
    }

    for bench in benchmarks:
        bench_name = bench.get("name", "")

        bench_type = name_mapping.get(bench_name)
        if bench_type is None:
            continue

        for state in bench.get("states", []):
            if state.get("is_skipped", False):
                continue

            axis_values = state.get("axis_values", [])
            key_value = None
            for av in axis_values:
                name = av.get("name", "")
                if name in ("batch_size", "sparsity_pct"):
                    try:
                        key_value = int(av.get("value", 0))
                    except (ValueError, TypeError):
                        continue

            if key_value is None:
                continue

            summaries = state.get("summaries", [])
            gpu_time = _get_summary_value(summaries, "nv/cold/time/gpu/mean")
            if gpu_time is None:
                gpu_time = _get_summary_value(summaries, "nv/cold/time/gpu/min")

            if gpu_time is not None and gpu_time > 0:
                gpu_time_ms = gpu_time * 1000
                results[bench_type][key_value] = gpu_time_ms

    return results


def plot_gpu_dense_nn_comparison(results: Dict[str, Dict[int, float]], output_path: Path) -> None:
    """Plot GPU Dense NN: GEMM vs GEMV with FAIR comparison (time-per-sample, throughput).
    
    This is a fair comparison because:
    - Both GEMM and GEMV process the same number of samples
    - GEMM: 1 batched call for all samples
    - GEMV: Internally processes per-sample (N separate operations)
    - Normalizing to time-per-sample shows true efficiency
    """
    gemm = results.get("DenseNN_GEMM", {})
    gemv = results.get("DenseNN_GEMV", {})
    cublas = results.get("DenseNN_cuBLAS", {})
    
    common_batches = sorted(set(gemm.keys()) & set(gemv.keys()))
    
    if not common_batches:
        #print("No common batch sizes found for GPU Dense NN comparison")
        return
    
    # Calculate metrics (times are in ms from parsing)
    gemm_times = [gemm.get(b, 0) for b in common_batches]
    gemv_times = [gemv.get(b, 0) for b in common_batches]
    cublas_times = [cublas.get(b, 0) for b in common_batches] if cublas else [0] * len(common_batches)
    
    batch_labels = [str(b) for b in common_batches]
    
    # Create single panel plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(common_batches))
    width = 0.35

    gemm_color = "#6A5ACD"  # purple
    gemv_color = "#2E8B57"  # green

    # Raw total time
    bars1 = ax.bar(x - width/2, gemm_times, width, label="Custom GEMM", color=gemm_color)
    bars2 = ax.bar(x + width/2, gemv_times, width, label="Custom GEMV", color=gemv_color)

    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels)
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Total Time (ms)", fontsize=11)
    ax.set_title("Raw Total Time\n(Both process same # samples)",
                      fontsize=11, fontweight='bold')
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    fig.suptitle("GPU Dense NN: GEMM vs GEMV - Total Time Comparison",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    _ensure_plots_dir(output_path.parent)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print(f"Generated: {output_path}")


def plot_sparse_vs_dense_cublas(results: Dict[str, Dict[int, float]], output_path: Path) -> None:
    """Plot Sparse Matrix-Matrix Operations vs Dense cuBLAS (Bonus Task comparison)."""
    spmm = results.get("SparseNN_SpMM", {})
    cublas = results.get("DenseNN_cuBLAS", {})

    common_batches = sorted(set(spmm.keys()) & set(cublas.keys()))

    if not common_batches:
        #print("No common batch sizes found for Sparse vs cuBLAS comparison")
        return

    spmm_times = extract_times_from_results(results, "SparseNN_SpMM", common_batches)
    cublas_times = extract_times_from_results(results, "DenseNN_cuBLAS", common_batches)

    spmm_speedups = calculate_speedups(cublas_times, spmm_times)

    fig, axes = setup_comparison_plot(2)

    x = np.arange(len(common_batches))
    width = 0.35
    batch_labels = [str(b) for b in common_batches]

    bars1 = axes[0].bar(x - width/2, spmm_times, width, label="Custom SpMM", color=COLORS['sparse'])
    bars2 = axes[0].bar(x + width/2, cublas_times, width, label="Dense cuBLAS", color=COLORS['cublas'])

    axes[0].set_xlabel("Batch Size", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[0].set_ylabel("Execution Time (ms)", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[0].set_title("Sparse Matrix-Matrix vs Dense cuBLAS: Execution Time",
                      fontsize=PLOT_CONFIG['title_fontsize'] - 1, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(batch_labels, fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])
    axes[0].set_yscale('log')
    axes[0].legend()

    add_value_annotations(axes[0], bars1, spmm_times)
    add_value_annotations(axes[0], bars2, cublas_times)

    x_speedup = np.arange(len(common_batches))
    width = 0.6

    spmm_colors = [COLORS['spmm_faster'] if s >= 1.1 else COLORS['neutral'] if s >= 1.0 else COLORS['spmm_slower'] for s in spmm_speedups]
    bars_spmm = axes[1].bar(x_speedup, spmm_speedups, width, label="SpMM Speedup", color=spmm_colors)

    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Parity (1.0x)')
    axes[1].axhline(y=1.1, color='green', linestyle=':', linewidth=2, label='Target (1.1x)')
    axes[1].set_xlabel("Batch Size", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[1].set_ylabel("Speedup (cuBLAS time / SpMM time)", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[1].set_title("SpMM Speedup over Dense cuBLAS (Target: 1.1x)",
                      fontsize=PLOT_CONFIG['title_fontsize'] - 1, fontweight='bold')
    axes[1].set_xticks(x_speedup)
    axes[1].set_xticklabels(batch_labels, fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])

    import matplotlib.patches as mpatches
    spmm_faster_patch = mpatches.Patch(color=COLORS['spmm_faster'], label='≥1.1x (Faster)')
    spmm_neutral_patch = mpatches.Patch(color=COLORS['neutral'], label='1.0-1.1x')
    spmm_slower_patch = mpatches.Patch(color=COLORS['spmm_slower'], label='<1.0x (Slower)')
    axes[1].legend(handles=[spmm_faster_patch, spmm_slower_patch],
                   loc='upper right', fontsize=PLOT_CONFIG['legend_fontsize'])

    add_speedup_annotations(axes[1], x_speedup, spmm_speedups)

    save_plot(fig, output_path, "Bonus Task: Sparse Operations vs Dense cuBLAS (Target: 1.1x faster)")


def plot_sparse_vs_cusparse(results: Dict[str, Dict[int, float]], output_path: Path) -> None:
    """Plot Custom Sparse Matrix-Matrix vs cuSPARSE."""
    spmm = results.get("SparseNN_SpMM", {})
    cusparse = results.get("SparseNN_cuSPARSE", {})

    common_batches = sorted(set(spmm.keys()) & set(cusparse.keys()))

    if not common_batches:
        #print("No common batch sizes found for Custom Sparse vs cuSPARSE comparison")
        return

    spmm_times = extract_times_from_results(results, "SparseNN_SpMM", common_batches)
    cusparse_times = extract_times_from_results(results, "SparseNN_cuSPARSE", common_batches)

    spmm_speedups = calculate_speedups(cusparse_times, spmm_times)

    fig, axes = setup_comparison_plot(2)

    x = np.arange(len(common_batches))
    width = 0.35
    batch_labels = [str(b) for b in common_batches]

    bars1 = axes[0].bar(x - width/2, spmm_times, width, label="Custom SpMM", color=COLORS['sparse'])
    bars2 = axes[0].bar(x + width/2, cusparse_times, width, label="cuSPARSE", color=COLORS['cusparse'])

    axes[0].set_xlabel("Batch Size", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[0].set_ylabel("Execution Time (ms)", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[0].set_title("Custom Sparse Matrix-Matrix vs cuSPARSE: Execution Time",
                      fontsize=PLOT_CONFIG['title_fontsize'] - 1, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(batch_labels, fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])
    axes[0].set_yscale('log')
    axes[0].legend()

    add_value_annotations(axes[0], bars1, spmm_times)
    add_value_annotations(axes[0], bars2, cusparse_times)

    x_speedup = np.arange(len(common_batches))
    width = 0.6

    spmm_colors = [COLORS['spmm_faster'] if s >= 1.0 else COLORS['spmm_slower'] for s in spmm_speedups]
    bars_spmm = axes[1].bar(x_speedup, spmm_speedups, width, label="SpMM Speedup", color=spmm_colors)

    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Parity (1.0x)')
    axes[1].set_xlabel("Batch Size", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[1].set_ylabel("Speedup (cuSPARSE time / SpMM time)", fontsize=PLOT_CONFIG['label_fontsize'])
    axes[1].set_title("SpMM Speedup over cuSPARSE",
                      fontsize=PLOT_CONFIG['title_fontsize'] - 1, fontweight='bold')
    axes[1].set_xticks(x_speedup)
    axes[1].set_xticklabels(batch_labels, fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])

    import matplotlib.patches as mpatches
    spmm_faster_patch = mpatches.Patch(color=COLORS['spmm_faster'], label='≥1.0x (Faster)')
    spmm_slower_patch = mpatches.Patch(color=COLORS['spmm_slower'], label='<1.0x (Slower)')
    axes[1].legend(handles=[spmm_faster_patch, spmm_slower_patch],
                   loc='upper right', fontsize=PLOT_CONFIG['legend_fontsize'])

    add_speedup_annotations(axes[1], x_speedup, spmm_speedups)

    save_plot(fig, output_path, "Custom Sparse Operations (SpMM & SpMV) vs cuSPARSE")


def _collect_nn_median_metrics(benchmarks: Iterable[Dict]) -> List[Dict]:
    results = []
    for bench in benchmarks:
        if bench.get("run_type") != "aggregate" or bench.get("aggregate_name") != "median":
            continue

        name = bench.get("name", "")
        if not (name.startswith("BM_DENSENN_GEMM") or name.startswith("BM_DENSENN_GEMV")):
            continue

        base_name = name.rsplit("_", 1)[0]
        parts = base_name.split("/")
        iter_idx = next((idx for idx, part in enumerate(parts) if part.startswith("iterations:")), None)

        if iter_idx is None or iter_idx < 1:
            continue

        batch_size = int(parts[iter_idx - 1])
        kind = "GEMM" if "GEMM" in parts[0] else "GEMV"

        results.append({
            "kind": kind,
            "batch": batch_size,
            "real_time": bench.get("real_time", 0.0),
            "accuracy": bench.get("Accuracy", None),
        })

    return results


def plot_dense_nn_benchmarks(json_path: str) -> None:

    json_file = Path(json_path)
    if not json_file.exists():
        return

    benches = _load_benchmark_data(json_file)
    nn_results = _collect_nn_median_metrics(benches)

    if not nn_results:
        return

    # Group by batch size for comparison
    gemm_by_batch = {e["batch"]: e for e in nn_results if e["kind"] == "GEMM"}
    gemv_by_batch = {e["batch"]: e for e in nn_results if e["kind"] == "GEMV"}
    
    common_batches = sorted(set(gemm_by_batch.keys()) & set(gemv_by_batch.keys()))
    
    if not common_batches:
        # Fallback to old behavior if no common batches
        nn_results.sort(key=lambda e: (0 if e["kind"] == "GEMV" else 1, e["batch"]))
        labels = [f"{e['kind']}\nB={e['batch']}" for e in nn_results]
        times  = [e["real_time"] for e in nn_results]
        accs   = [e["accuracy"] for e in nn_results]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        width = 0.6
        colors = ["#2E8B57" if "GEMV" in lbl else "#6A5ACD" for lbl in labels]
        bars = ax.bar(x, times, width, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Median real time (µs)")
        ax.set_title("Dense NN Inference: GEMM vs GEMV")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        _ensure_plots_dir(Path("plots"))
        fig.savefig("plots/dense_nn_gemm_vs_gemv.png", dpi=200)
        plt.close(fig)
        return
    
    # Calculate metrics for fair comparison
    gemm_total_times = []
    gemv_total_times = []
    gemm_per_sample = []
    gemv_per_sample = []
    gemm_throughput = []  
    gemv_throughput = []
    batch_labels = []
    
    for batch in common_batches:
        gemm = gemm_by_batch[batch]
        gemv = gemv_by_batch[batch]
        
        gemm_total_times.append(gemm["real_time"])
        gemv_total_times.append(gemv["real_time"])
        
        batch_labels.append(str(batch))
    
    # Create single panel plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(common_batches))
    width = 0.35

    gemm_color = "#6A5ACD"  # purple
    gemv_color = "#2E8B57"  # green

    # Raw total time
    bars1 = ax.bar(x - width/2, gemm_total_times, width, label="GEMM (batched)", color=gemm_color)
    bars2 = ax.bar(x + width/2, gemv_total_times, width, label="GEMV (per-sample)", color=gemv_color)

    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels)
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Total Time (µs)", fontsize=11)
    ax.set_title("Raw Total Time\nBoth process same # samples",
                      fontsize=11, fontweight='bold')
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    fig.suptitle("CPU Dense NN Inference: GEMM vs GEMV - Total Time Comparison",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    _ensure_plots_dir(Path("plots/Task_1"))
    fig.savefig("plots/Task_1/dense_nn_gemm_vs_gemv.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    #print("Generated: plots/Task_1/dense_nn_gemm_vs_gemv.png")


def _collect_sparse_nn_metrics(benchmarks: Iterable[Dict]) -> Dict[str, List[Dict]]:
    """Collect sparse NN results grouped by implementation (SpMM vs SpMV)"""
    results = {"SpMM": [], "SpMV": []}

    for bench in benchmarks:
        if bench.get("run_type") != "aggregate" or bench.get("aggregate_name") != "median":
            continue
        name = bench.get("name", "")
        if not (name.startswith("BM_SPARSENN_SPMM") or name.startswith("BM_SPARSENN_SPMV")):
            continue

        # Parse sparsity level from benchmark name
        # Expected format: BM_SPARSENN_SPMM/sparsity/batch/iterations:...
        base_name = name.rsplit("_", 1)[0]
        parts = base_name.split("/")

        if len(parts) < 2:
            continue

        try:
            sparsity = int(parts[1])
            batch = int(parts[2]) if len(parts) > 2 else 32  # Default to 32 if not specified
            kind = "SpMM" if "SPMM" in parts[0] else "SpMV"

            results[kind].append({
                "sparsity": sparsity,
                "batch": batch,
                "real_time": bench.get("real_time", 0.0),
                "accuracy": bench.get("Accuracy", 0.0),
            })
        except (ValueError, IndexError):
            continue

    # Sort by sparsity level
    for kind in results:
        results[kind].sort(key=lambda x: x["sparsity"])

    return results


def plot_sparse_nn_accuracy(json_path: str) -> None:
    """Plot accuracy degradation as sparsity increases"""
    json_file = Path(json_path)
    if not json_file.exists():
        #print(f"Warning: {json_path} not found, skipping sparse NN accuracy plot")
        return

    benches = _load_benchmark_data(json_file)
    sparse_results = _collect_sparse_nn_metrics(benches)

    if not sparse_results["SpMM"] and not sparse_results["SpMV"]:
        #print("No sparse NN results found, skipping accuracy plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy vs Sparsity
    if sparse_results["SpMM"]:
        sparsities = [r["sparsity"] for r in sparse_results["SpMM"]]
        accuracies = [r["accuracy"] * 100 for r in sparse_results["SpMM"]]
        ax1.plot(sparsities, accuracies, marker='o', linewidth=2, markersize=8,
                color='#6A5ACD', label='SpMM')

    if sparse_results["SpMV"]:
        sparsities = [r["sparsity"] for r in sparse_results["SpMV"]]
        accuracies = [r["accuracy"] * 100 for r in sparse_results["SpMV"]]
        ax1.plot(sparsities, accuracies, marker='s', linewidth=2, markersize=8,
                color='#2E8B57', label='SpMV')

    ax1.set_xlabel("Sparsity Level (%)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Sparse NN: Accuracy vs Sparsity", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% target')

    # Plot 2: Runtime vs Sparsity
    if sparse_results["SpMM"]:
        sparsities = [r["sparsity"] for r in sparse_results["SpMM"]]
        times = [r["real_time"] / 1000 for r in sparse_results["SpMM"]]  # Convert to ms
        ax2.plot(sparsities, times, marker='o', linewidth=2, markersize=8,
                color='#6A5ACD', label='SpMM')

    if sparse_results["SpMV"]:
        sparsities = [r["sparsity"] for r in sparse_results["SpMV"]]
        times = [r["real_time"] / 1000 for r in sparse_results["SpMV"]]
        ax2.plot(sparsities, times, marker='s', linewidth=2, markersize=8,
                color='#2E8B57', label='SpMV')

    ax2.set_xlabel("Sparsity Level (%)", fontsize=11)
    ax2.set_ylabel("Median Real Time (ms)", fontsize=11)
    ax2.set_title("Sparse NN: Runtime vs Sparsity", fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    _ensure_plots_dir(Path("plots/Task_2"))
    fig.savefig("plots/Task_2/sparse_nn_accuracy_vs_sparsity.png", dpi=200)
    plt.close(fig)
    #print("Generated: plots/Task_2/sparse_nn_accuracy_vs_sparsity.png")


def plot_gpu_sparse_nn_vs_sparsity(json_path: str) -> None:
    """
    GPU sparse NN comparison: runtime + accuracy across sparsity levels.
    Displays Custom (SpMM/SpMV) vs cuSPARSE (SpMM/SpMV).
    """

    # ------------------------------- Load NVBench File -------------------------------
    bench_file = Path(json_path)
    if not bench_file.exists():
        #print(f"[GPU Plot] Missing benchmark file: {json_path}")
        return

    with bench_file.open("r", encoding="utf8") as fp:
        raw_json = json.load(fp)

    entries = raw_json.get("benchmarks", [])
    if not entries:
        #print("[GPU Plot] No entries found inside NVBench JSON.")
        return

    # ------------------------------- Storage Structure -------------------------------
    # {kernel_type: {sparsity: {"time": µs, "acc": %}}}
    kernel_results = {
        "custom_spmm": {},
        "custom_spmv": {},
        "cusparse_spmm": {},
        "cusparse_spmv": {},
    }

    # Mapping from NVBench names → our internal keys
    kernel_map = {
        "sparse_nn_spmm": "custom_spmm",
        "sparse_nn_spmv": "custom_spmv",
        "cusparse_sparse_nn_spmm": "cusparse_spmm",
        "cusparse_sparse_nn_spmv": "cusparse_spmv",
    }

    # ------------------------------- Extract Benchmark Data -------------------------------
    for bench in entries:
        name = bench.get("name", "")
        if name not in kernel_map:
            continue

        internal_name = kernel_map[name]

        # Loop over states instead of summaries directly
        for st in bench.get("states", []):
            # 1) Find the sparsity value
            sparsity_val = None
            for axis_info in st.get("axis_values", []):
                if axis_info.get("name") == "sparsity":
                    raw_val = axis_info.get("value")
                    sparsity_val = int(float(raw_val))
                    break
            if sparsity_val is None:
                continue

            # 2) Extract GPU time + accuracy
            gpu_us = None
            acc_val = None

            for summary in st.get("summaries", []):
                tag = summary.get("tag", "")

                # Runtime
                if tag == "nv/cold/time/gpu/mean":
                    for d in summary.get("data", []):
                        if d.get("name") == "value":
                            gpu_us = float(d.get("value", 0.0)) * 1e6  # seconds → µs
                            break

                # Accuracy
                elif tag == "accuracy":
                    for d in summary.get("data", []):
                        if d.get("name") == "value":
                            acc_val = float(d.get("value", 0.0))
                            break

            # Store if something was found
            if gpu_us is not None or acc_val is not None:
                kernel_results[internal_name][sparsity_val] = {
                    "time": gpu_us or 0.0,
                    "acc": acc_val or 0.0,
                }

    # ------------------------------- Determine Sparsity Levels -------------------------------
    all_levels = set()
    for d in kernel_results.values():
        all_levels |= d.keys()
    sparsity_levels = sorted(all_levels)

    if not sparsity_levels:
        #print("[GPU Plot] No sparsity data detected.")
        return

    # ------------------------------- Plot Styling -------------------------------
    fig, (ax_runtime, ax_accuracy) = plt.subplots(
        1, 2, figsize=(16, 6), sharex=True, facecolor="white"
    )

    color_map = {
        "custom_spmm": "#e11d48",      # red
        "custom_spmv": "#f97316",      # orange
        "cusparse_spmm": "#2563eb",    # blue
        "cusparse_spmv": "#0ea5e9",    # cyan
    }
    name_map = {
        "custom_spmm": "Custom SpMM",
        "custom_spmv": "Custom SpMV",
        "cusparse_spmm": "cuSPARSE SpMM",
        "cusparse_spmv": "cuSPARSE SpMV",
    }
    style_map = {
        "custom_spmm": "-",
        "custom_spmv": "--",
        "cusparse_spmm": "-",
        "cusparse_spmv": "--",
    }
    marker_map = {
        "custom_spmm": "o",
        "custom_spmv": "s",
        "cusparse_spmm": "^",
        "cusparse_spmv": "d",
    }

    # ------------------------------- Plot 1: Runtime --------------------------------
    for key, series in kernel_results.items():
        y_vals = [series.get(s, {}).get("time", 0) for s in sparsity_levels]
        if any(y_vals):
            ax_runtime.plot(
                sparsity_levels,
                y_vals,
                label=name_map[key],
                color=color_map[key],
                linestyle=style_map[key],
                marker=marker_map[key],
                linewidth=2,
                markersize=7,
            )

    ax_runtime.set_title("GPU Runtime vs Sparsity", fontsize=13, fontweight="bold")
    ax_runtime.set_ylabel("Mean GPU Time (µs)")
    ax_runtime.set_xlabel("Sparsity Level (%)")
    ax_runtime.set_yscale("log")
    ax_runtime.grid(True, linestyle="--", alpha=0.25)
    ax_runtime.legend(fontsize=9)

    # ------------------------------- Plot 2: Accuracy --------------------------------
    for key, series in kernel_results.items():
        y_acc = [series.get(s, {}).get("acc", 0) for s in sparsity_levels]
        if any(y_acc):
            ax_accuracy.plot(
                sparsity_levels,
                y_acc,
                label=name_map[key],
                color=color_map[key],
                linestyle=style_map[key],
                marker=marker_map[key],
                linewidth=2,
                markersize=7,
            )

    ax_accuracy.set_title("Accuracy vs Sparsity", fontsize=13, fontweight="bold")
    ax_accuracy.set_ylabel("Accuracy (%)")
    ax_accuracy.set_xlabel("Sparsity Level (%)")
    ax_accuracy.set_ylim(0, 100)
    ax_accuracy.grid(True, linestyle="--", alpha=0.25)
    ax_accuracy.legend(fontsize=9)

    # ------------------------------- Title + Device -------------------------------
    gpu_name = raw_json.get("devices", [{}])[0].get("name", "Unknown GPU")
    fig.suptitle(
        "GPU Sparse NN — Performance vs Accuracy",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.03,
        f"Device: {gpu_name}",
        ha="center",
        fontsize=9,
        color="#555",
        style="italic",
    )

    # ------------------------------- Save Output --------------------------------
    fig.tight_layout()
    _ensure_plots_dir()
    out_path = PLOTS_DIR / "gpu_sparse_nn_vs_sparsity.png"
    fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    #print(f"[GPU Plot] Saved → {out_path}")


def plot_cpu_kernel_comparison(json_path: str) -> None:
    """
    Compare GEMM, GEMV, SpMM, SpMV kernel performance across tile sizes.
    Uses real_time for accurate parallel performance measurement.
    """
    benchmarks = _load_benchmark_data(Path(json_path))

    if not benchmarks:
        return

    # Collect mean real_time for each kernel type
    kernel_data: Dict[str, List[Tuple[str, float]]] = {
        "BM_GEMM": [], "BM_GEMV": [], "BM_SPMM": [], "BM_SPMV": []
    }

    for bench in benchmarks:
        if bench.get("run_type") != "aggregate" or bench.get("aggregate_name") != "mean":
            continue

        name = bench.get("name", "")
        kernel = name.split("/", 1)[0]

        if kernel not in kernel_data:
            continue

        # Extract tile sizes from name (e.g., BM_GEMM/256/256/256/32/64/...)
        parts = name.split("/")
        # Find tile parameters (last two numbers before iterations:)
        iter_idx = next((i for i, p in enumerate(parts) if p.startswith("iterations:")), None)

        if iter_idx and iter_idx >= 2:
            tile_label = f"{parts[iter_idx-2]}x{parts[iter_idx-1]}"
            kernel_data[kernel].append((tile_label, bench.get("real_time", 0.0)))

    # Check if we have data
    if not any(kernel_data.values()):
        #print("No kernel benchmark data found.")
        return

    # Find kernels with data and get their tile sizes
    kernels_with_data = {k: v for k, v in kernel_data.items() if v}

    if not kernels_with_data:
        #print("No kernels with data found.")
        return

    # Collect all unique tile sizes across all kernels
    all_tiles = set()
    for kernel, data in kernels_with_data.items():
        all_tiles.update(d[0] for d in data)

    # Sort tiles by their numeric values
    all_tiles = sorted(all_tiles, key=lambda x: tuple(int(t) for t in x.split('x')))

    #print(f"All available tile sizes: {all_tiles}")

    # Create data for all tile sizes, using None for missing data points
    filtered_kernel_data = {}
    for kernel, data in kernels_with_data.items():
        # Create a dict mapping tile sizes to times
        tile_to_time = {tile: time for tile, time in data}

        # Create data points for all tiles, using None for missing ones
        filtered_data = []
        for tile in all_tiles:
            time = tile_to_time.get(tile)
            filtered_data.append((tile, time))

        filtered_kernel_data[kernel] = filtered_data


    fig, ax = plt.subplots(figsize=(12, 6))

    tile_labels = [tile for tile, _ in filtered_kernel_data[list(filtered_kernel_data.keys())[0]]]
    x = np.arange(len(tile_labels))
    width = 0.2

    colors = {"BM_GEMM": "#2563eb", "BM_GEMV": "#dc2626", "BM_SPMM": "#16a34a", "BM_SPMV": "#9333ea"}
    labels = {"BM_GEMM": "GEMM (Dense MM)", "BM_GEMV": "GEMV (Dense MV)",
              "BM_SPMM": "SpMM (Sparse MM)", "BM_SPMV": "SpMV (Sparse MV)"}

    for i, (kernel, data) in enumerate(filtered_kernel_data.items()):
        # Filter out None values and get corresponding x positions
        valid_data = [(j, time) for j, (_, time) in enumerate(data) if time is not None]
        if valid_data:
            positions = [pos for pos, _ in valid_data]
            times = [time for _, time in valid_data]
            offset = (i - 1.5) * width
            bars = ax.bar(np.array(positions) + offset, times, width, label=labels[kernel], color=colors[kernel], alpha=0.85)

    ax.set_xlabel("Tile Size")
    ax.set_ylabel("Real Time (µs) — Log Scale")
    ax.set_title("CPU Kernel Performance: Dense vs Sparse Operations\n(256×256 matrices, lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(tile_labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3, which="both")
    ax.set_axisbelow(True)

    fig.tight_layout()
    _ensure_plots_dir(Path("plots"))
    fig.savefig(Path("plots/Task_1/cpu_kernel_comparison.png"), dpi=150, facecolor="white")
    plt.close(fig)
    #print(f"Saved: plots/cpu_kernel_comparison.png")





# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    #print("=" * 60)
    #print("Generating CPU Benchmark Plots")
    #print("=" * 60)

    # CPU kernel benchmarks
    plot_dense_nn_benchmarks("./logs/nn_cpu.json")

    # Sparse NN benchmarks are also in nn_cpu.json
    plot_sparse_nn_accuracy("./logs/nn_cpu.json")
    
    # CPU kernel comparison across tile sizes
    plot_cpu_kernel_comparison("./logs/project.json")


    gpu_json = "./logs/project_gpu.json"
    if Path(gpu_json).exists():
        results = _parse_nvbench_json(Path(gpu_json))

        plot_gpu_gemm_comparison(results, Path("plots/Task_3/gpu_gemm_vs_cublas.png"))
        plot_gpu_gemv_comparison(results, Path("plots/Task_3/gpu_gemv_vs_cublas.png"))
        
        
    else:
        #print(f"GPU kernel benchmark file not found: {gpu_json}")

    nn_gpu_json = "./logs/nn_gpu.json"
    if Path(nn_gpu_json).exists():
        nn_results = _parse_nn_nvbench_json(Path(nn_gpu_json))

        # GPU Dense NN: GEMM vs GEMV FAIR comparison
        #print("\n" + "=" * 60)
        #print("Generating GPU Dense NN Fair Comparison Plot")
        #print("=" * 60)
        plot_gpu_dense_nn_comparison(nn_results, Path("plots/Task_3/gpu_dense_nn_gemm_vs_gemv.png"))

        plot_sparse_vs_dense_cublas(nn_results, Path("plots/Task_3/sparse_nn_vs_dense_cublas.png"))
        plot_sparse_vs_cusparse(nn_results, Path("plots/Task_3/sparse_nn_vs_cusparse.png"))
        plot_gpu_sparse_nn_vs_sparsity("./logs/nn_gpu.json")
    else:
        #print(f"GPU NN benchmark file not found: {nn_gpu_json}")
