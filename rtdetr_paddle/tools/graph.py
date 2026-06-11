#!/usr/bin/env python3
"""
Generate comparison graphs from RT-DETR training results with different
positional encoding wave functions.

For each results_[timestamp] folder, generates:
  - One PDF per loss component comparing all available waves
  - One PDF per COCO eval metric comparing all available waves
  - Skips graphs that already exist on disk

Usage:
    python tools/graph.py
    python tools/graph.py --results-dir /path/to/results
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# Ensure PDFs embed TrueType fonts (required by arXiv and similar)
plt.rcParams["pdf.fonttype"] = "truetype"

# Canonical wave names and colours
WAVE_NAMES = ['sinusoid', 'triangular', 'square', 'sawtooth']
COLORS = {
    'sinusoid':   '#1f77b4',  # blue
    'triangular': '#ff7f0e',  # orange
    'square':     '#2ca02c',  # green
    'sawtooth':   '#d62728',  # red
}

# Human-readable labels for COCO eval metrics
COCO_EVAL_LABELS = {
    'AP':    'AP @ IoU=0.50:0.95',
    'AP50':  'AP @ IoU=0.50',
    'AP75':  'AP @ IoU=0.75',
    'APs':   'AP (small)',
    'APm':   'AP (medium)',
    'APl':   'AP (large)',
    'AR1':   'AR (maxDets=1)',
    'AR10':  'AR (maxDets=10)',
    'AR100': 'AR (maxDets=100)',
    'ARs':   'AR (small)',
    'ARm':   'AR (medium)',
    'ARl':   'AR (large)',
}


def load_json(path):
    """Load a JSON file, return list of floats or empty list on failure."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def draw_comparison(data_per_wave, ylabel, title, save_path, xlabel='Epoch'):
    """
    Draw a single comparison graph with one line per wave.

    Args:
        data_per_wave: dict {wave_name: [values]}
        ylabel: Y-axis label
        title: Plot title
        save_path: Output PDF path
        xlabel: X-axis label
    """
    plt.figure(figsize=(10, 6))
    has_data = False

    for wave_name in WAVE_NAMES:
        if wave_name not in data_per_wave:
            continue
        values = data_per_wave[wave_name]
        if not values:
            continue
        # Ensure values is a list and contains only numbers
        if isinstance(values, dict):
            continue
        try:
            values = list(values)
            values = [float(v) for v in values]
        except (TypeError, ValueError):
            continue
        has_data = True
        epochs = list(range(1, len(values) + 1))
        color = COLORS.get(wave_name, '#333333')
        plt.plot(epochs, values, color=color, linewidth=2,
                 label=wave_name.capitalize())

    if not has_data:
        plt.close()
        return

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def process_results_dir(results_dir):
    """
    Process a single results_[timestamp] directory.
    Discovers available waves, collects JSON data, and generates comparison PDFs.
    """
    results_dir = Path(results_dir)
    graphs_dir = results_dir / 'graphs'

    # Discover wave subdirectories
    available_waves = []
    for name in WAVE_NAMES:
        wave_dir = results_dir / name
        if wave_dir.is_dir():
            available_waves.append(name)

    # Also check for any wave dirs not in canonical list
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and d.name not in available_waves and d.name != 'graphs':
            # Check if it has loss/ or eval/ subdirs (i.e. is a wave dir)
            if (d / 'loss').is_dir() or (d / 'eval').is_dir():
                available_waves.append(d.name)

    if not available_waves:
        print(f"  No wave directories found in {results_dir}")
        return

    print(f"  Waves found: {available_waves}")

    # --- Collect all loss metric names ---
    all_loss_keys = set()
    for wave_name in available_waves:
        loss_dir = results_dir / wave_name / 'loss'
        if loss_dir.is_dir():
            for f in loss_dir.iterdir():
                if f.suffix == '.json':
                    all_loss_keys.add(f.stem)

    # --- Collect all eval metric names ---
    all_eval_keys = set()
    for wave_name in available_waves:
        eval_dir = results_dir / wave_name / 'eval'
        if eval_dir.is_dir():
            for f in eval_dir.iterdir():
                if f.suffix == '.json':
                    all_eval_keys.add(f.stem)

    # --- Generate loss graphs ---
    if all_loss_keys:
        loss_graphs_dir = graphs_dir / 'loss'
        os.makedirs(loss_graphs_dir, exist_ok=True)

        for loss_key in sorted(all_loss_keys):
            pdf_path = loss_graphs_dir / f'{loss_key}.pdf'
            if pdf_path.exists():
                print(f"  [skip] {pdf_path} already exists")
                continue

            data_per_wave = {}
            for wave_name in available_waves:
                json_path = results_dir / wave_name / 'loss' / f'{loss_key}.json'
                if json_path.exists():
                    data_per_wave[wave_name] = load_json(str(json_path))

            draw_comparison(
                data_per_wave,
                ylabel='Loss',
                title=f'{loss_key} — All Waves',
                save_path=pdf_path,
            )

    # --- Generate eval graphs ---
    if all_eval_keys:
        eval_graphs_dir = graphs_dir / 'eval'
        os.makedirs(eval_graphs_dir, exist_ok=True)

        for eval_key in sorted(all_eval_keys):
            pdf_path = eval_graphs_dir / f'{eval_key}.pdf'
            if pdf_path.exists():
                print(f"  [skip] {pdf_path} already exists")
                continue

            data_per_wave = {}
            for wave_name in available_waves:
                json_path = results_dir / wave_name / 'eval' / f'{eval_key}.json'
                if json_path.exists():
                    data_per_wave[wave_name] = load_json(str(json_path))

            label = COCO_EVAL_LABELS.get(eval_key, eval_key)
            draw_comparison(
                data_per_wave,
                ylabel=label,
                title=f'{eval_key} — All Waves',
                save_path=pdf_path,
            )


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison graphs from RT-DETR training results')
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Path to a specific results_[timestamp] dir, or to the parent '
             'results/ folder (default: rtdetr_paddle/results)')
    args = parser.parse_args()

    if args.results_dir:
        base = Path(args.results_dir)
    else:
        # Default: rtdetr_paddle/results
        base = Path(__file__).resolve().parent.parent / 'results'

    if not base.exists():
        print(f"Directory does not exist: {base}")
        sys.exit(1)

    # If the given path IS a results_[timestamp] dir (has wave subdirs), process it directly
    has_wave_subdirs = any(
        (base / w).is_dir() for w in WAVE_NAMES
    )

    if has_wave_subdirs:
        results_dirs = [base]
    else:
        # Search for results_* folders inside base
        results_dirs = sorted(
            [d for d in base.iterdir()
             if d.is_dir() and d.name.startswith('results_')],
            key=lambda x: x.name
        )

    if not results_dirs:
        print(f"No results_* folders found in {base}")
        sys.exit(1)

    print(f"Found {len(results_dirs)} results folder(s)")

    for rdir in results_dirs:
        print(f"\n{'=' * 60}")
        print(f"Processing: {rdir.name}")
        print(f"{'=' * 60}")
        process_results_dir(rdir)

    print(f"\n{'=' * 60}")
    print("Graph generation completed")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
