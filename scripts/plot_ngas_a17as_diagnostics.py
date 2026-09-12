#!/usr/bin/env python3
"""Render source-backed A1.7A-S diagnostic figures."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / 'outputs/ngas_a1/trajectory_utility_a17as_v1/derived'
FIGURES = ROOT / 'reports/figures/ngas_a17as'
SOURCE = FIGURES / 'source_data'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ngas_a17as_matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
QA_TOOLS = Path(os.environ.get(
    'NATURE_FIGURE_QA_TOOLS',
    str(Path.home() / '.codex/skills/nature-figure/scripts')))
sys.path.insert(0, str(QA_TOOLS))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none', 'svg.hashsalt': 'ngas-a17as-v1',
    'pdf.fonttype': 42, 'font.size': 7, 'axes.labelsize': 7,
    'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5, 'axes.spines.right': False,
    'axes.spines.top': False, 'axes.linewidth': .7, 'legend.frameon': False,
})

COLORS = {'blue': '#3F6C8E', 'orange': '#C47A4A', 'green': '#6F9E76',
          'purple': '#7A6F9E', 'neutral': '#59636E'}
UTILITY_LABELS = {
    'U0_IMMEDIATE': 'U0 immediate',
    'U1_SHORT_HORIZON': 'U1 short horizon',
    'U2_COST_NORMALIZED': 'U2 cost normalized',
    'U3_STOCHASTIC_ROBUSTNESS': 'U3 robustness',
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict, key: str) -> float | None:
    text = row.get(key, '')
    return None if text in {'', 'None', 'NA'} else float(text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, stem: str, panel_ids: list[str] | None = None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, panel_ids=panel_ids,
        json_out=FIGURES / f'{stem}.alignment.json',
        overlay_svg=FIGURES / f'{stem}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    options = {'bbox_inches': 'tight', 'facecolor': 'white'}
    fig.savefig(
        FIGURES / f'{stem}.pdf',
        metadata={'Creator': 'RI-ACRSP NGAS A1.7A-S',
                  'CreationDate': None, 'ModDate': None}, **options)
    fig.savefig(
        FIGURES / f'{stem}.svg',
        metadata={'Creator': 'RI-ACRSP NGAS A1.7A-S', 'Date': None}, **options)
    fig.savefig(FIGURES / f'{stem}.png', dpi=600, **options)
    fig.savefig(FIGURES / f'{stem}.tiff', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'}, **options)
    for path in (FIGURES / f'{stem}.svg', FIGURES / f'{stem}.alignment.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def ranking_figure() -> dict:
    rows = [row for row in read_csv(DERIVED / 'critic_ranking_summary.csv')
            if row['dimension'] == 'scale_stage']
    write_csv(SOURCE / 'clean_critic_ranking_by_scale_stage.csv', rows)
    utilities = list(UTILITY_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.2), constrained_layout=True)
    for panel, (ax, utility) in enumerate(zip(axes.flat, utilities)):
        matrix = np.full((3, 3), np.nan)
        labels = [['' for _ in range(3)] for _ in range(3)]
        for y, scale in enumerate(('S', 'M', 'L')):
            for x, stage in enumerate(('EARLY', 'MIDDLE', 'LATE')):
                row = next(item for item in rows
                           if item['utility'] == utility
                           and item['group'] == f'{scale}|{stage}')
                metric = value(row, 'mean_spearman_valid_states')
                if metric is not None:
                    matrix[y, x] = metric
                labels[y][x] = (
                    f"{metric:.2f}\n" if metric is not None else 'NA\n') + (
                    f"n={row['valid_spearman_states']}/3")
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-.5, vmax=.5, aspect='auto')
        for y in range(3):
            for x in range(3):
                ax.text(x, y, labels[y][x], ha='center', va='center', fontsize=6,
                        color='white' if np.isfinite(matrix[y, x])
                        and abs(matrix[y, x]) > .28 else '#20252A')
        ax.set_xticks(range(3), ['Early', 'Middle', 'Late'])
        ax.set_yticks(range(3), ['S', 'M', 'L'])
        ax.set_title(f"{'abcd'[panel]}  {UTILITY_LABELS[utility]}",
                     loc='left', fontweight='bold')
        ax.set_xlabel('Search stage')
        ax.set_ylabel('Scale')
    fig.colorbar(image, ax=axes, shrink=.75, label='Mean Spearman rho on valid states')
    fig.suptitle('Clean critic ranking remains heterogeneous across scale and stage',
                 x=.01, ha='left', fontsize=9, fontweight='bold')
    save_figure(fig, 'clean_critic_ranking_by_scale_stage', ['a', 'b', 'c', 'd'])
    return {'stem': 'clean_critic_ranking_by_scale_stage',
            'source_rows': len(rows),
            'claim': 'Clean rank quality varies by utility, scale, and search stage.'}


def support_figure() -> dict:
    wanted = (
        'U0_nonconstant_to_U1_nonconstant',
        'U0_constant_to_U1_nonconstant',
        'U0_constant_to_U3_nonconstant',
        'U0_positive_best_to_U1_positive_best',
        'not_U0_positive_best_to_U1_positive_best',
    )
    rows = [row for row in read_csv(DERIVED / 'utility_support_transitions.csv')
            if row['dimension'] == 'overall' and row['transition'] in wanted]
    rows.sort(key=lambda row: wanted.index(row['transition']))
    write_csv(SOURCE / 'utility_support_transitions.csv', rows)
    labels = ['U0 signal → U1 signal', 'U0 flat → U1 signal',
              'U0 flat → U3 signal', 'U0 positive → U1 positive',
              'U0 nonpositive → U1 positive']
    values = [float(row['fraction']) if row['fraction'] else 0. for row in rows]
    fig, ax = plt.subplots(figsize=(7.09, 3.3), constrained_layout=True)
    bars = ax.barh(range(len(rows)), values,
                   color=[COLORS['blue'], COLORS['orange'], COLORS['green'],
                          COLORS['purple'], COLORS['neutral']])
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Conditional transition fraction')
    ax.set_title('Longer-horizon and robustness utilities recover signal absent in U0',
                 loc='left')
    for bar, row in zip(bars, rows):
        ax.text(bar.get_width() + .015, bar.get_y() + bar.get_height() / 2,
                f"{row['numerator']}/{row['denominator']}", va='center', fontsize=6.5)
    save_figure(fig, 'utility_support_transitions')
    return {'stem': 'utility_support_transitions', 'source_rows': len(rows),
            'claim': 'Conditional transition denominators expose utility-support recovery.'}


def alignment_figure() -> dict:
    rows = [row for row in read_csv(DERIVED / 'utility_alignment_summary.csv')
            if row['dimension'] == 'overall']
    write_csv(SOURCE / 'utility_alignment_summary.csv', rows)
    labels = [f"{row['left'][0:2]}/{row['right'][0:2]}" for row in rows]
    rho = [value(row, 'mean_spearman_valid_states') for row in rows]
    same = [float(row['same_top1_all_states_rate']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.2), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x, [0 if item is None else item for item in rho], color=COLORS['blue'])
    axes[0].axhline(0, color='#A8ADB3', lw=.7)
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel('Mean Spearman rho')
    axes[0].set_title('a  Pair-informative rank agreement', loc='left', fontweight='bold')
    for index, row in enumerate(rows):
        axes[0].text(index, -.94, f"n={row['valid_spearman_denominator']}/27",
                     ha='center', va='bottom', rotation=90, fontsize=5.5)
    axes[1].bar(x, same, color=COLORS['orange'])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('All-state same Top-1 rate')
    axes[1].set_title('b  Deterministic Top-1 agreement', loc='left', fontweight='bold')
    for index, row in enumerate(rows):
        axes[1].text(index, same[index] + .025,
                     f"{row['same_top1_all_states_numerator']}/27",
                     ha='center', va='bottom', fontsize=5.5)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha='right')
        ax.set_xlabel('Utility pair')
    fig.suptitle('Utility alignment depends on the named conditioning set',
                 x=.01, ha='left', fontsize=9, fontweight='bold')
    save_figure(fig, 'utility_alignment_summary', ['a', 'b'])
    return {'stem': 'utility_alignment_summary', 'source_rows': len(rows),
            'claim': 'Rank and Top-1 agreement use explicit, distinct denominators.'}


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    contracts = [ranking_figure(), support_figure(), alignment_figure()]
    source_checks = {
        'ranking_exact_36_rows': contracts[0]['source_rows'] == 36,
        'support_exact_5_rows': contracts[1]['source_rows'] == 5,
        'alignment_exact_6_pairs': contracts[2]['source_rows'] == 6,
    }
    source_validation = {
        'schema': 'ngas-a17as-figure-source-validation-v1',
        'status': 'PASS' if all(source_checks.values()) else 'FAIL',
        'checks': source_checks,
    }
    (FIGURES / 'source_validation.json').write_text(
        json.dumps(source_validation, indent=2, sort_keys=True) + '\n')
    generated_suffixes = {
        '.pdf', '.svg', '.png', '.tiff', '.alignment.json', '.csv'}
    artifacts = {
        relative(path): sha256(path) for path in sorted(FIGURES.rglob('*'))
        if path.is_file()
        and (''.join(path.suffixes[-2:]) == '.alignment.json'
             or path.suffix in generated_suffixes
             or path.name == 'source_validation.json')
    }
    manifest = {
        'schema': 'ngas-a17as-diagnostic-figures-v1',
        'backend': 'python-matplotlib',
        'archetype': 'quantitative diagnostic figures',
        'contracts': contracts,
        'data_integrity': {
            'excluded_rows': 0,
            'population': '27 frozen clean non-R12 development states',
            'conditional_denominators_explicit': True,
        },
        'export_formats': ['pdf', 'svg', 'png', 'tiff'],
        'artifacts': artifacts,
    }
    (FIGURES / 'figure_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': source_validation['status'],
                      'figures': len(contracts),
                      'manifest': relative(FIGURES / 'figure_manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
