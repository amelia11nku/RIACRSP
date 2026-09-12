#!/usr/bin/env python3
"""Render source-backed A1.7A-R diagnostic figures with the Python backend."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ngas_a1/trajectory_utility_a17ar_v1/derived'
FIGURES = ROOT / 'reports/figures/ngas_a17ar'
SOURCE = FIGURES / 'source_data'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ngas_a17ar_matplotlib')
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
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': .7,
    'legend.frameon': False,
})

COLORS = {
    'clean': '#3F6C8E',
    'r12': '#C47A4A',
    'neutral': '#59636E',
    'accent': '#6F9E76',
}
ORIGIN_LABEL = {
    'CLEAN_NON_R12_DEVELOPMENT': 'Clean non-R12',
    'R12_DEVELOPMENT_EXPOSED': 'R12 development-exposed',
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


def save_figure(fig, stem: str, *, panel_ids: list[str] | None = None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, panel_ids=panel_ids,
        json_out=FIGURES / f'{stem}.alignment.json',
        overlay_svg=FIGURES / f'{stem}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    options = {'bbox_inches': 'tight', 'facecolor': 'white'}
    fig.savefig(FIGURES / f'{stem}.pdf', **options)
    fig.savefig(FIGURES / f'{stem}.svg', **options)
    fig.savefig(FIGURES / f'{stem}.png', dpi=600, **options)
    fig.savefig(FIGURES / f'{stem}.tiff', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'}, **options)
    for svg_path in (FIGURES / f'{stem}.svg', FIGURES / f'{stem}.alignment.svg'):
        lines = svg_path.read_text().splitlines()
        svg_path.write_text('\n'.join(line.rstrip() for line in lines) + '\n')
    plt.close(fig)


def critic_rank_figure() -> dict:
    rows = [row for row in read_csv(OUT / 'critic_ranking_summary.csv')
            if row['utility'] == 'U0_IMMEDIATE'
            and row['dimension'] in {
                'dataset_origin+scale', 'dataset_origin+search_stage'}]
    write_csv(SOURCE / 'critic_rank_quality.csv', rows)
    categories = [('S', 'Scale S'), ('M', 'Scale M'), ('L', 'Scale L'),
                  ('0-20%', 'Early'), ('40-60%', 'Middle'), ('80-100%', 'Late')]
    fig, ax = plt.subplots(figsize=(7.09, 3.0), constrained_layout=True)
    offsets = {'CLEAN_NON_R12_DEVELOPMENT': -.13,
               'R12_DEVELOPMENT_EXPOSED': .13}
    for origin, color in (('CLEAN_NON_R12_DEVELOPMENT', COLORS['clean']),
                          ('R12_DEVELOPMENT_EXPOSED', COLORS['r12'])):
        xs, ys = [], []
        for index, (token, _label) in enumerate(categories):
            match = [row for row in rows if row['group'] == f'{origin}|{token}']
            if not match:
                continue
            metric = value(match[0], 'mean_spearman_informative')
            if metric is None:
                continue
            x = index + offsets[origin]
            xs.append(x); ys.append(metric)
            ax.text(x, metric + .012, f"{match[0]['informative_states']}/{match[0]['states']}",
                    color=color, fontsize=5.5, ha='center', va='bottom')
        ax.plot(xs, ys, marker='o', ms=4.5, lw=0, linestyle='none', color=color,
                label=ORIGIN_LABEL[origin])
    ax.axhline(0, color='#A8ADB3', lw=.7, zorder=0)
    ax.axvline(2.5, color='#D8DADD', lw=.7)
    ax.set_xticks(range(len(categories)), [label for _token, label in categories])
    ax.set_ylabel('Mean Spearman rho on informative U0 states')
    ax.set_title('Immediate-utility ranking is weak across scale and search stage', loc='left')
    ax.legend(loc='upper left', ncols=2)
    ax.text(.99, .02, 'Labels: informative states / audited states',
            transform=ax.transAxes, ha='right', va='bottom', color=COLORS['neutral'], fontsize=5.5)
    ax.set_ylim(-.05, max(.2, ax.get_ylim()[1]))
    save_figure(fig, 'critic_rank_quality_by_scale_stage')
    return {'stem': 'critic_rank_quality_by_scale_stage', 'source_rows': len(rows),
            'claim': 'U0 rank correlation is weak across audited scale and stage strata.'}


def utility_alignment_figure() -> dict:
    rows = [row for row in read_csv(OUT / 'utility_alignment.csv')
            if row['left'] == 'U0_IMMEDIATE' and row['right'] == 'U1_SHORT_HORIZON']
    write_csv(SOURCE / 'u0_u1_ranking_agreement.csv', rows)
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    for index, (origin, color) in enumerate((
            ('CLEAN_NON_R12_DEVELOPMENT', COLORS['clean']),
            ('R12_DEVELOPMENT_EXPOSED', COLORS['r12']))):
        subset = [row for row in rows if row['dataset_origin'] == origin]
        valid = [row for row in subset if row['spearman'] not in {'', 'None', 'NA'}]
        xs = index + np.linspace(-.08, .08, num=len(valid))
        ys = [float(row['spearman']) for row in valid]
        ax.scatter(xs, ys, s=20, color=color, alpha=.75, edgecolors='white', linewidths=.35)
        mean = float(np.mean(ys)) if ys else float('nan')
        ax.plot([index - .16, index + .16], [mean, mean], color='#20252A', lw=1.5)
        same = np.mean([row['same_top1'] == 'True' for row in subset])
        ax.text(index, -.97,
                f'informative n={len(valid)}/{len(subset)}\nall-state same Top-1={same:.2f}',
                ha='center', va='bottom', fontsize=5.7, color=COLORS['neutral'])
    ax.axhline(0, color='#A8ADB3', lw=.7)
    ax.set_xticks([0, 1], [ORIGIN_LABEL['CLEAN_NON_R12_DEVELOPMENT'],
                          ORIGIN_LABEL['R12_DEVELOPMENT_EXPOSED']])
    ax.set_ylabel('U0 versus U1 action-rank Spearman rho')
    ax.set_ylim(-1.02, 1.02)
    ax.set_title('Informative U0/U1 rankings often align, but signal is sparse', loc='left')
    save_figure(fig, 'u0_u1_ranking_agreement')
    return {'stem': 'u0_u1_ranking_agreement', 'source_rows': len(rows),
            'claim': 'Conditional U0-U1 action rankings agree, while utility is often uninformative.'}


def cost_normalized_figure() -> dict:
    rows = [row for row in read_csv(OUT / 'utility_alignment.csv')
            if row['left'] == 'U1_SHORT_HORIZON'
            and row['right'] == 'U2_COST_NORMALIZED']
    write_csv(SOURCE / 'cost_normalized_utility.csv', rows)
    fig, ax = plt.subplots(figsize=(4.5, 3.1), constrained_layout=True)
    metrics = ('Mean rank rho', 'Same Top-1 rate', 'Top-5 overlap fraction')
    width = .32
    for index, (origin, color) in enumerate((
            ('CLEAN_NON_R12_DEVELOPMENT', COLORS['clean']),
            ('R12_DEVELOPMENT_EXPOSED', COLORS['r12']))):
        subset = [row for row in rows if row['dataset_origin'] == origin]
        valid = [float(row['spearman']) for row in subset if row['spearman']]
        ys = [float(np.mean(valid)),
              float(np.mean([row['same_top1'] == 'True' for row in subset])),
              float(np.mean([float(row['top5_overlap']) / 5. for row in subset]))]
        xs = np.arange(3) + (index - .5) * width
        ax.bar(xs, ys, width=width, color=color, label=ORIGIN_LABEL[origin])
        ax.text(xs[0], ys[0] - .035, f'n={len(valid)}', ha='center', va='top',
                fontsize=5.5, color='white', fontweight='bold')
    ax.set_xticks(range(3), metrics)
    ax.set_ylabel('Agreement between U1 and U2 action rankings')
    ax.set_ylim(0, 1.22)
    ax.set_title('Cost normalization leaves short-horizon ranking nearly unchanged', loc='left')
    ax.legend(loc='upper left')
    save_figure(fig, 'cost_normalized_utility_summary')
    return {'stem': 'cost_normalized_utility_summary', 'source_rows': len(rows),
            'claim': 'Cost normalization leaves the short-horizon action order nearly unchanged.'}


def staleness_figure() -> dict:
    rows = [row for row in read_csv(OUT / 'prior_staleness_summary.csv')
            if row['dimension'] == 'dataset_origin']
    write_csv(SOURCE / 'prior_staleness_curve.csv', rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    for origin, color in (('CLEAN_NON_R12_DEVELOPMENT', COLORS['clean']),
                          ('R12_DEVELOPMENT_EXPOSED', COLORS['r12'])):
        subset = sorted((row for row in rows if row['group'] == origin),
                        key=lambda row: int(row['offset']))
        x = [int(row['offset']) for row in subset]
        bank = [float(row['mean_semantic_bank_jaccard']) for row in subset]
        rank = [float(row['mean_common_rank_spearman']) for row in subset]
        ax.plot(x, bank, marker='o', ms=4, lw=1.35, color=color,
                label=f"{ORIGIN_LABEL[origin]}: bank overlap")
        ax.plot(x, rank, marker='s', ms=3.5, lw=1.1, ls='--', color=color,
                label=f"{ORIGIN_LABEL[origin]}: common-action rank")
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_xlabel('Iterations since frozen refresh')
    ax.set_ylabel('Similarity to refresh state')
    ax.set_ylim(0, 1.04)
    ax.set_title('Candidate-bank churn is distinct from common-action rank drift', loc='left')
    ax.legend(loc='center right', fontsize=5.8)
    save_figure(fig, 'prior_staleness_curve')
    return {'stem': 'prior_staleness_curve', 'source_rows': len(rows),
            'claim': 'Semantic bank overlap collapses while common-action order remains stable.'}


def trial_value_figure() -> dict:
    rows = [row for row in read_csv(OUT / 'candidate_trial_curve.csv')
            if row['dimension'] == 'overall']
    rows.sort(key=lambda row: int(row['trial']))
    write_csv(SOURCE / 'candidate_trial_marginal_value.csv', rows)
    x = [int(row['trial']) for row in rows]
    marginal = [float(row['mean_raw_marginal_best_reduction']) for row in rows]
    later = [float(row['probability_best_changes_after_k']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.0), constrained_layout=True)
    axes[0].plot(x, marginal, marker='o', ms=4, lw=1.4, color=COLORS['clean'])
    axes[0].set_ylabel('Mean marginal best-makespan reduction')
    axes[0].set_title('a  Marginal outcome value', loc='left', fontweight='bold')
    axes[1].plot(x, later, marker='o', ms=4, lw=1.4, color=COLORS['r12'])
    axes[1].set_ylabel('Probability the best outcome changes later')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('b  Remaining selection uncertainty', loc='left', fontweight='bold')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xlabel('Trial cap k')
        for cap in (1, 2, 4):
            ax.axvline(cap, color='#D8DADD', lw=.55, zorder=0)
    fig.suptitle('Additional stochastic trials retain measurable value through trial 8',
                 x=.01, ha='left', fontsize=8.5, fontweight='bold')
    save_figure(fig, 'candidate_trial_marginal_value', panel_ids=['a', 'b'])
    return {'stem': 'candidate_trial_marginal_value', 'source_rows': len(rows),
            'claim': 'Early caps lose quality and often miss a later best repair realization.'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    contracts = [
        critic_rank_figure(), utility_alignment_figure(),
        cost_normalized_figure(), staleness_figure(), trial_value_figure(),
    ]
    artifacts = {}
    for path in sorted(FIGURES.rglob('*')):
        if path.is_file() and path.name != 'figure_manifest.json':
            artifacts[relative(path)] = sha256(path)
    manifest = {
        'schema': 'ngas-a17ar-diagnostic-figures-v1',
        'backend': 'python-matplotlib',
        'archetype': 'quantitative diagnostic figures',
        'contracts': contracts,
        'data_integrity': {
            'excluded_rows': 0,
            'R12_claim_scope': 'development-exposed diagnostics only',
            'statistical_scope': 'state-level exploratory summaries',
        },
        'export_formats': ['pdf', 'svg', 'png', 'tiff'],
        'artifacts': artifacts,
    }
    (FIGURES / 'figure_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'figures': len(contracts),
                      'manifest': relative(FIGURES / 'figure_manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
