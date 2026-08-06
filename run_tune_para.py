# -*- coding: utf-8 -*-
"""
Six-factor full-factorial DOE, ANOVA, and publication-style figures for BLCMA.

Tuned factors:
- explorer_ratio: share of the population used by the Q-guided explorer pool.
- pc: exploiter crossover probability.
- pm: exploiter mutation probability.
- q_epsilon: epsilon-greedy exploration probability shared by explorer/exploiter Q selectors.
- q_alpha: learning rate shared by explorer/exploiter Q selectors.
- population_size: total population size.

Fixed budget setting:
- a per-instance time budget of 2 * num_operations seconds
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from statsmodels.formula.api import ols

from BLCMA import BLCMA
from data.loader import load_instance


OUTDIR = Path("doe_out_0615_5factor")
OUTDIR.mkdir(exist_ok=True)

PC_LEVELS = [0.70, 0.80, 0.95]
PM_LEVELS = [0.10, 0.20, 0.30]
Q_ALPHA_LEVELS = [0.10, 0.20, 0.30]
Q_EPSILON_LEVELS = [0.05, 0.10, 0.20]
POPULATION_SIZE_LEVELS = [100, 150, 200]

FACTORS = ["pc", "pm", "q_alpha", "q_epsilon", "population_size"]
FACTOR_LEVELS = {
    "pc": PC_LEVELS,
    "pm": PM_LEVELS,
    "q_alpha": Q_ALPHA_LEVELS,
    "q_epsilon": Q_EPSILON_LEVELS,
    "population_size": POPULATION_SIZE_LEVELS,
}
RAW_COLUMNS = ["instance", "seed", *FACTORS, "makespan", "elapsed_s", "ok"]
FACTOR_LABELS = {
    "pc": r"$p_c$",
    "pm": r"$p_m$",
    "q_alpha": r"$\alpha$",
    "q_epsilon": r"$\varepsilon$",
    "population_size": r"$P$",
}
INTERACTION_COLORS = ["#4E79A7", "#ECA55D", "#458652"]
INTERACTION_MARKERS = ["o", "s", "D"]
INTERACTION_LINESTYLES = ["-", "--", ":"]
BLACK = "#1F252B"
GREY = "#D7DEE3"
LIGHT_GREY = "#F3F5F6"

FIXED_PARAMS = dict(
    migration_interval=10,
    migration_rate=0.10,
    elite_ratio=0.05,
    max_iterations=200000,
    early_stop_patience=5000,
    run_time_ratio=2.0,
    explorer_ratio=0.30,  
)
PLOT_FONT_SIZE = 15

# Four representative tuning instances, one from each available benchmark family/scale:
# - Fattahi06: small classical FJSP-style instance. S06
# - kumar10: medium transport-constrained instance. S19
# - MK01: larger MK instance with more operations. 无
# - Behnke01: high machine-flexibility instance. M20
INSTANCE_NAMES = ["AFAISP-S20", "AFAISP-M10", "AFAISP-M15"]
N_RUNS_PER_COMBO = 10
SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
    122, 133, 144, 155, 166,
    177, 188, 199, 211, 222,
]


def time_budget_seconds(instance) -> float:
    return instance.num_operations * 2


def make_job_key(instance, seed, ndigits=4, **params):
    return (str(instance), int(seed), *(
        round(float(params[factor]), ndigits) for factor in FACTORS
    ))


def generate_param_grid() -> list[dict[str, float]]:
    combos = []
    for values in itertools.product(*(FACTOR_LEVELS[factor] for factor in FACTORS)):
        combos.append(dict(zip(FACTORS, values)))
    return combos


def normalize_raw_runs(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep persisted DOE rows in a stable schema for resume and appends."""
    df_raw = df_raw.copy()
    for column in RAW_COLUMNS:
        if column not in df_raw.columns:
            df_raw[column] = pd.NA
    df_raw = df_raw[RAW_COLUMNS]

    numeric_columns = ["seed", *FACTORS, "makespan", "elapsed_s"]
    for column in numeric_columns:
        df_raw[column] = pd.to_numeric(df_raw[column], errors="coerce")
    df_raw["instance"] = df_raw["instance"].astype("string")
    df_raw["ok"] = df_raw["ok"].astype("boolean")
    return df_raw


def append_raw_row(df_raw: pd.DataFrame, row: dict) -> pd.DataFrame:
    row_df = normalize_raw_runs(pd.DataFrame([row]))
    if df_raw.empty:
        return row_df.reset_index(drop=True)
    return pd.concat([df_raw, row_df], ignore_index=True)


def run_single(instance_name, seed, pc, pm, q_epsilon, q_alpha, population_size) -> float:
    random.seed(seed)
    np.random.seed(seed)

    ins = load_instance(instance_name)
    solver = BLCMA(
        instance=ins,
        population_size=int(population_size),
        pc_e=pc,              # kept for constructor compatibility; explorer no longer uses GA.
        pm_e=pm,
        pc_x=pc,
        pm_x=pm,
        q_alpha_e=q_alpha,
        q_alpha_x=q_alpha,
        q_epsilon_e=q_epsilon,
        q_epsilon_x=q_epsilon,
        **FIXED_PARAMS,
    )
    solver.max_run_time = time_budget_seconds(ins)
    best_sol, _curve = solver.evolve()
    return float(best_sol[1])


def run_doe_all(resume=True) -> pd.DataFrame:
    raw_path = OUTDIR / "raw_runs.csv"

    if resume and raw_path.exists():
        df_raw = normalize_raw_runs(pd.read_csv(raw_path))
        print(f"[resume] loaded {raw_path} with {len(df_raw)} rows")
    else:
        df_raw = normalize_raw_runs(pd.DataFrame(columns=RAW_COLUMNS))

    done_keys = set()
    if not df_raw.empty:
        complete_rows = df_raw.dropna(subset=["instance", "seed", *FACTORS])
        for row in complete_rows.itertuples(index=False):
            done_keys.add(
                make_job_key(row.instance, row.seed, **{
                    factor: getattr(row, factor) for factor in FACTORS
                })
            )

    combos = generate_param_grid()
    total_jobs = len(combos) * len(INSTANCE_NAMES) * N_RUNS_PER_COMBO
    completed_count = len(done_keys)
    job_id = 0
    print(f"[info] DOE combinations: {len(combos)}; total jobs: {total_jobs}; done: {completed_count}")

    t0 = time.time()
    for combo in combos:
        combo_desc = ", ".join(f"{k}={combo[k]}" for k in FACTORS)
        print(f"\n=== {combo_desc} ===")

        for inst in INSTANCE_NAMES:
            for rep_i in range(N_RUNS_PER_COMBO):
                seed = SEEDS[rep_i % len(SEEDS)]
                job_key = make_job_key(inst, seed, **combo)
                job_id += 1

                if job_key in done_keys:
                    print(f"[{job_id}/{total_jobs}] skip {inst}, seed={seed}")
                    continue

                print(f"[{job_id}/{total_jobs}] run {inst}, seed={seed}", flush=True)
                start = time.time()
                try:
                    makespan = run_single(inst, seed, **combo)
                    ok = True
                except Exception as exc:
                    print(f"  -> failed: {exc}")
                    makespan = float("inf")
                    ok = False

                row = {
                    "instance": inst,
                    "seed": seed,
                    **combo,
                    "makespan": makespan,
                    "elapsed_s": time.time() - start,
                    "ok": ok,
                }
                df_raw = append_raw_row(df_raw, row)
                done_keys.add(job_key)
                completed_count += 1
                df_raw.to_csv(raw_path, index=False)

    print(f"\nRaw results saved to {raw_path}")
    print(f"Elapsed in this run: {time.time() - t0:.1f}s")
    return df_raw


def compute_summary(df_raw: pd.DataFrame):
    df_valid = df_raw[(df_raw["ok"].astype(bool)) & np.isfinite(df_raw["makespan"])].copy()
    if df_valid.empty:
        raise ValueError("No valid DOE rows are available for analysis.")

    best_run_per_instance = (
        df_valid.groupby("instance", as_index=False)["makespan"]
        .min()
        .rename(columns={"makespan": "best_makespan_instance"})
    )
    df_valid = df_valid.merge(best_run_per_instance, on="instance", how="left")
    df_valid["rel_mk"] = df_valid["makespan"] / df_valid["best_makespan_instance"]
    df_valid["rpd"] = (
        (df_valid["makespan"] - df_valid["best_makespan_instance"])
        / df_valid["best_makespan_instance"]
    )

    group_cols_inst = [*FACTORS, "instance"]
    df_inst_mean = (
        df_valid.groupby(group_cols_inst, as_index=False)
        .agg(
            mean_makespan=("makespan", "mean"),
            std_makespan=("makespan", "std"),
            mean_rel_mk=("rel_mk", "mean"),
            std_rel_mk=("rel_mk", "std"),
            mean_rpd=("rpd", "mean"),
            std_rpd=("rpd", "std"),
            worst_rel_mk=("rel_mk", "max"),
            n_runs=("makespan", "count"),
        )
    )
    df_inst_mean["rel_mk"] = df_inst_mean["mean_rel_mk"]
    df_inst_mean["rpd"] = df_inst_mean["mean_rpd"]

    df_combo_sum = (
        df_valid.groupby(FACTORS, as_index=False)
        .agg(
            avg_makespan=("makespan", "mean"),
            mean_rel_mk=("rel_mk", "mean"),
            median_rel_mk=("rel_mk", "median"),
            std_rel_mk=("rel_mk", "std"),
            worst_rel_mk=("rel_mk", "max"),
            mean_rpd=("rpd", "mean"),
            median_rpd=("rpd", "median"),
            std_rpd=("rpd", "std"),
            n_runs=("makespan", "count"),
            n_instances=("instance", "nunique"),
        )
        .sort_values(["mean_rel_mk", "worst_rel_mk", "std_rel_mk"], ascending=True)
        .reset_index(drop=True)
    )

    combo_rank_rows = []
    for inst, part in df_valid.groupby("instance", sort=True):
        ranked = (
            part.groupby(FACTORS, as_index=False)["rel_mk"]
            .mean()
            .sort_values("rel_mk", ascending=True)
            .reset_index(drop=True)
        )
        ranked["instance"] = inst
        ranked["instance_rank"] = np.arange(1, len(ranked) + 1)
        combo_rank_rows.append(ranked)
    df_combo_ranks = pd.concat(combo_rank_rows, ignore_index=True)
    df_rank_stability = (
        df_combo_ranks.groupby(FACTORS, as_index=False)
        .agg(
            mean_instance_rank=("instance_rank", "mean"),
            worst_instance_rank=("instance_rank", "max"),
            std_instance_rank=("instance_rank", "std"),
        )
    )
    df_combo_sum = df_combo_sum.merge(df_rank_stability, on=FACTORS, how="left")
    df_combo_sum = (
        df_combo_sum
        .sort_values(
            ["mean_rel_mk", "worst_rel_mk", "mean_instance_rank", "std_rel_mk"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    df_valid.to_csv(OUTDIR / "raw_runs_valid.csv", index=False)
    df_inst_mean.to_csv(OUTDIR / "summary_per_instance.csv", index=False)
    df_combo_sum.to_csv(OUTDIR / "summary_by_combo.csv", index=False)
    best_run_per_instance.to_csv(OUTDIR / "normalization_baselines.csv", index=False)
    df_combo_ranks.to_csv(OUTDIR / "combo_instance_ranks.csv", index=False)

    print("\nTop DOE candidates by normalized robust mean:")
    print(df_combo_sum.head(10).to_string(index=False, float_format="%.4f"))
    return df_valid, df_inst_mean, df_combo_sum


def _anova_formula(response: str, with_interactions: bool) -> str:
    factor_terms = " + ".join(f"C({f})" for f in FACTORS)
    if with_interactions:
        rhs = f"({factor_terms})**2"
    else:
        rhs = factor_terms
    return f"{response} ~ {rhs}"


def run_anova(df_for_anova: pd.DataFrame):
    model_data = df_for_anova.copy()
    main_model = ols(_anova_formula("rpd", with_interactions=False), data=model_data).fit()
    anova_main = sm.stats.anova_lm(main_model, typ=2)
    anova_main.to_csv(OUTDIR / "anova_main_effects.csv")

    try:
        inter_model = ols(_anova_formula("rpd", with_interactions=True), data=model_data).fit()
        anova_inter = sm.stats.anova_lm(inter_model, typ=2)
        anova_inter.to_csv(OUTDIR / "anova_with_interactions.csv")
    except Exception as exc:
        anova_inter = pd.DataFrame({"warning": [str(exc)]})
        anova_inter.to_csv(OUTDIR / "anova_with_interactions_unavailable.csv", index=False)
        print(f"[warning] Interaction ANOVA unavailable for this design: {exc}")

    print(f"\nANOVA main effects saved to {OUTDIR / 'anova_main_effects.csv'}")
    print(anova_main)
    print(f"\nANOVA with two-way interactions saved to {OUTDIR / 'anova_with_interactions.csv'}")
    return anova_main, anova_inter


def set_plot_style() -> None:
    sns.set_theme(context="paper", style="white", font_scale=1.0)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": PLOT_FONT_SIZE,  # 调整全局字体大小
            "axes.labelsize": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7, 
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def _format_numeric_axis(ax, format_x: bool = True) -> None:
    """Use a consistent two-decimal numeric format across DOE figures."""
    if format_x:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _mean_sem(series: pd.Series) -> float:
    n = int(series.count())
    if n <= 1:
        return 0.0
    return float(series.std(ddof=1) / np.sqrt(n))


def plot_main_effects_grid(df_for_plot: pd.DataFrame) -> pd.DataFrame:
    source_rows = []
    for factor in FACTORS:
        part = (
            df_for_plot.groupby(factor, as_index=False)
            .agg(mean_mrm=("rel_mk", "mean"), sem_mrm=("rel_mk", _mean_sem), n=("rel_mk", "count"))
        )
        part.insert(0, "factor", factor)
        part.rename(columns={factor: "level"}, inplace=True)
        source_rows.append(part)
    source = pd.concat(source_rows, ignore_index=True)
    source.to_csv(OUTDIR / "main_effects_source_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(7.2, 1.8 * len(FACTORS)), 4.2))
    color = "#0B5CAD"
    level_spacing = 1.5
    section_gap = 0.0
    x_positions, x_labels = [], []
    section_centers, boundaries = [], []
    cursor = 0.0

    for factor_index, factor in enumerate(FACTORS):
        part = source[source["factor"] == factor].sort_values("level")
        positions = np.arange(len(part), dtype=float) * level_spacing + cursor

        ax.plot(
            positions,
            part["mean_mrm"],
            marker="o",
            markersize=3.5, 
            linewidth=0.9, 
            color=color,
        )
        x_positions.extend(positions.tolist())
        x_labels.extend(f"{float(level):.2f}" for level in part["level"])
        section_centers.append(float(np.mean(positions)))

        cursor = float(positions[-1] + section_gap + level_spacing)
        if factor_index < len(FACTORS) - 1:
            boundaries.append(float(positions[-1] + (section_gap + level_spacing) / 2.0))

    global_mean = float(df_for_plot["rel_mk"].mean())
    ax.axhline(global_mean, color="#8A8A8A", linestyle="--", linewidth=0.8, zorder=0)
    for boundary in boundaries:
        ax.axvline(boundary, color="#8A8A8A", linewidth=0.65, zorder=0)
    
    ax.set_xticks(x_positions )
    ax.set_xticklabels(x_labels, rotation=25, ha="right")  # 设置倾斜的 x 轴标签
    _format_numeric_axis(ax, format_x=False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_ylabel("MRM (↓)")
    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)  # 增加左右边距以适应倾斜标签
    ax.grid(False)

    # Only show two y-axis ticks: bottom and top with margins.
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_bottom = y_min + 0.05 * y_range  # 下刻度距离底部 5% 的距离
    y_top = y_max - 0.05 * y_range      # 上刻度距离顶部 5% 的距离
    ax.set_yticks([y_bottom, y_top])
    ax.set_yticklabels([f"{y_bottom:.2f}", f"{y_top:.2f}"])


    for center, factor in zip(section_centers, FACTORS):
        ax.text(
            center,
            1.015,
            FACTOR_LABELS[factor],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=PLOT_FONT_SIZE,
            fontweight="bold",
            clip_on=False,
        )

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_linewidth(0.65)
    ax.spines["right"].set_linewidth(0.65)
    fig.tight_layout(pad=0.8, rect=(0, 0, 1, 0.96))
    output_path = OUTDIR / "main_effects_grid.pdf"
    try:
        fig.savefig(output_path, bbox_inches="tight")
    except PermissionError:
        print(f"[warning] Could not overwrite open figure file: {output_path}")
    plt.close(fig)
    return source


def _interaction_source(df_for_plot: pd.DataFrame, factor_x: str, factor_trace: str) -> pd.DataFrame:
    source = (
        df_for_plot.groupby([factor_x, factor_trace], as_index=False)
        .agg(mean_mrm=("rel_mk", "mean"), sem_mrm=("rel_mk", _mean_sem), n=("rel_mk", "count"))
        .sort_values([factor_trace, factor_x])
    )
    return source


def plot_interactions(df_for_plot: pd.DataFrame) -> pd.DataFrame:
    """Plot a symmetric factor-by-factor interaction matrix.

    Columns define the x-axis factor and rows define the line/trace factor.
    The diagonal is intentionally blank.
    """
    all_sources = []
    grid_size = len(FACTORS)
    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(max(10.0, 1.85 * grid_size + 2.4), max(8.0, 1.65 * grid_size)),
        sharey=True,
        squeeze=False,
    )
    # hspace controls the vertical gap reserved for each subplot title row.
    fig.subplots_adjust(left=0.075, right=0.80, bottom=0.10, top=0.97, wspace=0.00, hspace=0.15)

    # Reserve a title band at the top of every matrix row. The axes are shortened
    # while the original cell boundaries remain the continuous matrix grid.
    title_band_fraction = 0.14
    column_bounds = [axes[0, col].get_position().x0 for col in range(grid_size)]
    column_bounds.append(axes[0, grid_size - 1].get_position().x1)
    row_bounds = [axes[row, 0].get_position().y1 for row in range(grid_size)]
    row_bounds.append(axes[grid_size - 1, 0].get_position().y0)
    title_separator_bounds = []
    for row in range(grid_size):
        original_position = axes[row, 0].get_position()
        plot_top = original_position.y1 - original_position.height * title_band_fraction
        title_separator_bounds.append(plot_top)
        for col in range(grid_size):
            position = axes[row, col].get_position()
            axes[row, col].set_position(
                [position.x0, position.y0, position.width, plot_top - position.y0]
            )

    # The diagonal makes the first visible plot vary by row and the bottom plot
    # vary by column. Keep labels only on those outermost visible plots.
    leftmost_visible = [1 if row == 0 else 0 for row in range(grid_size)]
    bottommost_visible = [grid_size - 2 if col == grid_size - 1 else grid_size - 1 for col in range(grid_size)]

    for row, factor_trace in enumerate(FACTORS):
        trace_levels = sorted(float(level) for level in FACTOR_LEVELS[factor_trace])
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=INTERACTION_COLORS[idx % len(INTERACTION_COLORS)],
                marker=INTERACTION_MARKERS[idx % len(INTERACTION_MARKERS)],
                linestyle=INTERACTION_LINESTYLES[idx % len(INTERACTION_LINESTYLES)],
                linewidth=0.8,
                markersize=3.2,
            )
            for idx in range(len(trace_levels))
        ]

        for col, factor_x in enumerate(FACTORS):
            ax = axes[row, col]
            if row == col:
                ax.set_visible(False)
                continue

            source = _interaction_source(df_for_plot, factor_x, factor_trace)
            source.insert(0, "trace_factor", factor_trace)
            source.insert(0, "x_factor", factor_x)
            all_sources.append(source)

            for idx, (_level, part) in enumerate(source.groupby(factor_trace, sort=True)):
                part = part.sort_values(factor_x)
                x_positions = np.arange(1, len(part) + 1)
                ax.plot(
                    x_positions,
                    part["mean_mrm"],
                    marker=INTERACTION_MARKERS[idx % len(INTERACTION_MARKERS)],
                    markersize=3.2,  # 调整标记大小以适应更紧凑的子图
                    linewidth=0.8,  # 调整线宽以适应更紧凑的子图
                    linestyle=INTERACTION_LINESTYLES[idx % len(INTERACTION_LINESTYLES)],
                    color=INTERACTION_COLORS[idx % len(INTERACTION_COLORS)],
                )

            ax.set_title(
                f"{FACTOR_LABELS[factor_x]} $\\times$ {FACTOR_LABELS[factor_trace]}",
                pad=3,  # Adjust this value to change the title-to-axes gap.
                fontweight="bold",
            )
            x_levels = sorted(source[factor_x].astype(float).unique())
            ax.set_xticks(np.arange(1, len(x_levels) + 1))
            ax.set_xticklabels(f"{level:.2f}" for level in x_levels)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
            ax.grid(axis="y", color=GREY, linewidth=0.45)
            ax.set_facecolor("white")

            # The complete matrix grid is drawn once at figure level below.
            # Hiding subplot spines avoids doubled or broken separator lines.
            for spine in ax.spines.values():
                spine.set_visible(False)

            show_y = col == leftmost_visible[row]
            # Draw all x tick labels later on one shared matrix baseline.
            show_x = False
            ax.tick_params(labelleft=show_y, labelbottom=show_x)
            if not show_x:
                ax.tick_params(axis="x", length=0)

        row_center = (row_bounds[row] + row_bounds[row + 1]) / 2.0
        fig.legend(
            handles=legend_handles,
            labels=[f"{level:.2f}" for level in trace_levels],
            title=FACTOR_LABELS[factor_trace],
            loc="center left",
            bbox_to_anchor=(0.815, row_center),
            handlelength=2.4,
            borderaxespad=0.0,
        )

    fig.supylabel("MRM (↓)", x=0.018, fontsize=PLOT_FONT_SIZE, fontweight="bold")

    # Put every column's ticks and factor label on one common baseline. This also
    # moves the P-column ticks below the matrix instead of below its last subplot.
    visible_axes = [
        axes[row, col]
        for row in range(grid_size)
        for col in range(grid_size)
        if row != col
    ]
    matrix_bottom = min(ax.get_position().y0 for ax in visible_axes)
    tick_label_y = matrix_bottom - 0.012
    column_label_y = matrix_bottom - 0.060
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()

    for col, factor_x in enumerate(FACTORS):
        ax = axes[bottommost_visible[col], col]
        levels = sorted(float(level) for level in FACTOR_LEVELS[factor_x])
        x_positions = np.arange(1, len(levels) + 1)
        for level_index, (x_position, level) in enumerate(zip(x_positions, levels)):
            display_x = ax.transData.transform((x_position, 0.0))[0]
            figure_x = to_figure.transform((display_x, 0.0))[0]

            # With wspace=0, edge labels are aligned inward to avoid collisions
            # between neighboring columns.
            if level_index == 0:
                horizontal_alignment = "left"
            elif level_index == len(levels) - 1:
                horizontal_alignment = "right"
            else:
                horizontal_alignment = "center"
            fig.text(
                figure_x,
                tick_label_y,
                f"{level:.2f}",
                ha=horizontal_alignment,
                va="top",
                fontsize=PLOT_FONT_SIZE,
            )

        center_x = (ax.get_position().x0 + ax.get_position().x1) / 2.0
        fig.text(
            center_x,
            column_label_y,
            FACTOR_LABELS[factor_x],
            ha="center",
            va="top",
            fontsize=PLOT_FONT_SIZE,
            fontweight="bold",
        )

    # Draw a continuous matrix grid, including the blank diagonal cells.
    matrix_left = min(ax.get_position().x0 for ax in visible_axes)
    matrix_right = max(ax.get_position().x1 for ax in visible_axes)
    matrix_top = row_bounds[0]

    for x_position in column_bounds:
        fig.add_artist(
            Line2D(
                [x_position, x_position],
                [matrix_bottom, matrix_top],
                transform=fig.transFigure,
                color=BLACK,
                linewidth=0.65,
                clip_on=False,
            )
        )
    for y_position in row_bounds:
        fig.add_artist(
            Line2D(
                [matrix_left, matrix_right],
                [y_position, y_position],
                transform=fig.transFigure,
                color=BLACK,
                linewidth=0.65,
                clip_on=False,
            )
        )
    for y_position in title_separator_bounds:
        fig.add_artist(
            Line2D(
                [matrix_left, matrix_right],
                [y_position, y_position],
                transform=fig.transFigure,
                color=BLACK,
                linewidth=0.65,
                clip_on=False,
            )
        )

    output_path = OUTDIR / "interaction_grid.pdf"
    try:
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    except PermissionError:
        print(f"[warning] Could not overwrite open figure file: {output_path}")
    plt.close(fig)

    combined = pd.concat(all_sources, ignore_index=True)
    combined.to_csv(OUTDIR / "interaction_source_data_all.csv", index=False)
    return combined


def phase_run_doe() -> pd.DataFrame:
    return run_doe_all(resume=True)


def phase_analyze(df_raw=None) -> None:
    if df_raw is None:
        raw_path = OUTDIR / "raw_runs.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing DOE raw file: {raw_path}")
        df_raw = normalize_raw_runs(pd.read_csv(raw_path))

    df_valid, df_inst_mean, df_combo_sum = compute_summary(df_raw)
    df_for_anova = df_valid.copy()

    run_anova(df_for_anova)

    set_plot_style()
    plot_main_effects_grid(df_for_anova)
    plot_interactions(df_for_anova)

    best = df_combo_sum.iloc[0].to_dict()
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    with open(OUTDIR / "best_candidate.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nBest robust candidate by normalized mean:")
    for key, value in best.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nFigures and source data saved to {OUTDIR}")


def main():
    df_raw = phase_run_doe()
    phase_analyze(df_raw)
    print("\nDone.")


if __name__ == "__main__":
    main()

