#!/usr/bin/env python3
"""Shared Nature-style rendering and QA helpers for manuscript figures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


os.environ.setdefault("MPLCONFIGDIR", "/tmp/riacrsp-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
NATURE_FIGURE_ROOT = Path(
    os.environ.get(
        "NATURE_FIGURE_SKILL_ROOT",
        Path.home() / ".codex/skills/nature-figure",
    )
)
NATURE_FIGURE_SCRIPTS = NATURE_FIGURE_ROOT / "scripts"
FINAL_WIDTH_MM = 180.0
POINTS_PER_MM = 72.0 / 25.4

PALETTE = {
    "ga": "#B4C0D8",
    "dcga": "#8495B8",
    "dabc": "#60718F",
    "lghga": "#6E9D97",
    "alns": "#596B8C",
    "csgni": "#B64342",
    "neutral": "#767676",
    "grid": "#D9D9D9",
    "black": "#272727",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def require_times_new_roman() -> Path:
    user_font_root = Path.home() / ".local/share/fonts/msttcorefonts"
    for font_path in sorted(user_font_root.glob("*.ttf")):
        font_manager.fontManager.addfont(font_path)
    try:
        resolved = font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman"),
            fallback_to_default=False,
        )
    except ValueError as error:
        raise RuntimeError(
            "Times New Roman is not installed; refusing to create a noncompliant paper figure"
        ) from error
    return Path(resolved)


def apply_publication_style() -> Path:
    """Apply the manuscript's explicit Times New Roman and vector-text contract."""
    font_path = require_times_new_roman()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    return font_path


def add_panel_label(axis: Any, label: str) -> None:
    offset = ScaledTranslation(-5 / 72, 3 / 72, axis.figure.dpi_scale_trans)
    axis.text(
        0,
        1,
        label,
        transform=axis.transAxes + offset,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="bottom",
        color=PALETTE["black"],
    )


def style_axis(axis: Any) -> None:
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, zorder=0)
    axis.tick_params(direction="out")
    axis.spines["left"].set_color(PALETTE["black"])
    axis.spines["bottom"].set_color(PALETTE["black"])


def _nature_script(name: str) -> Path:
    path = NATURE_FIGURE_SCRIPTS / name
    if not path.is_file():
        raise RuntimeError(
            f"nature-figure QA script is unavailable: {path}; set NATURE_FIGURE_SKILL_ROOT"
        )
    return path


def _alignment_helper() -> Any:
    scripts = str(NATURE_FIGURE_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def _run_pdf_audits(pdf_path: Path, base_path: Path) -> dict[str, object]:
    text_result = subprocess.run(
        [
            sys.executable,
            str(_nature_script("audit_pdf_text.py")),
            str(pdf_path),
            "--min-pt",
            "5",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        text_payload = json.loads(text_result.stdout)
    except json.JSONDecodeError:
        text_payload = {
            "status": "NOT_AUDITABLE",
            "stdout": text_result.stdout,
            "stderr": text_result.stderr,
        }
    text_payload["pdf"] = str(pdf_path.relative_to(ROOT))
    atomic_json(base_path.with_suffix(".pdf-text-audit.json"), text_payload)
    if text_result.returncode:
        raise RuntimeError(
            f"PDF glyph-size audit failed for {pdf_path}: {text_result.stderr or text_result.stdout}"
        )

    collision_json = base_path.with_suffix(".collision-audit.json")
    collision_overlay = base_path.with_suffix(".collision-audit.pdf")
    # The audit tool may omit an overlay when the new render has no findings.
    # Remove any prior diagnostic first so a passed rerender cannot retain stale
    # labels from an older figure revision.
    collision_overlay.unlink(missing_ok=True)
    collision_result = subprocess.run(
        [
            sys.executable,
            str(_nature_script("audit_figure_collisions.py")),
            str(pdf_path),
            "--json-out",
            str(collision_json),
            "--overlay-pdf",
            str(collision_overlay),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if collision_result.returncode:
        raise RuntimeError(
            "rendered collision audit failed or was not auditable: "
            f"{collision_result.stderr or collision_result.stdout}"
        )
    try:
        collision_payload = json.loads(collision_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"rendered collision audit did not produce readable JSON: {collision_json}"
        ) from error
    collision_verdict = collision_payload.get("verdict")
    if collision_verdict in {"FIX BEFORE DELIVERY", "NOT AUDITABLE", None}:
        raise RuntimeError(
            f"rendered collision audit is not deliverable: {collision_verdict}"
        )
    collision_payload["pdf"] = str(pdf_path.relative_to(ROOT))
    atomic_json(collision_json, collision_payload)
    return {
        "pdf_text_audit": str(base_path.with_suffix(".pdf-text-audit.json").relative_to(ROOT)),
        "collision_audit": str(collision_json.relative_to(ROOT)),
        "collision_overlay": (
            str(collision_overlay.relative_to(ROOT))
            if collision_overlay.is_file() else None
        ),
        "collision_verdict": collision_verdict,
    }


def save_publication_figure(
    figure: Any,
    axes: Sequence[Any],
    panel_ids: Sequence[str],
    base_path: Path,
    qa_metadata: dict[str, object],
) -> list[Path]:
    """Run alignment QA, export the complete bundle, then audit the final PDF."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_json = base_path.with_suffix(".alignment.json")
    alignment_svg = base_path.with_suffix(".alignment.svg")
    require_alignment = _alignment_helper()
    alignment = require_alignment(
        figure,
        axes=list(axes),
        panel_ids=list(panel_ids),
        row_groups=[list(panel_ids)],
        json_out=alignment_json,
        overlay_svg=alignment_svg,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

    svg_path = base_path.with_suffix(".svg")
    pdf_path = base_path.with_suffix(".pdf")
    eps_path = base_path.with_suffix(".eps")
    tiff_path = base_path.with_suffix(".tiff")
    png_path = base_path.with_suffix(".png")
    # Preserve the declared 180-mm page width. ``bbox_inches='tight'`` crops the
    # PDF page to artist extents and silently changes the final physical size.
    figure.savefig(svg_path)
    figure.savefig(pdf_path)
    figure.savefig(eps_path)
    figure.savefig(tiff_path, dpi=600)
    figure.savefig(png_path, dpi=600)
    outputs = [svg_path, pdf_path, eps_path, tiff_path, png_path]

    pdf_audits = _run_pdf_audits(pdf_path, base_path)
    qa_payload = {
        **qa_metadata,
        "status": (
            "AUTOMATED_QA_PASS_VISUAL_REVIEW_REQUIRED"
            if pdf_audits["collision_verdict"] == "REVIEW REQUIRED"
            else "AUTOMATED_QA_PASS"
        ),
        "backend": "python-matplotlib",
        "final_width_mm": FINAL_WIDTH_MM,
        "font": "Times New Roman",
        "minimum_rendered_glyph_pt": 5,
        "alignment": {
            "report": str(alignment_json.relative_to(ROOT)),
            "overlay": str(alignment_svg.relative_to(ROOT)),
            "verdict": alignment.get("verdict"),
            "tolerance_pt": 1.5,
        },
        "pdf_audits": pdf_audits,
        "exports": [str(path.relative_to(ROOT)) for path in outputs],
    }
    atomic_json(base_path.with_suffix(".qa.json"), qa_payload)
    plt.close(figure)
    return outputs
