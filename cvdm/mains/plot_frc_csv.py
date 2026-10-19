import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {"Spatial_Frequency", "FRC"}


def _read_frc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    return df


def _crossing_frequency(freq: np.ndarray, values: np.ndarray, threshold: float) -> float:
    for i in range(1, len(freq)):
        y0 = float(values[i - 1])
        y1 = float(values[i])
        if (y0 >= threshold and y1 < threshold) or (y0 <= threshold and y1 > threshold):
            x0 = float(freq[i - 1])
            x1 = float(freq[i])
            if x1 == x0:
                return x0
            alpha = (threshold - y0) / (y1 - y0)
            return x0 + alpha * (x1 - x0)
    return float("nan")


def _resolution_nm(freq_cross: float) -> float:
    if np.isfinite(freq_cross) and freq_cross > 0:
        return 1.0 / freq_cross
    return float("nan")


def _pick_curve(df: pd.DataFrame, prefer_smoothed: bool) -> Tuple[np.ndarray, np.ndarray, str]:
    x = df["Spatial_Frequency"].to_numpy(dtype=float)
    if prefer_smoothed and "Smoothed_FRC" in df.columns:
        y = df["Smoothed_FRC"].to_numpy(dtype=float)
        label_suffix = " (smoothed)"
    else:
        y = df["FRC"].to_numpy(dtype=float)
        label_suffix = ""
    return x, y, label_suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot publication-quality FRC curves for CVDM vs ThunderSTORM.")
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--cvdm_csv", default=None, help="Path to FRC-CVDM.csv")
    parser.add_argument("--thunder_csv", default=None, help="Path to FRC-ThunderSTORM.csv")
    parser.add_argument("--out_png", default=None, help="Output PNG path")
    parser.add_argument("--out_pdf", default=None, help="Optional output PDF path")
    parser.add_argument("--title", default="Fourier Ring Correlation", help="Plot title")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300)")
    parser.add_argument("--pixel_size_nm", type=float, default=None, help="Physical pixel size in nm used to convert Spatial_Frequency from px^-1 to nm^-1")
    parser.add_argument("--use_smoothed", action="store_true", help="Use Smoothed_FRC if available")
    parser.add_argument("--show_threshold", action="store_true", help="Draw threshold line if 'Threshold' column exists")
    args = parser.parse_args()

    cfg = {}
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

    cvdm_csv = args.cvdm_csv if args.cvdm_csv is not None else cfg.get("cvdm_csv")
    thunder_csv = args.thunder_csv if args.thunder_csv is not None else cfg.get("thunder_csv")
    out_png_arg = args.out_png if args.out_png is not None else cfg.get("out_png")
    out_pdf_arg = args.out_pdf if args.out_pdf is not None else cfg.get("out_pdf", None)
    title = args.title if args.title != "Fourier Ring Correlation" else cfg.get("title", "Fourier Ring Correlation")
    dpi = int(args.dpi) if args.dpi != 300 else int(cfg.get("dpi", 300))
    pixel_size_nm = (
        float(args.pixel_size_nm)
        if args.pixel_size_nm is not None
        else float(cfg.get("pixel_size_nm", 1.0))
    )
    use_smoothed = bool(args.use_smoothed or cfg.get("use_smoothed", False))
    show_threshold = bool(args.show_threshold or cfg.get("show_threshold", False))

    if not cvdm_csv or not thunder_csv or not out_png_arg:
        raise ValueError("Provide cvdm_csv, thunder_csv, and out_png via --config or CLI arguments")

    cvdm_path = Path(cvdm_csv)
    thunder_path = Path(thunder_csv)
    out_png = Path(out_png_arg)
    out_pdf = Path(out_pdf_arg) if out_pdf_arg else None

    df_cvdm = _read_frc_csv(cvdm_path)
    df_thunder = _read_frc_csv(thunder_path)

    x_cvdm = df_cvdm["Spatial_Frequency"].to_numpy(dtype=float)
    y_cvdm_raw = df_cvdm["FRC"].to_numpy(dtype=float)
    y_cvdm_smooth = df_cvdm["Smoothed_FRC"].to_numpy(dtype=float) if "Smoothed_FRC" in df_cvdm.columns else None

    x_thunder = df_thunder["Spatial_Frequency"].to_numpy(dtype=float)
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be > 0")
    x_cvdm = x_cvdm / pixel_size_nm
    x_thunder = x_thunder / pixel_size_nm

    y_thunder_raw = df_thunder["FRC"].to_numpy(dtype=float)
    y_thunder_smooth = (
        df_thunder["Smoothed_FRC"].to_numpy(dtype=float) if "Smoothed_FRC" in df_thunder.columns else None
    )

    style_candidates = [
        "seaborn-v0_8-whitegrid",
        "seaborn-whitegrid",
        "ggplot",
        "default",
    ]
    for style_name in style_candidates:
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    fig, ax = plt.subplots(figsize=(7.8, 4.2))

    ax.plot(
        x_cvdm,
        y_cvdm_raw,
        color="#C62828",
        linewidth=1.2,
        alpha=0.45,
        label="CVDM (raw)",
    )
    ax.plot(
        x_thunder,
        y_thunder_raw,
        color="#1565C0",
        linewidth=1.2,
        alpha=0.45,
        label="ThunderSTORM (raw)",
    )

    if y_cvdm_smooth is not None:
        ax.plot(
            x_cvdm,
            y_cvdm_smooth,
            color="#C62828",
            linewidth=2.4,
            alpha=1.0,
            label="CVDM (smoothed)",
        )
    if y_thunder_smooth is not None:
        ax.plot(
            x_thunder,
            y_thunder_smooth,
            color="#1565C0",
            linewidth=2.4,
            alpha=1.0,
            label="ThunderSTORM (smoothed)",
        )

    threshold_candidates = []
    if "Threshold" in df_cvdm.columns and len(df_cvdm["Threshold"]) > 0:
        threshold_candidates.append(float(df_cvdm["Threshold"].iloc[0]))
    if "Threshold" in df_thunder.columns and len(df_thunder["Threshold"]) > 0:
        threshold_candidates.append(float(df_thunder["Threshold"].iloc[0]))

    threshold_value = float(np.nanmean(threshold_candidates)) if threshold_candidates else None
    if threshold_value is not None and np.isfinite(threshold_value):
        if show_threshold:
            ax.axhline(
                threshold_value,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"Threshold={threshold_value:.3f}",
            )

        y_cvdm_cross = y_cvdm_smooth if y_cvdm_smooth is not None else y_cvdm_raw
        y_thunder_cross = y_thunder_smooth if y_thunder_smooth is not None else y_thunder_raw

        freq_cross_cvdm = _crossing_frequency(x_cvdm, y_cvdm_cross, threshold_value)
        freq_cross_thunder = _crossing_frequency(x_thunder, y_thunder_cross, threshold_value)

        if np.isfinite(freq_cross_cvdm):
            ax.axvline(freq_cross_cvdm, color="#C62828", linestyle=":", linewidth=1.4, alpha=0.95)
        if np.isfinite(freq_cross_thunder):
            ax.axvline(freq_cross_thunder, color="#1565C0", linestyle=":", linewidth=1.4, alpha=0.95)

        res_cvdm = _resolution_nm(freq_cross_cvdm)
        res_thunder = _resolution_nm(freq_cross_thunder)

        note = []
        if np.isfinite(res_cvdm):
            note.append(f"CVDM: {res_cvdm:.1f} nm")
        if np.isfinite(res_thunder):
            note.append(f"ThunderSTORM: {res_thunder:.1f} nm")
        if note:
            fig.text(
                0.76,
                0.12,
                "\n".join(note),
                ha="left",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
            )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(r"Spatial frequency (nm$^{-1}$)", fontsize=11)
    ax.set_ylabel("FRC", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(right=0.72)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    if out_pdf is not None:
        print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
