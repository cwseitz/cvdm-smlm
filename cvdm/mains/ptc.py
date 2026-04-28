import argparse
import os

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import yaml


def compute_frame_stats(stack: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    means = []
    variances = []
    for frame in stack:
        values = frame.astype(float) - offset
        means.append(float(np.mean(values)))
        variances.append(float(np.var(values, ddof=1)))
    return np.asarray(means), np.asarray(variances)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple PTC scatter + linear fit from a TIFF stack.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    stack_path = config["stack"]
    out_dir = config["out_dir"]
    prefix = config.get("prefix", "ptc")
    offset = float(config.get("offset", 0.0))
    fixed_intercept = config.get("fixed_intercept_adu", None)
    log_log = bool(config.get("log_log", False))
    if fixed_intercept is not None:
        fixed_intercept = float(fixed_intercept)

    os.makedirs(out_dir, exist_ok=True)

    stack = tifffile.imread(stack_path)
    if stack.ndim != 3:
        raise ValueError("Expected a 3D TIFF stack (frames, height, width)")

    means, variances = compute_frame_stats(stack, offset)

    if len(means) < 2:
        raise ValueError("Need at least two frames to fit PTC")

    if log_log:
        if fixed_intercept is not None:
            raise ValueError("fixed_intercept_adu is not compatible with log-log fitting")
        mask = (means > 0) & (variances > 0)
        means = means[mask]
        variances = variances[mask]
        if len(means) < 2:
            raise ValueError("Need at least two positive mean/variance points for log-log fit")
        log_means = np.log10(means)
        log_variances = np.log10(variances)
        coeffs = np.polyfit(log_means, log_variances, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        xfit = np.linspace(log_means.min(), log_means.max(), 200)
        yfit = slope * xfit + intercept
        fit_label = f"log-log fit: slope={slope:.3f}, log10_intercept={intercept:.3f}"
    else:
        if fixed_intercept is None:
            coeffs = np.polyfit(means, variances, 1)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            fit_label = f"fit: gain={slope:.3f}, read_noise_var={intercept:.3f}"
        else:
            x = means
            y = variances - fixed_intercept
            denom = float(np.sum(x * x))
            if denom <= 0:
                raise ValueError("Cannot fit gain with fixed intercept: zero mean energy")
            slope = float(np.sum(x * y) / denom)
            intercept = float(fixed_intercept)
            fit_label = f"fit: gain={slope:.3f}, fixed_read_noise_var={intercept:.3f}"

        xfit = np.linspace(means.min(), means.max(), 200)
        yfit = slope * xfit + intercept

    fig, ax = plt.subplots(figsize=(6, 5))
    if log_log:
        ax.plot(log_means, log_variances, "o", ms=3, alpha=0.6, label="frame samples")
        ax.plot(xfit, yfit, "-", label=fit_label)
        ax.set_xlabel("log10(Mean)")
        ax.set_ylabel("log10(Variance)")
    else:
        ax.plot(means, variances, "o", ms=3, alpha=0.6, label="frame samples")
        ax.plot(xfit, yfit, "-", label=fit_label)
        ax.set_xlabel("Mean (ADU)")
        ax.set_ylabel("Variance (ADU^2)")
    ax.set_title("PTC (simple)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plot_path = os.path.join(out_dir, f"{prefix}-ptc.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    csv_path = os.path.join(out_dir, f"{prefix}-ptc.csv")
    np.savetxt(csv_path, np.column_stack([means, variances]), delimiter=",", header="mean,variance", comments="")

    print(f"Saved PTC plot: {plot_path}")
    print(f"Saved PTC CSV: {csv_path}")
    if log_log:
        print({"log_slope": slope, "log10_intercept": intercept})
    else:
        print({"gain": slope, "read_noise_var": intercept})


if __name__ == "__main__":
    main()
