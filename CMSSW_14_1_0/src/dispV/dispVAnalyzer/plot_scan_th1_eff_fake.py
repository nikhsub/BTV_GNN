#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import uproot
import matplotlib.pyplot as plt


def read_th1(root_obj, name):
    if name not in root_obj:
        raise KeyError(f"Missing histogram '{name}'")
    vals, edges = root_obj[name].to_numpy()
    vals = vals.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, vals


def safe_div(num, den):
    out = np.zeros_like(num, dtype=np.float64)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def make_plot(x, eff, fr, title, xlabel, out_png, out_pdf):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(x, eff, color="red", marker="o", linewidth=1.8, markersize=3, label="efficiency")
    l2 = ax2.plot(x, fr, color="blue", marker="s", linewidth=1.8, markersize=3, label="fake rate / event")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("efficiency", color="red")
    ax2.set_ylabel("#fakes / event", color="blue")
    ax1.set_ylim(0.0, 1.50)
    ax1.tick_params(axis="y", colors="red")
    ax2.tick_params(axis="y", colors="blue")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(title)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot TH1 scan efficiency and fake rate on same canvas.")
    p.add_argument("--input", required=True, help="Input ROOT file")
    p.add_argument("--dir", default="demo", help="TDirectory containing histograms")
    p.add_argument("--outdir", default="scan_th1_plots", help="Output directory")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    flavors = [("B", "B"), ("BtoC", "BtoC"), ("C", "C")]
    scans = [
        ("seed", "SeedSVcut", "scan_seed"),
        ("node", "NodeSVmin", "scan_node"),
        ("edge", "EdgeMin", "scan_edge"),
    ]

    with uproot.open(args.input) as f:
        root_obj = f[args.dir] if args.dir else f

        for scan_key, xlab, prefix in scans:
            x_events, events = read_th1(root_obj, f"{prefix}_events")
            for flab, suffix in flavors:
                x_num, num = read_th1(root_obj, f"{prefix}_num_{suffix}")
                x_den, den = read_th1(root_obj, f"{prefix}_den_{suffix}")
                x_fake, fake = read_th1(root_obj, f"{prefix}_fake_{suffix}")

                if not (np.allclose(x_num, x_den) and np.allclose(x_num, x_fake) and np.allclose(x_num, x_events)):
                    raise RuntimeError(f"Bin mismatch for {prefix} {suffix}")

                eff = safe_div(num, den)
                fr = safe_div(fake, events)

                base = f"{scan_key}_{suffix}"
                make_plot(
                    x_num,
                    eff,
                    fr,
                    title=f"{flab}: efficiency vs fake rate scan ({xlab})",
                    xlabel=xlab,
                    out_png=outdir / f"{base}.png",
                    out_pdf=outdir / f"{base}.pdf",
                )

    print(f"Wrote plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
