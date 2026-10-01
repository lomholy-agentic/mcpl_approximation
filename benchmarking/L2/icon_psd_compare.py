"""Compare the sample PSD from the MCPL run and the PyTorch-model run of icon_sample.instr.

Usage:
    python icon_psd_compare.py <mcpl_run_dir> <torch_run_dir> <output_dir> [torch_scale]

torch_scale multiplies the PyTorch PSD in the "scaled" difference. The raw
difference is always written as well.
"""
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_psd(run_dir, name="image.dat"):
    """Return (I, I_err, N, xylimits) from a McStas 2D monitor file."""
    path = os.path.join(run_dir, name)
    with open(path) as f:
        lines = f.read().splitlines()
    n = None
    xylim = None
    blocks = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# type: array_2d"):
            n = [int(v) for v in line[line.index("(") + 1 : line.index(")")].split(",")]
        if line.startswith("# xylimits:"):
            xylim = [float(v) for v in line.split(":")[1].split()]
        if line.startswith("# Data [") or line.startswith("# Errors [") or line.startswith("# Events ["):
            key = line.split("]")[1].strip().rstrip(":")
            key = line.split()[1]
            rows = [
                [float(v) for v in lines[i + 1 + r].split()] for r in range(n[1])
            ]
            blocks[key] = np.array(rows)
            i += n[1]
        i += 1
    return blocks["Data"], blocks["Errors"], blocks["Events"], xylim


def plot_image(img, xylim, title, filename, cmap="viridis", symmetric=False, label="I"):
    fig, ax = plt.subplots()
    vmax = np.abs(img).max() if symmetric else img.max()
    vmin = -vmax if symmetric else 0
    im = ax.imshow(
        img,
        origin="lower",
        extent=xylim,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(xlabel="x [cm]", ylabel="y [cm]", title=title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mcpl_dir, torch_dir, out_dir = sys.argv[1:4]
    torch_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    I_m, E_m, N_m, xylim = load_psd(mcpl_dir)
    I_t, E_t, N_t, _ = load_psd(torch_dir)

    shutil.copy(os.path.join(mcpl_dir, "image.dat"), os.path.join(out_dir, "ICON_PSD_mcpl.dat"))
    shutil.copy(os.path.join(torch_dir, "image.dat"), os.path.join(out_dir, "ICON_PSD_torch.dat"))

    plot_image(I_m, xylim, "PSD: MCPL source", os.path.join(out_dir, "ICON_PSD_mcpl.png"))
    plot_image(I_t, xylim, "PSD: PyTorch CFM source", os.path.join(out_dir, "ICON_PSD_torch.png"))

    diff_raw = I_m - I_t
    diff_scaled = I_m - torch_scale * I_t
    np.savetxt(os.path.join(out_dir, "ICON_PSD_difference.dat"), diff_raw)
    np.savetxt(os.path.join(out_dir, "ICON_PSD_difference_scaled.dat"), diff_scaled)
    plot_image(
        diff_raw, xylim, "PSD difference: MCPL - PyTorch (raw)",
        os.path.join(out_dir, "ICON_PSD_difference.png"),
        cmap="RdBu_r", symmetric=True, label="dI",
    )
    plot_image(
        diff_scaled, xylim, f"PSD difference: MCPL - {torch_scale:.4g} x PyTorch",
        os.path.join(out_dir, "ICON_PSD_difference_scaled.png"),
        cmap="RdBu_r", symmetric=True, label="dI",
    )

    err = np.sqrt(E_m**2 + (torch_scale * E_t) ** 2)
    print(f"total I  mcpl            : {I_m.sum():.6g}")
    print(f"total I  torch (raw)     : {I_t.sum():.6g}")
    print(f"total I  torch (scaled)  : {torch_scale * I_t.sum():.6g}")
    print(f"raw ratio torch/mcpl     : {I_t.sum() / I_m.sum():.6g}")
    print(f"scaled ratio torch/mcpl  : {torch_scale * I_t.sum() / I_m.sum():.6g}")
    print(f"scaled L2 sum(dI^2)      : {np.sum(diff_scaled**2):.6g}")
    mask = err > 0
    print(f"scaled chi2/pixel        : {np.sum((diff_scaled[mask] / err[mask]) ** 2) / mask.sum():.4g}")
    ny, nx = I_m.shape
    for (iy, ix) in [(ny // 2, nx // 2), (ny // 2, nx // 2 + 20), (ny // 2 + 20, nx // 2), (ny // 2 - 20, nx // 2 - 20)]:
        print(f"pixel ({iy},{ix}): mcpl {I_m[iy, ix]:.5g}  torch(scaled) {torch_scale * I_t[iy, ix]:.5g}  err {err[iy, ix]:.3g}")
