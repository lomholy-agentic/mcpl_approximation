"""L2 benchmark of the PSI_ICON MCPL file against the trained PyTorch (CFM) model.

For each particle count n the MCPL side runs icon_sample.instr on a random
subset of n neutrons (weights rescaled by N_total / n, as in mcpl_split.py),
and the PyTorch side runs the Source_ML_torch component directly with ncount=n,
so no MCPL file is written for the model. The loss is sum((I_ref - I)^2) over
the sample PSD, with I_ref the PSD of the full MCPL file.

Source_ML_torch scales every weight by ncount / n_training_samples. For a model
trained on unscaled file weights the PSD of the n-neutron run is therefore
multiplied by n / n_train relative to the weight sum of n neutrons drawn from
the model. The PyTorch PSD is divided by that factor and multiplied by
N_total / n, so that its total weight is the weight sum of the full file.

Usage (from benchmarking/L2, with MCSTAS_CC_OVERRIDE / CONDA_PREFIX set as in
mcstas_comps/torchscript/README.md):
    python icon_l2_benchmark.py --ref-dir <full MCPL run dir> --out-dir ~/Desktop
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import np2mcpl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.append(str(REPO / "utils"))
sys.path.append(str(HERE))
from data_load import load_mcpl_file_random  # noqa: E402
from icon_psd_compare import load_psd  # noqa: E402

TORCH_COMP_DIR = REPO / "mcstas_comps" / "torchscript"


def make_torch_instr(src, dst):
    """Write a copy of icon_sample.instr without the MCPL source, so that its
    INITIALIZE does not override ncount."""
    text = Path(src).read_text()
    text = re.sub(r"COMPONENT MCPL_source = MCPL_input\(.*?ABSOLUTE\n\n", "", text, flags=re.S)
    text = re.sub(r'string run_from_mcpl = ".*?",\n', "", text)
    text = re.sub(r"int use_ml = 0,\n", "", text)
    text = text.replace(" WHEN (use_ml)", "")
    text = text.replace("icon_sample", "icon_sample_torch")
    Path(dst).write_text(text)


def write_mcpl(data, name):
    out = np.zeros((len(data), 10), dtype=np.float32)
    out[:, 0] = 2112
    out[:, 1:4] = data[:, 6:9]
    out[:, 4:7] = data[:, 3:6]
    out[:, 7] = data[:, 2]
    out[:, 8] = data[:, 1]
    out[:, 9] = data[:, 0]
    np2mcpl.save(name, out)


def build(instr, work, params):
    work.mkdir(parents=True, exist_ok=True)
    for f in ("Source_ML_torch.comp", "torch_wrap.h"):
        shutil.copy(TORCH_COMP_DIR / f, work / f)
    shutil.copy(instr, work / instr.name)
    cmd = ["mcrun", "-c", instr.name, "-n", "10", "-d", "build_run"] + params
    # mcrun also runs the binary once, which can fail on string-parameter
    # quoting; only the compiled binary is needed here.
    subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    binary = work / (instr.stem + ".out")
    if not binary.exists():
        raise RuntimeError(f"Compilation of {instr} failed")
    return binary


def run(binary, n, outdir, params, cwd):
    if Path(outdir).exists():
        shutil.rmtree(outdir)
    t0 = time.time()
    subprocess.run(
        [str(binary), "-n", str(int(n)), "-d", str(outdir)] + params,
        cwd=cwd, check=True, capture_output=True, text=True,
    )
    return time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default=str(HERE / "icon_work"))
    ap.add_argument("--mcpl", default=str(REPO / "data_files/mcpl_files/PSI_ICON.mcpl.gz"))
    ap.add_argument("--model", default=str(REPO / "data_files/models/CFM_sampler.pt"))
    ap.add_argument("--transformer", default=str(REPO / "data_files/preprocess/gaussian_transformer.bin"))
    ap.add_argument("--n-train", type=int, default=1_000_000)
    ap.add_argument("--n-sizes", type=int, default=14)
    ap.add_argument("--max-size", type=int, default=30_000_000)
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)

    ref_I, _, _, _ = load_psd(args.ref_dir)

    full = load_mcpl_file_random(args.mcpl, 10**9).numpy()
    n_total = len(full)
    print(f"Loaded {n_total} neutrons", flush=True)

    # Build the two instruments
    small = work / "build_small"
    write_mcpl(full[:2000], str(work / "build_small"))
    mcpl_bin = build(HERE / "icon_sample.instr", work / "mcpl_build",
                     [f"run_from_mcpl={work}/build_small.mcpl.gz"])
    make_torch_instr(HERE / "icon_sample.instr", work / "icon_sample_torch.instr")
    torch_params = [f"run_ml={args.model}", f"run_transformer={args.transformer}"]
    torch_bin = build(work / "icon_sample_torch.instr", work / "torch_build", torch_params)

    sizes = np.unique(np.geomspace(1_000, args.max_size, args.n_sizes).astype(int))
    rng = np.random.default_rng(0)

    rows = []
    for n in list(sizes) + [n_total]:
        row = {"n": int(n)}

        # MCPL subset (the full file is the reference itself and is skipped)
        if n < n_total:
            idx = rng.choice(n_total, n, replace=False)
            sub = full[idx].copy()
            sub[:, 0] *= n_total / n
            name = work / f"subset_{n}"
            write_mcpl(sub, str(name))
            d = work / f"run_mcpl_{n}"
            row["t_mcpl"] = run(mcpl_bin, 1, d, [f"run_from_mcpl={name}.mcpl.gz"], HERE)
            I, _, _, _ = load_psd(d)
            row["I_mcpl"] = float(I.sum())
            row["loss_mcpl"] = float(np.sum((ref_I - I) ** 2))
            os.remove(f"{name}.mcpl.gz")
            shutil.rmtree(d)

        # PyTorch model through the component
        d = work / f"run_torch_{n}"
        row["t_torch"] = run(torch_bin, n, d, torch_params, HERE)
        I, _, _, _ = load_psd(d)
        scale = (n_total / n) * (args.n_train / n)
        row["I_torch_raw"] = float(I.sum())
        row["I_torch"] = float((scale * I).sum())
        row["loss_torch"] = float(np.sum((ref_I - scale * I) ** 2))
        shutil.rmtree(d)

        rows.append(row)
        print(row, flush=True)

    # Table
    with open(out_dir / "ICON_L2_loss_table.csv", "w") as f:
        f.write("n,I_ref,I_mcpl,I_torch_scaled,I_torch_raw,loss_mcpl,loss_torch\n")
        for r in rows:
            f.write(
                f"{r['n']},{ref_I.sum():.6g},{r.get('I_mcpl', float('nan')):.6g},{r['I_torch']:.6g},"
                f"{r['I_torch_raw']:.6g},{r.get('loss_mcpl', float('nan')):.6g},{r['loss_torch']:.6g}\n"
            )

    # Plot
    fig, ax = plt.subplots()
    m = [r for r in rows if "loss_mcpl" in r]
    ax.plot([r["n"] for r in m], [r["loss_mcpl"] for r in m], "+b", label="MCPL subset (MCPL_input)")
    ax.plot([r["n"] for r in rows], [r["loss_torch"] for r in rows], "+r", label="CFM (Source_ML_torch)")
    ax.axvline(args.n_train, color="purple", linestyle="--", label=f"Trained on {args.n_train:.0e} neutrons")
    ax.axvline(n_total, color="gray", linestyle=":", label=f"Full file ({n_total:.3g} neutrons)")
    ax.set(xscale="log", yscale="log", xlabel="Number of particles [#]", ylabel="L2 loss [sum dI**2]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ICON_L2_loss_comparison.png", dpi=150)
