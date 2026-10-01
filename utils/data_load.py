# Author: Daniel Lomholt Christensen
import mcpl
import numpy as np
import torch
from torch import nn
import np2mcpl
import math
import onnx


def load_mcpl_file(
    filepath: str, n_particles: int = 0, offset: int = 0, print_status: bool = False
):
    mcplfile = mcpl.MCPLFile(filepath)
    max_p = mcplfile.nparticles
    if n_particles == 0:
        print(f"load_mcpl_file: Loading all {max_p} neutrons in mcpl file")
        n_particles = max_p
    if max_p < n_particles:
        print(
            f"load_mcpl_file: Warning! Requested {n_particles} neutrons,"
            + f" but there were only {max_p} neutrons available."
        )
        print(f"Loading {max_p} neutrons.")
        n_particles = max_p

    data = np.zeros([n_particles, 12], dtype=np.float32)

    for i, p in enumerate(mcplfile.particles):
        if i < offset:
            continue
        if i % 10000 == 0 and print_status == True:
            print(f"Loading particles. {i - offset} currently loaded.")
        j = i - offset
        data[j : (j + 1), 0] = np.asarray(p.weight)
        data[j : (j + 1), 1] = np.asarray(p.ekin)
        data[j : (j + 1), 2] = np.asarray(p.time)
        data[j : (j + 1), 3:6] = np.asarray([p.ux, p.uy, p.uz]).T
        data[j : (j + 1), 6:9] = np.asarray([p.x, p.y, p.z]).T
        data[j : (j + 1), 9:12] = np.asarray([p.polx, p.poly, p.polz]).T
        if j >= n_particles + offset:
            break
    data = torch.tensor(data, dtype=torch.float32)
    return data


def load_mcpl_file_random(filepath: str, n_particles: int, seed: int = 0):
    """Load a random subset (without replacement) of n_particles from an MCPL
    file, streaming the file block by block."""
    mcplfile = mcpl.MCPLFile(filepath)
    max_p = mcplfile.nparticles
    n_particles = min(int(n_particles), max_p)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(max_p, n_particles, replace=False))

    data = np.zeros([n_particles, 12], dtype=np.float32)
    start = 0
    filled = 0
    for b in mcplfile.particle_blocks:
        n_block = len(b.weight)
        lo = np.searchsorted(chosen, start)
        hi = np.searchsorted(chosen, start + n_block)
        local = chosen[lo:hi] - start
        k = len(local)
        data[filled : filled + k, 0] = b.weight[local]
        data[filled : filled + k, 1] = b.ekin[local]
        data[filled : filled + k, 2] = b.time[local]
        data[filled : filled + k, 3] = b.ux[local]
        data[filled : filled + k, 4] = b.uy[local]
        data[filled : filled + k, 5] = b.uz[local]
        data[filled : filled + k, 6] = b.x[local]
        data[filled : filled + k, 7] = b.y[local]
        data[filled : filled + k, 8] = b.z[local]
        data[filled : filled + k, 9] = b.polx[local]
        data[filled : filled + k, 10] = b.poly[local]
        data[filled : filled + k, 11] = b.polz[local]
        filled += k
        start += n_block
    return torch.tensor(data, dtype=torch.float32)


def dim_reduction(data):
    out = torch.zeros((data.shape[0], 6))
    out[:, 0:7] = data[:, 0:7]
    return out


def preprocess(data):
    data = dim_reduction(data)

    eps = 1e-12
    data[:, 0] = torch.log(data[:, 0] + eps)  # weight
    data[:, 1] = torch.log(data[:, 1] + eps)  # energy

    # Make sure angles are bounded between + pi and +- pi
    theta = data[:, 2]
    phi = data[:, 3]
    theta = torch.clamp(theta / torch.pi, eps, 1 - eps)
    phi = torch.clamp((phi + torch.pi) / 2 / torch.pi, eps, 1 - eps)
    theta = torch.logit(theta, eps=eps)
    phi = torch.logit(phi, eps=eps)

    data[:, 2] = theta
    data[:, 3] = phi

    # For positions x,y:
    xmin = data[:, 4].min()
    xmax = data[:, 4].max()
    ymin = data[:, 5].min()
    ymax = data[:, 5].max()
    dx = xmax - xmin
    dy = ymax - ymin
    # Use lower eps for clamps
    eps = 1e-7
    x = torch.clamp((data[:, 4] - xmin) / dx, eps, 1 - eps)
    y = torch.clamp((data[:, 5] - ymin) / dy, eps, 1 - eps)

    data[:, 4] = torch.logit(x, eps=eps)
    data[:, 5] = torch.logit(y, eps=eps)

    return data, xmin, ymin, dx, dy


# ==============================================================================
# =================     IMPLEMENT RANK GAUSSIAN SCALING ========================
# ==============================================================================
SQRT2 = math.sqrt(2.0)


def save_transform_binary(path, grid, sorted_cols):
    grid = np.asarray(grid, dtype=np.float64)
    sorted_cols = np.asarray(sorted_cols, dtype=np.float64)

    n = grid.shape[0]
    d = sorted_cols.shape[0]

    assert sorted_cols.shape == (d, n)

    with open(path, "wb") as f:
        np.array([n], dtype=np.int32).tofile(f)
        np.array([d], dtype=np.int32).tofile(f)
        grid.tofile(f)
        sorted_cols.tofile(f)


def load_transform_binary(path):
    with open(path, "rb") as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]

        grid = np.fromfile(f, dtype=np.float64, count=n)
        sorted_cols = np.fromfile(f, dtype=np.float64, count=n * d)

    sorted_cols = sorted_cols.reshape(d, n)

    return grid, sorted_cols


def normal_icdf_np(u):
    u = np.asarray(u, dtype=np.float64)
    u = np.clip(u, 1e-7, 1.0 - 1e-7)

    t = torch.from_numpy(2.0 * u - 1.0)
    return (SQRT2 * torch.erfinv(t)).numpy()


def normal_cdf_np(z):
    z = torch.as_tensor(z, dtype=torch.float64)
    return (0.5 * (1.0 + torch.erf(z / SQRT2))).numpy()


def get_transformer_limits(file_path="gaussian_transformer.bin"):
    grid, sorted_cols = load_transform_binary(file_path)
    lims = (sorted_cols[:, 0], sorted_cols[:, -1])

    return lims, grid, sorted_cols


def transform(data, jitter=1e-12, file_path="gaussian_transformer.bin"):
    rng = np.random.default_rng()
    data = np.asarray(data, dtype=np.float32)

    n, d = data.shape

    grid = (np.arange(n, dtype=np.float32) + 0.5) / n
    sorted_cols = []

    output = np.zeros((n, d), dtype=np.float32)

    for j in range(d):
        col = data[:, j]

        scale = np.std(col)
        noise = rng.normal(0.0, jitter * max(scale, 1.0), size=n)

        order = np.argsort(col + noise, kind="mergesort")

        ranks = np.zeros(n, dtype=np.float32)
        ranks[order] = np.arange(n, dtype=np.float32)

        u = (ranks + 0.5) / n

        output[:, j] = normal_icdf_np(u).astype(np.float32)
        sorted_cols.append(np.sort(col).astype(np.float32))
    save_transform_binary(file_path, grid, sorted_cols)
    output = torch.asarray(output)
    torch.save(output, "../../data_files/samples/gaussian_input.pkl")
    return output


def inverse_transform(data, file_path="gaussian_transformer.bin"):
    grid, sorted_cols = load_transform_binary(file_path)
    data = np.asarray(data, dtype=np.float64)
    u = normal_cdf_np(data)

    n, d = data.shape
    output = np.zeros((n, d), dtype=np.float32)

    lo = grid[0]
    hi = grid[-1]

    for j in range(d):
        uj = np.clip(u[:, j], lo, hi)
        output[:, j] = np.interp(uj, grid, sorted_cols[j])

    return output


# ==============================================================================
# ==============================================================================
# ==============================================================================


def normalize(data):
    # Normalizes between 0 and 1
    xmin = data.min()
    xmax = data.max()
    dx = xmax - xmin
    data = (data - xmin) / dx
    return data, xmin, dx


def preprocess_nn(data):
    data = dim_reduction(data)
    mins = np.zeros((data.shape[1]))
    dxs = np.zeros((data.shape[1]))
    data[:, 0] = torch.log(data[:, 0] * 1e20 + 1)  # weight
    data[:, 1] = torch.log(data[:, 1] * 1e10 + 1)  # energy
    for i in range(data.shape[1]):
        data[:, i], mins[i], dxs[i] = normalize(data[:, i])
    return data, mins, dxs


def postprocess(data, xmin, ymin, dx, dy):
    output = torch.zeros((data.shape[0], 7))
    output[:, 0] = torch.exp(data[:, 0])
    output[:, 1] = torch.exp(data[:, 1])
    theta = torch.special.expit(data[:, 2])
    theta *= torch.pi
    phi = torch.special.expit(data[:, 3])
    phi *= 2 * torch.pi
    phi -= torch.pi
    data[:, 2] = theta
    data[:, 3] = phi
    output[:, 2] = np.cos(data[:, 3]) * np.sin(data[:, 2])
    output[:, 3] = np.sin(data[:, 3]) * np.sin(data[:, 2])
    output[:, 4] = np.cos(data[:, 2])

    x = torch.special.expit(data[:, 4])
    y = torch.special.expit(data[:, 5])
    x *= dx
    y *= dy
    x += xmin
    y += ymin

    output[:, 5] = x
    output[:, 6] = y
    return output


def inverse_norm(data, min, dx):
    data = data * dx + min
    return data


def postprocess_nn(data, mins=None, dxs=None, filename=""):
    if filename != "":
        mins, dxs = np.load(filename)
    for i in range(data.shape[1]):
        data[:, i] = inverse_norm(data[:, i], mins[i], dxs[i])
    output = torch.zeros((data.shape[0], 7))
    output[:, 0] = (np.exp(data[:, 0]) - 1) / 1e20
    output[:, 1] = (np.exp(data[:, 1]) - 1) / 1e10

    output[:, 2] = np.cos(data[:, 3]) * np.sin(data[:, 2])
    output[:, 3] = np.sin(data[:, 3]) * np.sin(data[:, 2])
    output[:, 4] = np.cos(data[:, 2])
    output[:, 5:] = data[:, 4:]

    return output


def save_data_as_mcpl(data, filename):
    # Convert the input data which is 6d, to np2mcpl data, which is 10 d
    output = np.zeros((data.shape[0], 10), dtype=np.float32)
    output[:, 0] = 2112  # PDG code
    output[:, 1] = data[:, 5]  # Position in xy
    output[:, 2] = data[:, 6]  # Position in xy
    output[:, 3] = 6040  # Position in Z
    # Convert angles into direction vector
    output[:, 4] = data[:, 2]
    output[:, 5] = data[:, 3]
    output[:, 6] = data[:, 4]

    output[:, 7] = 1  # Time is irrelevant here
    output[:, 8] = data[:, 1]  # Energy
    output[:, 9] = data[:, 0]  # Weight

    np2mcpl.save(filename, output)


def export_model_as_onnx(input_model: nn.Module, onnx_path: str, device: str):
    input_model = input_model.to(device=device)
    input_model.eval()
    dummy_input = torch.randn(10000, 12).to(device)
    # Export ONNX with external weights
    prog = torch.onnx.export(
        input_model,
        dummy_input,
        external_data=False,
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        dynamic_shapes={
            "x": {0: "batch"},
        },
    )
    prog.save(onnx_path, external_data=True)

    # Make a metadata file
    model = onnx.load(onnx_path)
    meta1 = onnx.StringStringEntryProto()
    meta1.key = "author"
    meta1.value = "Daniel Lomholt Christensen"
    meta2 = onnx.StringStringEntryProto()
    meta2.key = "n_training_samples"
    meta2.value = str(1000000)
    model.metadata_props.extend([meta1, meta2])
    onnx.save_model(
        model,
        onnx_path,
    )
    print("✅ Export complete!")


def export_model_as_torchscript(
    input_model: nn.Module,
    torch_path: str,
    device: str,
    n_training_samples: int = 1_000_000,
):
    input_model = input_model.to(device=device)
    input_model.eval()
    scripted_module = torch.jit.script(input_model)
    # TorchScript's analog of ONNX's model.metadata_props: named byte blobs
    # embedded in the file, read back via torch::jit::load(path, device, extra_files).
    torch.jit.save(
        scripted_module,
        torch_path,
        _extra_files={"n_training_samples": str(n_training_samples)},
    )
    print("✅ Export complete!")
