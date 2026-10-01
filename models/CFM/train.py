import sys
import os
from model import VelocityField
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("../../utils/")
from data_load import load_mcpl_file, load_mcpl_file_random, transform
import argparse
from tqdm import tqdm
import torch
import numpy as np
import copy

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================


def add_arguments(parser):
    parser.add_argument("--n_particles", default=1e6)
    parser.add_argument("--model_filename", default="../../data_files/models/CFM.pth")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--random_subset", action="store_true")
    parser.add_argument("--input_mcpl", type=str, default= "../../data_files/mcpl_files/ODIN.mcpl.gz")


# ---------------------------------------------------------
# Function to update weights with a moving average
# ---------------------------------------------------------

def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1.0 - decay)


# ---------------------------------------------------------
#  Flow Matching loss
# ---------------------------------------------------------


def flow_matching_loss(model, x0, x1, t):
    # Interpolate path
    xt = (1 - t) * x0 + t * x1

    # True velocity for straight-line FM
    v_target = x1 - x0

    # Model velocity
    v_pred = model(xt, t)
    # Weighted MSE
    loss = (((v_pred - v_target) ** 2).mean(dim=-1)).mean()

    return loss

# ---------------------------------------------------------
#   Sample Time
# ---------------------------------------------------------

def sample_t(batch_size, device):
    mode = torch.rand(batch_size, 1, device=device)

    t_uniform = torch.rand(batch_size, 1, device=device)
    t_near_1 = 1.0 - torch.rand(batch_size, 1, device=device).pow(2)
    t_near_0 = torch.rand(batch_size, 1, device=device).pow(2)

    t = torch.where(
        mode < 0.70,
        t_uniform,
        torch.where(mode < 0.90, t_near_1, t_near_0),
    )

    return t

# ---------------------------------------------------------
#  Training loop
# ---------------------------------------------------------
def train(
        model,
        dataset,
        filename,
        device="mps",
        steps=10_000,
        lr=1e-3,
        val_size=100000,
        batch_size=3000
        ):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    losses = []
    val_losses = []
    # Prepare validation step
    perm = torch.randperm(dataset.shape[0])
    val = dataset[perm[:val_size]].to(device)
    dataset = dataset[perm[val_size:]]
    val_x1 = val
    val_x0 = torch.randn_like(val_x1)
    val_t = torch.rand(val_x1.shape[0], 1, device=device)

    N = dataset.shape[0]
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    for step in tqdm(range(steps)):
        idx = torch.randint(0, N, (batch_size,))
        x0 = torch.randn(batch_size, dataset.shape[1]).to(device)
        x1 = dataset[idx].to(device)
        t = sample_t(batch_size, device)
        loss = flow_matching_loss(model, x0, x1, t)

        # Make the loss, an average of more than 1 step
        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model)
        losses.append(loss.item())

        if step % 1000 == 0:
            # Check how the model performs against validation loss
            ema.eval()

            with torch.no_grad():
                val_loss = flow_matching_loss(ema, val_x0, val_x1, val_t).item()

            val_losses.append([step, val_loss])

            print(f"step {step}: train {loss.item():.5g}, val {val_loss:.5g}")

            if val_loss < best_loss:
                best_val_loss = val_loss

                print(f"Saving best model at step {step}, val loss {best_val_loss:.5g}")

                torch.save(
                    {
                        "state_dict": ema.state_dict(),
                        "step": step,
                        "val_loss": best_val_loss,
                    },
                    filename,
                )

    return losses, val_losses


# ---------------------------------------------------------
# USAGE
# ---------------------------------------------------------


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
input_mcpl = args.input_mcpl

if args.random_subset:
    data = load_mcpl_file_random(input_mcpl, int(args.n_particles))
else:
    data = load_mcpl_file(input_mcpl, int(args.n_particles))
data = torch.asarray(transform(data, file_path="../../data_files/preprocess/gaussian_transformer.bin"), dtype=torch.float32)

dim = data.shape[1]
device = args.device

model = VelocityField().to(device)

losses, val_losses = train(model, data, args.model_filename)

np.save("../../data_files/losses/cfm_train.npy", np.array(losses))
np.save("../../data_files/losses/cfm_val.npy", np.array(val_losses))
