import os
import argparse
import glob
import numpy as np
import csv
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from models import HyperNetwork, CNF
from PIL import Image
from distributions import get_sampler

matplotlib.use("agg")


parser = argparse.ArgumentParser()
parser.add_argument("--adjoint", action="store_true")
parser.add_argument("--viz", action="store_true")
parser.add_argument("--niters", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_samples", type=int, default=512)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--weight_c1", type=float, default=0.02)
parser.add_argument("--weight_c2", type=float, default=0.0001)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--train_dir", type=str, default="./checkpoints")
parser.add_argument("--results_dir", type=str, default="./results")
parser.add_argument("--log_dir", type=str, default="./logs")
parser.add_argument("--distribution_name", type=str, default="checkerboard")
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def curvature(traj):
    # Return: C1 = \int ||dv/dt||dt, C2 = \int ||d^2v/dt^2||dt
    if isinstance(traj, list):
        traj = torch.stack(traj)
    velocity = traj[1:] - traj[:-1]
    accel = velocity[1:] - velocity[:-1]

    vel_norm = torch.norm(velocity, dim=-1)
    accel_norm = torch.norm(accel, dim=-1)
    C1 = torch.mean(vel_norm)
    C2 = torch.mean(accel_norm)

    return C1, C2


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == "__main__":
    t0 = 0
    t1 = 10
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # model
    func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device),
    )
    loss_meter = RunningAverageMeter()

    target_sampler = get_sampler(args.distribution_name, device)
    log_file = os.path.join(
        args.log_dir,
        "{}_{}_{}.csv".format(args.distribution_name, args.weight_c1, args.weight_c2),
    )
    fields = ["iter", "loss", "NLL", "C1", "C2", "nfe"]

    # if args.train_dir is not None:
    #     if not os.path.exists(args.train_dir):
    #         os.makedirs(args.train_dir)
    #     ckpt_path = os.path.join(args.train_dir, "ckpt.pth")
    #     if os.path.exists(ckpt_path):
    #         checkpoint = torch.load(ckpt_path)
    #         func.load_state_dict(checkpoint["func_state_dict"])
    #         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #         print("Loaded ckpt from {}".format(ckpt_path))

    with open(log_file, "w+") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()

            x, logp_diff_t1 = target_sampler.sample(args.num_samples)

            func.reset_nfe()
            z_t, logp_diff_t = odeint(
                func,
                (x, logp_diff_t1),
                torch.tensor(np.linspace(t1, t0, 11)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method="dopri5",
            )

            nfe_f = func.get_nfe()

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

            logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
            loss_nll = -logp_x.mean(0)
            C1, C2 = curvature(z_t)

            # weight_cur = np.min(
            #     [args.weight_cur * itr / args.niters * 2, args.weight_cur]
            # )
            loss = loss_nll + C1 * args.weight_c1 + C2 * args.weight_c2

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            print(
                f"Iter: {itr}, loss: {loss_meter.avg:.4f}, NLL: {loss_nll.item():.4f}, C1: {C1.item():.8f}, C2: {C2.item():.8f}, nfe: {nfe_f}"
            )

            # Write logs
            log_line = [
                itr,
                loss_meter.avg,
                loss_nll.item(),
                C1.item(),
                C2.item(),
                nfe_f,
            ]
            csvwriter.writerow(log_line)

    # Save checkpoints
    ckpt_path = os.path.join(
        args.train_dir,
        "{}_{}_{}.pth".format(args.distribution_name, args.weight_c1, args.weight_c2),
    )
    torch.save(
        {
            "func_state_dict": func.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )

    if args.viz:
        viz_samples = 30000
        viz_timesteps = 41
        target_sample, _ = target_sampler.sample(viz_samples)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

            z_t_samples, _ = odeint(
                func,
                (z_t0, logp_diff_t0),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method="dopri5",
            )

            # cur = curvature(z_t_samples)
            # print(f"Curvature: {cur[0].item():.8f}")

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

            z_t_density, logp_diff_t = odeint(
                func,
                (z_t1, logp_diff_t1),
                torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method="dopri5",
            )

            # Create plots for each timestep
            for t, z_sample, z_density, logp_diff in zip(
                np.linspace(t0, t1, viz_timesteps),
                z_t_samples,
                z_t_density,
                logp_diff_t,
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                plt.axis("off")
                plt.margins(0, 0)
                fig.suptitle(f"{t:.2f}s")

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title("Target")
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title("Samples")
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title("Log Probability")
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(
                    *target_sample.detach().cpu().numpy().T,
                    bins=300,
                    density=True,
                    range=[[-1.5, 1.5], [-1.5, 1.5]],
                )

                ax2.hist2d(
                    *z_sample.detach().cpu().numpy().T,
                    bins=300,
                    density=True,
                    range=[[-1.5, 1.5], [-1.5, 1.5]],
                )

                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                ax3.tricontourf(
                    *z_t1.detach().cpu().numpy().T,
                    np.exp(logp.detach().cpu().numpy()),
                    200,
                )

                plt.savefig(
                    os.path.join(args.results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                    pad_inches=0.2,
                    bbox_inches="tight",
                )
                plt.close()

            img, *imgs = [
                Image.open(f)
                for f in sorted(
                    glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg"))
                )
            ]
            img.save(
                fp=os.path.join(args.results_dir, "cnf-viz.gif"),
                format="GIF",
                append_images=imgs,
                save_all=True,
                duration=250,
                loop=0,
            )

        print(
            "Saved visualization animation at {}".format(
                os.path.join(args.results_dir, "cnf-viz.gif")
            )
        )
