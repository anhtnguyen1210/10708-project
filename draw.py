import os
import argparse
import glob
import numpy as np
import matplotlib
import torch
import matplotlib.pyplot as plt
import pandas as pd

from models import HyperNetwork, CNF
from distributions import get_sampler
from utils import get_model
from PIL import Image
from cnf import curvature

matplotlib.use("agg")

parser = argparse.ArgumentParser()
parser.add_argument("--adjoint", action="store_true")
parser.add_argument("--num_samples", type=int, default=512)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--weight_c1", type=float, default=1)
parser.add_argument("--weight_c2", type=float, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--train_dir", type=str, default="./oldcheckpoints")
parser.add_argument("--results_dir", type=str, default="./results")
parser.add_argument("--log_dir", type=str, default="./oldlogs")
parser.add_argument("--distribution_name", type=str, default="two_circles")
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def merge_image(imgs, margin=0):
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width + margin * (len(imgs) - 1), max_height))

    x_offset = 0
    for im in imgs:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + margin

    return new_im


def histo_target_distribution(distribution_name, device="cuda", num_points=30000):
    sampler = get_sampler(distribution_name, device)
    X, _ = sampler.sample(num_points)

    fig = plt.figure(figsize=(4, 4), dpi=300)
    plt.tight_layout()
    plt.axis("off")
    plt.margins(0, 0)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.hist2d(
        *X.detach().cpu().numpy().transpose(1, 0),
        bins=300,
        density=True,
        range=[[-1.5, 1.5], [-1.5, 1.5]],
    )
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    plt.savefig("hist_true_{}.png".format(distribution_name), bbox_inches="tight")


def through_time(
    c1, c2, distribution_name, device="cuda", viz_samples=30000, viz_timesteps=41
):
    target_sampler = get_sampler(distribution_name, device)
    target_sample, _ = target_sampler.sample(viz_samples)
    func, p_z0 = get_model(c1, c2, distribution_name, args.train_dir)
    t0 = 0
    t1 = 10

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
            np.linspace(t0, t1, viz_timesteps), z_t_samples, z_t_density, logp_diff_t
        ):
            fig = plt.figure(figsize=(4, 4), dpi=200)

            plt.tight_layout()
            plt.axis("off")
            plt.margins(0, 0)

            ax2 = fig.add_subplot(1, 1, 1)
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])

            ax2.hist2d(
                *z_sample.detach().cpu().numpy().T,
                bins=300,
                density=True,
                range=[[-1.5, 1.5], [-1.5, 1.5]],
            )

            plt.savefig(
                os.path.join(args.results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                pad_inches=0.2,
                bbox_inches="tight",
            )
            plt.close()

        img, *imgs = [
            Image.open(f)
            for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))
        ]

        chosen_images = [imgs[9], imgs[19], imgs[29], imgs[39]]
        file_name = "{}_{}_{}_throughtime.png".format(distribution_name, c1, c2)
        new_img = merge_image(chosen_images)
        new_img.save(file_name)

    

def c1_draw(distribution_name, c1s=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], checkpoints_folder='./oldcheckpoints', num_points=30000, device='cuda'):
    NLLs = []
    curvatures = []
    target_sampler = get_sampler(distribution_name, device)
    t0 = 0.
    t1 = 10.
    for c1 in c1s:
        func, p_z0 = get_model(c1, 0.0, distribution_name, checkpoints_folder)    
        with torch.no_grad():
            x, logp_diff_t1 = target_sampler.sample(num_points)
            func.reset_nfe()
            z_t, logp_diff_t = odeint(
            func,
            (x, logp_diff_t1),
            torch.tensor(np.linspace(t1, t0, 11)).to(device),
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
            )
            C1, C2 = curvature(z_t)
            curvatures.append(C1.item())

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
            logp_x = p_z0.log_prob(z_t0) - logp_diff_t0.view(-1)
            nll = -logp_x.mean(0)
            NLLs.append(nll.item())
    
    # print(NLLs)
    # print(curvatures)
    plt.plot(c1s, NLLs, label='NLL')
    plt.plot(c1s, curvatures, label='C1')
    plt.legend()
    plt.xlabel('weight_c1 - {}'.format(distribution_name))
    plt.savefig('{}.png'.format(distribution_name))
            

def main():
    # through_time(0.0, 0.0, args.distribution_name)
    c1_draw('checkerboard')


if __name__ == "__main__":
    main()
