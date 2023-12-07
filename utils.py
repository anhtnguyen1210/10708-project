import os
import torch

from models import CNF


def get_model(
    c1,
    c2,
    distribution_name,
    ckpt_folder="./checkpoints",
    in_out_dim=2,
    hidden_dim=32,
    width=64,
    device="cuda",
):
    file_name = os.path.join(
        ckpt_folder, "{}_{}_{}.pth".format(distribution_name, c1, c2)
    )
    state_dict = torch.load(file_name)
    func = CNF(in_out_dim=in_out_dim, hidden_dim=hidden_dim, width=width).to(device)
    func.load_state_dict(state_dict["func_state_dict"])

    p_z0 = p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device),
    )
    return func, p_z0


def main():
    sooka = get_model(0.0, 0.0, "two_circles")
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
