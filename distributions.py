from typing import Optional
import torch
import torch.distributions as D


class Sampler2D:
    def __init__(self, distribution, device) -> None:
        super(Sampler2D, self).__init__()
        self.distribution = distribution
        self.device = device

    def sample(self, n):
        X = self.distribution.sample(n).to(self.device)
        logp_diff_t1 = torch.zeros(n, 1, dtype=torch.float32).to(self.device)
        return X, logp_diff_t1


class TwoCircles(D.Distribution):
    def __init__(
        self, noise, factor, batch_shape=None, event_shape=None, validate_args=None
    ):
        super().__init__(batch_shape, event_shape, validate_args)
        self.noise = noise
        self.factor = factor
        self.noise_generator = D.MultivariateNormal(
            torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]) * noise
        )

    def sample(self, n):
        n_samples_out = n // 2
        n_samples_in = n - n_samples_out

        linspace_out = torch.linspace(0, 2 * torch.pi, n_samples_out)
        linspace_in = torch.linspace(0, 2 * torch.pi, n_samples_in)
        outer_circ_x = torch.cos(linspace_out)
        outer_circ_y = torch.sin(linspace_out)
        inner_circ_x = torch.cos(linspace_in) * self.factor
        inner_circ_y = torch.sin(linspace_in) * self.factor

        X = torch.stack(
            [
                torch.concatenate([outer_circ_x, inner_circ_x]),
                torch.concatenate([outer_circ_y, inner_circ_y]),
            ],
            dim=0,
        ).transpose(1, 0)
        noise = self.noise_generator.sample((n,))
        X += noise

        return X


class TwoDiracs(D.Distribution):
    def __init__(
        self, loc, noise, batch_shape=None, event_shape=None, validate_args=None
    ):
        super().__init__(batch_shape, event_shape, validate_args)
        self.loc = loc
        self.noise = noise
        if noise > 0.0:
            self.noise_generator = D.MultivariateNormal(
                torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]) * noise
            )

    def sample(self, n):
        n_first = n // 2
        n_second = n - n_first

        X = torch.concatenate(
            [
                torch.tensor(self.loc[0]).view(-1, 2).repeat(n_first, 1),
                torch.tensor(self.loc[1]).view(-1, 2).repeat(n_second, 1),
            ]
        )
        if self.noise > 0.0:
            noise = self.noise_generator.sample((n,))
            X += noise
        return X


class MultipleDiracs(D.Distribution):
    def __init__(
        self, num_diracs, noise, batch_shape=None, event_shape=None, validate_args=None
    ):
        super().__init__(batch_shape, event_shape, validate_args)
        self.num_diracs = num_diracs
        linspace = torch.linspace(0, 2 * torch.pi, self.num_diracs + 1)
        xs = torch.cos(linspace)
        ys = torch.sin(linspace)
        self.X = torch.stack([xs, ys], dim=1)

        self.noise = noise
        if noise > 0.0:
            self.noise_generator = D.MultivariateNormal(
                torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]) * noise
            )

    def sample(self, n):
        sample_idx = torch.randint(0, self.num_diracs, (n,))
        X = self.X[sample_idx]

        if self.noise > 0.0:
            X += self.noise_generator.sample((n,))
        return X


class CheckerBoard(D.Distribution):
    def __init__(
        self,
        shape,
        range=[-1.0, 2.0],
        noise=0.0,
        batch_shape=None,
        event_shape=None,
        validate_args=None,
    ):
        super().__init__(batch_shape, event_shape, validate_args)
        self.noise = noise
        if noise > 0.0:
            self.noise_generator = D.MultivariateNormal(
                torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]) * noise
            )

        n_rows, n_cols = shape
        range_len = range[1] - range[0]
        range_x = torch.arange(range[0], range[1], range_len / (n_cols + 1))
        range_y = torch.arange(range[0], range[1], range_len / (n_rows + 1))

        i = 0
        low_x = []
        high_x = []
        while 2 * i + 1 < range_x.shape[0]:
            low_x.append(range_x[2 * i])
            high_x.append(range_x[2 * i + 1])
            i += 1
        low_x, high_x = torch.tensor(low_x), torch.tensor(high_x)

        i = 0
        low_y = []
        high_y = []
        while 2 * i + 1 < range_y.shape[0]:
            low_y.append(range_y[2 * i])
            high_y.append(range_y[2 * i + 1])
            i += 1
        low_y, high_y = torch.tensor(low_y), torch.tensor(high_y)

        comp_x = D.Independent(D.Uniform(low_x, high_x), 0)
        mix_x = D.Categorical(torch.ones_like(low_x))
        self.x_sampler_1 = D.MixtureSameFamily(mix_x, comp_x)

        comp_y = D.Independent(D.Uniform(low_y, high_y), 0)
        mix_y = D.Categorical(torch.ones_like(low_y))
        self.y_sampler_1 = D.MixtureSameFamily(mix_y, comp_y)


        i = 0
        low_x = []
        high_x = []
        while 2 * i + 2 < range_x.shape[0]:
            low_x.append(range_x[2 * i + 1])
            high_x.append(range_x[2 * i + 2])
            i += 1
        low_x, high_x = torch.tensor(low_x), torch.tensor(high_x)

        i = 0
        low_y = []
        high_y = []
        while 2 * i + 2 < range_y.shape[0]:
            low_y.append(range_y[2 * i + 1])
            high_y.append(range_y[2 * i + 2])
            i += 1
        low_y, high_y = torch.tensor(low_y), torch.tensor(high_y)

        comp_x = D.Independent(D.Uniform(low_x, high_x), 0)
        mix_x = D.Categorical(torch.ones_like(low_x))
        self.x_sampler_2 = D.MixtureSameFamily(mix_x, comp_x)

        comp_y = D.Independent(D.Uniform(low_y, high_y), 0)
        mix_y = D.Categorical(torch.ones_like(low_y))
        self.y_sampler_2 = D.MixtureSameFamily(mix_y, comp_y)

    def sample(self, n):
        n_first = n // 2
        n_second = n - n_first

        sample_first = torch.stack(
            [self.x_sampler_1.sample((n_first,)), self.y_sampler_1.sample((n_first,))], dim=1
        )

        sample_second = torch.stack(
            [self.x_sampler_2.sample((n_second,)), self.y_sampler_2.sample((n_second,))], dim=1
        )

        X = torch.concatenate([sample_first, sample_second], dim=0)
        if self.noise > 0.0:
            X += self.noise_generator((n, ))
        return X


def get_distribution(distribution_name):
    if distribution_name == "two_circles":
        distribution = TwoCircles(0.006, 0.5)
    elif distribution_name == "two_diracs":
        distribution = TwoDiracs([[0.0, 1.0], [0.0, -1.0]], 0.00001)
    elif distribution_name == "checkerboard":
        distribution = CheckerBoard((4, 4))
    elif distribution_name == "multiple_diracs":
        distribution = MultipleDiracs(8, 0.006)
    else:
        raise Exception
    return distribution


def get_sampler(distribution_name, device):
    distribution = get_distribution(distribution_name)
    return Sampler2D(distribution, device)


def main():
    sampler = get_sampler("checkerboard", "cuda")
    X, _ = sampler.sample(30000)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 1)
    axs.hist2d(
        *X.detach().cpu().numpy().transpose(1, 0),
        bins=300,
        density=True,
        range=[[-1.5, 1.5], [-1.5, 1.5]]
    )
    plt.savefig("something.png")


if __name__ == "__main__":
    main()
