import torch

from src import envs, algorithms, core


class Policy(torch.nn.Module):
    def __init__(self, obs_size, num_actions, mlp_size, num_layers):
        super(Policy, self).__init__()
        self._network = core.mlp(obs_size, num_actions, mlp_size, num_layers)

    def forward(self, obs):
        return torch.distributions.Categorical(logits=self._network(obs))


class Value(torch.nn.Module):
    def __init__(self, obs_size, mlp_size, num_layers):
        super(Value, self).__init__()
        self._network = core.mlp(obs_size, 1, mlp_size, num_layers)

    def forward(self, obs):
        return self._network(obs).squeeze(-1)


def main():
    # Training hyperparameters
    device = 'cuda'
    epochs = 200
    batch_size = 4096
    lr = 1e-3
    # Architecture hyperparameters
    mlp_size = 64
    num_layers = 1

    env = envs.CartPole(device=device)
    policy = Policy(env.obs_size, env.num_actions, mlp_size, num_layers).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    algorithm = algorithms.VanillaPolicyGradient(policy=policy)

    observations = core.train(env, policy, optim, algorithm, epochs, batch_size)

    env.render(observations[:, 0])


if __name__ == '__main__':
    main()
