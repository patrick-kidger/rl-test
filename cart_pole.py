import torch

from src import envs, algorithms, misc


class Policy(torch.nn.Module):
    def __init__(self, obs_size, num_actions, mlp_size, num_layers):
        super(Policy, self).__init__()
        self._network = misc.mlp(obs_size, num_actions, mlp_size, num_layers)

    def forward(self, obs):
        return torch.distributions.Categorical(logits=self._network(obs))


class Value(torch.nn.Module):
    def __init__(self, obs_size, mlp_size, num_layers):
        super(Value, self).__init__()
        self._network = misc.mlp(obs_size, 1, mlp_size, num_layers)

    def forward(self, obs):
        return self._network(obs).squeeze(-1).exp()


def main():
    # Training hyperparameters
    device = 'cuda'
    pre_value_iters = 200
    value_iters = 100
    epochs = 300
    batch_size = 4096
    policy_lr = 1e-3
    value_lr = 1e-3
    # Architecture hyperparameters
    mlp_size = 64
    num_layers = 1

    env = envs.CartPole(device=device)
    policy = Policy(env.obs_size, env.num_actions, mlp_size, num_layers).to(device)
    policy_optim = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    value = Value(env.obs_size, mlp_size, num_layers).to(device)
    value_optim = torch.optim.Adam(value.parameters(), lr=value_lr)
    algorithm = algorithms.AdvantageActorCritic(policy=policy, policy_optim=policy_optim,
                                                value=value, value_optim=value_optim, pre_value_iters=pre_value_iters,
                                                value_iters=value_iters)

    observations = misc.train(env, policy, algorithm, batch_size, epochs)

    env.render(observations[:, 0])


if __name__ == '__main__':
    main()
