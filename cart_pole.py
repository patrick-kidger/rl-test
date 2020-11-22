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
    # General training hyperparameters
    device = 'cuda'
    batch_size = 4096
    policy_lr = 1e-3
    # PPO training hyperparameters
    ppo__value_lr = 1e-3
    ppo__pre_value_iters = 200
    ppo__value_iters = 100
    ppo__policy_iters = 50
    ppo__epochs = 10
    ppo__eps = 0.1
    # VPG training hyperparameters
    vpg__policy_iters = 2
    vpg__epochs = 20
    # Architecture hyperparameters
    mlp_size = 64
    num_layers = 1

    # Set up environment and policy
    env = envs.CartPole(device=device)
    policy = Policy(env.obs_size, env.num_actions, mlp_size, num_layers).to(device)
    policy_optim = torch.optim.Adam(policy.parameters(), lr=policy_lr)

    # Train via PPO -- this gets it most of the way there, but struggles on the last bit.
    value = Value(env.obs_size, mlp_size, num_layers).to(device)
    value_optim = torch.optim.Adam(value.parameters(), lr=ppo__value_lr)
    algorithm = algorithms.ProximalPolicyOptimisation(policy=policy,
                                                      policy_optim=policy_optim,
                                                      policy_iters=ppo__policy_iters,
                                                      value=value,
                                                      value_optim=value_optim,
                                                      pre_value_iters=ppo__pre_value_iters,
                                                      value_iters=ppo__value_iters,
                                                      eps=ppo__eps)
    misc.before_train(env, policy, algorithm, batch_size)
    misc.train(env, policy, algorithm, batch_size, epochs=ppo__epochs)

    # Switch to VPG for the last bit of training
    algorithm = algorithms.VanillaPolicyGradient(policy=policy,
                                                 policy_optim=policy_optim,
                                                 policy_iters=vpg__policy_iters)
    observations = misc.train(env, policy, algorithm, batch_size, epochs=vpg__epochs)

    env.render(observations[:, 0])


if __name__ == '__main__':
    main()
