import torch


class AbstractAlgorithm:
    def __init__(self, policy, policy_optim, **kwargs):
        super(AbstractAlgorithm, self).__init__(**kwargs)
        self._policy = policy
        self._policy_optim = policy_optim

    def before_train(self, observations, actions, rewards):
        pass

    def step(self, observations, actions, rewards):
        raise NotImplementedError


class VanillaPolicyGradient(AbstractAlgorithm):
    """Vanilla policy gradient with reward-to-go."""

    def weights(self, observations, actions, rewards):
        return rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])

    def step(self, observations, actions, rewards):
        weights = self.weights(observations, actions, rewards)
        performance = -(self._policy(observations[:-1]).log_prob(actions) * weights).sum(dim=0).mean(dim=0)
        performance.backward()
        self._policy_optim.step()
        self._policy_optim.zero_grad()


class AdvantageActorCritic(VanillaPolicyGradient):
    def __init__(self, value, value_optim, pre_value_iters, value_iters, **kwargs):
        super(AdvantageActorCritic, self).__init__(**kwargs)
        self._value = value
        self._value_optim = value_optim
        self._pre_value_iters = pre_value_iters
        self._value_iters = value_iters

    def _train_value(self, observations, rewards, iters):
        reward_to_go = rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])
        for _ in range(iters):
            value = self._value(observations)  # shape (time, batch_size)
            loss = torch.nn.functional.mse_loss(value[1:], reward_to_go)
            loss.backward()
            self._value_optim.step()
            self._value_optim.zero_grad()
        return value.detach()

    def before_train(self, observations, actions, rewards):
        self._value.before_train(observations, actions, rewards)
        self._train_value(observations, rewards, self._pre_value_iters)

    def weights(self, observations, actions, rewards):
        value = self._train_value(observations, rewards, self._value_iters)
        return rewards + value[1:] - value[:-1]


class ProximalPolicyOptimisation(AbstractAlgorithm):
    pass
