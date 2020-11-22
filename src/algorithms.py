import torch


class Algorithm:
    def __init__(self, policy, **kwargs):
        super(Algorithm, self).__init__(**kwargs)
        self._policy = policy

    def before_train(self, observations, actions, rewards):
        pass

    def __call__(self, observations, actions, rewards):
        raise NotImplementedError


class VanillaPolicyGradient(Algorithm):
    """Vanilla policy gradient with reward-to-go."""

    def weights(self, observations, actions, rewards):
        return rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])

    def __call__(self, observations, actions, rewards):
        weights = self.weights(observations, actions, rewards)
        return -(self._policy(observations[:-1]).log_prob(actions) * weights).sum(dim=0).mean(dim=0)


class AdvantageActorCritic(VanillaPolicyGradient):
    def __init__(self, value, pre_epochs, optim, **kwargs):
        super(AdvantageActorCritic, self).__init__(**kwargs)
        self._value = value
        self._pre_epochs = pre_epochs
        self._optim = optim

    def before_train(self, observations, actions, rewards):
        for _ in range(self._pre_epochs):
            value = self._value(observations)  # shape (time, batch_size)
            loss = torch.nn.functional.mse_loss(value[1:], rewards)
            loss.backward()
            self._optim.step()
            self._optim.zero_grad()

    def weights(self, observations, actions, rewards):
        value = self._value(observations)  # shape (time, batch_size)
        loss = torch.nn.functional.mse_loss(value[1:], rewards)
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        # value = value.detach()
        return rewards + value[1:] - value[:-1]


class GeneralisedAdvantageEstimation(VanillaPolicyGradient):
    def __init__(self, gamma, lambda_, value, **kwargs):
        super(GeneralisedAdvantageEstimation, self).__init__(**kwargs)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.value = value

    def weights(self, observations, actions, rewards):
        pass
