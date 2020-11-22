import torch


def mlp(in_size, out_size, mlp_size, num_layers):
    model = [torch.nn.Linear(in_size, mlp_size),
             torch.nn.ReLU()]
    for _ in range(num_layers - 1):
        model.append(torch.nn.Linear(mlp_size, mlp_size))
        model.append(torch.nn.ReLU())
    model.append(torch.nn.Linear(mlp_size, out_size))
    return torch.nn.Sequential(*model)


def train(env, policy, optim, algorithm, epochs, batch_size):
    observations, actions, rewards = env(policy, batch_dims=(batch_size,))
    # observations has shape (time=300, batch_size, obs_size=4)
    # actions has shape (time=299, batch_size, num-actions=2)
    # rewards has shape (time=299, batch_size)
    # observations has one more time because it includes the final observation:
    #   observation_0
    #   action_0
    #   reward_0
    #   observation_1
    #   ...
    #   observation_n-1
    #   action_n-1
    #   reward_n-1
    #   observation_n

    algorithm.before_train(observations, actions, rewards)

    for epoch in range(epochs):
        observations, actions, rewards = env(policy, batch_dims=(batch_size,))
        rewards = rewards.to(observations.dtype)
        performance = algorithm(observations, actions, rewards)
        performance.backward()
        optim.step()
        optim.zero_grad()

        print(f"Epoch: {epoch} Reward: {rewards.sum(dim=0).mean(dim=0)}")

    return observations  # return last observation batch
