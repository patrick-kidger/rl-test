import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torchdiffeq


class Env:
    def __call__(self, *args, **kwargs):
        with torch.no_grad():  # no cheating!
            return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ODEEnv(Env):
    def x0(self, batch_dims):
        raise NotImplementedError

    def ts(self):
        raise NotImplementedError

    def reward(self, obs):
        raise NotImplementedError

    def d_obs(self, obs, action):
        raise NotImplementedError

    def forward(self, policy, batch_dims):
        # TODO: think about how to properly make everything work in continuous time / other integrators
        # TODO: incorporate log-probs

        actions = []
        #log_probs = []

        def f(t, obs):
            dist = policy(obs)
            action = dist.sample()

            actions.append(action)
            #log_probs.append(dist.log_probs(action))

            return self.d_obs(obs, action)

        # TODO: tidy up the ts
        observations = torchdiffeq.odeint(f, self.x0(batch_dims), self.ts(), method='euler')
        rewards = self.reward(observations[1:])  # [1:] because you don't get a reward for where you start!

        return observations, torch.stack(actions), rewards#, torch.stack(log_probs)


class CartPole(ODEEnv):
    obs_size = 4
    num_actions = 2

    def __init__(self,
                 device='cpu',
                 # Reward parameters
                 theta_threshold=math.pi / 4,
                 pos_threshold=2.4,
                 # Simulation parameters
                 tau=0.01,
                 tspan=300,
                 # Problem parameters
                 force_mag=10.,
                 gravity=9.8,
                 mass_cart=1.0,
                 mass_pole=0.1,
                 length_pole=1.,
                 init_range=0.05,
                 ):

        self._device = device
        self._tspan = tspan
        self._theta_threshold = theta_threshold
        self._pos_threshold = pos_threshold

        self._left = torch.full((), -force_mag, device=self._device)
        self._right = torch.full((), force_mag, device=self._device)
        self._ts = torch.linspace(0, tau * (tspan - 1), tspan, device=self._device)

        self._gravity = gravity
        self._mass_pole = mass_pole
        self._inv_mass_total = 1 / (mass_cart + mass_pole)
        self._half_length_pole = 0.5 * length_pole
        self._half_length_pole__mass_pole = self._half_length_pole * mass_pole
        self._init_range = init_range

    def x0(self, batch_dims):
        return torch.rand(*batch_dims, self.obs_size, device=self._device) * (2 * self._init_range) - self._init_range

    def ts(self):
        return self._ts

    def reward(self, obs):
        pos = obs[..., 0]
        theta = obs[..., 2]
        return ((-self._pos_threshold < pos) &
                (pos < self._pos_threshold) &
                (-self._theta_threshold < theta) &
                (theta < self._theta_threshold))

    def d_obs(self, obs, action):
        d_pos = obs[..., 1]
        theta = obs[..., 2]
        d_theta = obs[..., 3]
        force = torch.where(action.to(bool), self._left, self._right)

        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        sin_theta = theta.sin()
        cos_theta = theta.cos()
        _tmp1 = (force + self._half_length_pole__mass_pole * sin_theta * d_theta ** 2) * self._inv_mass_total
        _tmp2 = self._gravity * sin_theta - cos_theta * _tmp1
        _tmp3 = self._half_length_pole * (4.0 / 3.0 - self._mass_pole * cos_theta ** 2 * self._inv_mass_total)
        dd_theta = _tmp2 / _tmp3
        dd_pos = _tmp1 - self._half_length_pole__mass_pole * dd_theta * cos_theta * self._inv_mass_total

        return torch.stack([d_pos, dd_pos, d_theta, dd_theta], dim=-1)

    def render(self, observations):
        # https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html

        assert observations.shape == (self._tspan, self.obs_size), f"Expected shape {(self._tspan, self.obs_size)}, " \
                                                                   f"got shape {observations.shape}"
        pos = observations[:, 0]
        theta = observations[:, 1]

        pole_length = 2 * self._half_length_pole

        bottom_x_pos = pos
        bottom_y_pos = torch.full_like(pos, 0)
        top_x_pos = bottom_x_pos + theta.sin() * pole_length
        top_y_pos = bottom_y_pos + theta.cos() * pole_length

        fig = plt.figure()
        fig.set_dpi(100)
        fig.set_size_inches(6, 6)
        xlim = self._pos_threshold + pole_length + 0.5
        ylim = pole_length + 0.5
        ax = plt.axes(xlim=(-xlim, xlim), ylim=(-ylim, ylim))

        cart = plt.Circle((-500, -500), 0.1, fc='y')
        pole = plt.Line2D((-500, -500), (-500, -500), lw=2)

        def init():
            cart.center = (bottom_x_pos[0].item(), bottom_y_pos[0].item())
            pole.set_xdata((bottom_x_pos[0].item(), top_x_pos[0].item()))
            pole.set_ydata((bottom_y_pos[0].item(), top_y_pos[0].item()))
            ax.add_patch(cart)
            ax.add_line(pole)
            return cart, pole

        def animate(i):
            cart.center = (bottom_x_pos[i].item(), bottom_y_pos[i].item())
            pole.set_xdata((bottom_x_pos[i].item(), top_x_pos[i].item()))
            pole.set_ydata((bottom_y_pos[i].item(), top_y_pos[i].item()))
            return cart, pole

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=self._tspan, interval=40, blit=True)
        plt.show()
