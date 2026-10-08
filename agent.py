"""
MO-ADRL actor-critic agent, Eq. (5), (6) and (18).

Actor   A_t = pi_theta(S_t*)            deterministic continuous policy (Eq. 5)
Critic  Q_phi(S_t*, A_t)                state-action value (Eq. 6)
Target networks with soft update factor tau (Table 3).
Exploration noise scale eps_t = eps_min + (eps_max - eps_min) * U_t   (Eq. 18),
where U_t is the normalised short-term variability of renewable output.
"""
import copy
import numpy as np
import torch
import torch.nn as nn


def mlp(inp, out, hidden, layers, out_act=None):
    mods, d = [], inp
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    if out_act is not None:
        mods.append(out_act)
    return nn.Sequential(*mods)


class MOADRLAgent:
    def __init__(self, state_dim, action_dim, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.actor = mlp(state_dim, action_dim, cfg.hidden_units, cfg.actor_layers, nn.Tanh()).to(device)
        self.critic = mlp(state_dim + action_dim, 1, cfg.hidden_units, cfg.critic_layers).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.action_dim = action_dim
        self.rng = np.random.default_rng(cfg.seed)

    def exploration_scale(self, uncertainty):
        u = float(np.clip(uncertainty, 0.0, 1.0))
        return self.cfg.eps_min + (self.cfg.eps_max - self.cfg.eps_min) * u      # Eq. (18)

    @torch.no_grad()
    def act(self, s, uncertainty=0.0, explore=True):
        a = self.actor(torch.as_tensor(s, device=self.device)[None]).cpu().numpy()[0]
        if explore:
            a = a + self.rng.normal(0, self.exploration_scale(uncertainty), self.action_dim)
        return np.clip(a, -1, 1)

    def update(self, buffer, progress):
        cfg = self.cfg
        idx, (s, a, r, s2, d), w = buffer.sample(cfg.batch_size, progress)
        T = lambda x: torch.as_tensor(x, device=self.device)
        s, a, r, s2, d, w = map(T, (s, a, r, s2, d, w))
        with torch.no_grad():
            q_t = self.critic_t(torch.cat([s2, self.actor_t(s2)], 1))
            y = r + cfg.gamma * (1 - d) * q_t
        q = self.critic(torch.cat([s, a], 1))
        td = y - q
        loss_c = (w * td.pow(2)).mean()
        self.opt_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()
        loss_a = -self.critic(torch.cat([s, self.actor(s)], 1)).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
        buffer.update(idx, td.detach().cpu().numpy().ravel())
        return float(loss_c.item()), float(loss_a.item())

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
