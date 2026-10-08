"""
Training script for MO-ADRL (Algorithm 1).

Usage
    python train.py --episodes 5000 --seed 0 --penetration 0.8
    python train.py --episodes 50 --forecast-epochs 1      # quick smoke test

Outputs (written to ./results)
    train_log_seed{S}.csv  : episode, episodic score G_e (Eq. 22), moving average (Eq. 23),
                             critic loss, actor loss, mean objective weights
    summary_seed{S}.json   : convergence episode e_c (Eq. 24) and final moving-average score
    mo_adrl_seed{S}.pt     : trained actor/critic weights
"""
import argparse
import csv
import json
import os
import numpy as np
import torch

from config import GridConfig, LearnerConfig, OBJECTIVES
from profiles import build_profiles
from forecaster import Forecaster
from grid_env import IntelligentGridEnv
from per_buffer import PrioritizedReplay
from agent import MOADRLAgent


def moving_average(x, m):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    for e in range(len(x)):
        lo = max(0, e - m + 1)
        out[e] = (c[e + 1] - c[lo]) / (e + 1 - lo)                     # Eq. (23)
    return out


def convergence_episode(ma, tol):
    """First episode after which the moving average stays within tol of its final value (Eq. 24)."""
    final = ma[-1]
    inside = np.abs(ma - final) <= tol * abs(final)
    for e in range(len(ma)):
        if inside[e:].all():
            return e + 1
    return len(ma)


def renewable_uncertainty(prof, k, window=8):
    """U_t in Eq. (18): normalised short-term variability of renewable output."""
    lo = max(0, k - window)
    seg = prof["pv"][lo:k + 1] + prof["wind"][lo:k + 1]
    if seg.size < 2:
        return 0.0
    return float(np.clip(np.std(np.diff(seg)) / 60.0, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--penetration", type=float, default=0.8)
    ap.add_argument("--forecast-epochs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    g = GridConfig(renewable_penetration=args.penetration)
    l = LearnerConfig(seed=args.seed)
    if args.episodes:
        l.episodes = args.episodes
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    prof = build_profiles(g, seed=args.seed)
    fc = Forecaster(lookback=l.lookback)
    fc.fit(prof, epochs=args.forecast_epochs, seed=args.seed)              # Algorithm 1, step 10
    env = IntelligentGridEnv(prof, g, l, forecaster=fc, rng=np.random.default_rng(args.seed))
    agent = MOADRLAgent(env.state_dim, env.action_dim, l)
    buf = PrioritizedReplay(l.buffer_size, env.state_dim, env.action_dim,
                            l.per_alpha, l.per_beta0, l.per_eps, seed=args.seed)

    scores, rows, steps = [], [], 0
    for ep in range(l.episodes):
        s = env.reset()
        G, lc, la, nupd = 0.0, 0.0, 0.0, 0
        wsum = np.zeros(len(OBJECTIVES))
        done = False
        while not done:
            k = env._idx()
            a = agent.act(s, renewable_uncertainty(prof, k), explore=True)
            s2, r, done, info = env.step(a)
            buf.add(s, a, r, s2, float(done))
            G += r + l.reward_offset                                           # Eq. (22)
            wsum += info["w"]
            s = s2
            steps += 1
            if len(buf) >= max(l.batch_size, min(args.warmup, l.buffer_size)):
                c, act_l = agent.update(buf, progress=ep / l.episodes)
                lc += c; la += act_l; nupd += 1
        scores.append(G)
        ma = moving_average(scores, l.ma_window)
        rows.append([ep + 1, G, ma[-1], lc / max(nupd, 1), la / max(nupd, 1)] +
                    list(wsum / g.steps_per_day))
        if (ep + 1) % 50 == 0:
            print(f"episode {ep + 1:5d}  G={G:8.2f}  MA={ma[-1]:8.2f}")

    ma = moving_average(scores, l.ma_window)
    e_c = convergence_episode(ma, l.conv_tolerance)
    with open(os.path.join(args.out, f"train_log_seed{args.seed}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["episode", "score", "moving_avg", "critic_loss", "actor_loss"] +
                    [f"w_{o}" for o in OBJECTIVES])
        wr.writerows(rows)
    with open(os.path.join(args.out, f"summary_seed{args.seed}.json"), "w") as f:
        json.dump({"convergence_episode": int(e_c), "final_moving_avg": float(ma[-1]),
                   "episodes": l.episodes, "penetration": args.penetration}, f, indent=2)
    agent.save(os.path.join(args.out, f"mo_adrl_seed{args.seed}.pt"))
    print(f"convergence episode e_c = {e_c},  final moving-average score = {ma[-1]:.2f}")


if __name__ == "__main__":
    main()
