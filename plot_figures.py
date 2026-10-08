"""
Plotting of convergence diagnostics from the logs written by train.py and evaluate.py.

    python plot_figures.py --logs results --out figures

* reward_convergence.png  : raw episodic score, moving average (Eq. 23), +/-1 std across
                            seeds and the convergence episode e_c (Eq. 24)
* baseline_convergence.png: best-so-far objective of PSO and GA versus iteration
                            (non-increasing by construction for a minimisation problem)
* adaptive_weights.png    : objective weights w_i(t) of Eq. (15) over one representative day
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import GridConfig, LearnerConfig, OBJECTIVES
from profiles import build_profiles
from grid_env import IntelligentGridEnv
from baselines import rule_based_schedule
from train import moving_average, convergence_episode

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def reward_convergence(logdir, out):
    files = sorted(glob.glob(os.path.join(logdir, "train_log_seed*.csv")))
    if not files:
        print("no training logs found")
        return
    S = np.array([np.genfromtxt(f, delimiter=",", names=True)["score"] for f in files])
    l = LearnerConfig()
    ma = np.array([moving_average(s, l.ma_window) for s in S])
    m, sd = ma.mean(0), ma.std(0)
    e_c = convergence_episode(m, l.conv_tolerance)
    ep = np.arange(1, S.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(ep, S[0], color="0.75", lw=0.5, label="Episodic score (seed 0)")
    ax.plot(ep, m, color="C3", lw=1.6, label=f"{l.ma_window}-episode moving average")
    ax.fill_between(ep, m - sd, m + sd, color="C3", alpha=0.2, label="±1 std over seeds")
    ax.axvline(e_c, ls="--", color="k", lw=0.9)
    ax.text(e_c, ax.get_ylim()[0], f"  e_c = {e_c}", va="bottom")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Episodic score G_e")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "reward_convergence.png"), dpi=300)


def baseline_convergence(logdir, out):
    f = os.path.join(logdir, "baseline_convergence.csv")
    if not os.path.exists(f):
        print("no baseline convergence file found")
        return
    d = np.genfromtxt(f, delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for name, c in (("PSO", "C0"), ("GA", "C2")):
        ax.plot(d["iteration"], d[f"{name}_mean"], color=c, label=name)
        ax.fill_between(d["iteration"], d[f"{name}_mean"] - d[f"{name}_std"],
                        d[f"{name}_mean"] + d[f"{name}_std"], color=c, alpha=0.2)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best-so-far objective")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out, "baseline_convergence.png"), dpi=300)


def adaptive_weights(out, day=172):
    g, l = GridConfig(), LearnerConfig()
    env = IntelligentGridEnv(build_profiles(g), g, l)
    sched = rule_based_schedule(env, day)
    env.reset(day=day)
    W = []
    for t in range(g.steps_per_day):
        _, _, _, info = env.step(sched[t])
        W.append(info["w"])
    W = np.array(W)
    h = np.arange(g.steps_per_day) * g.dt_hours
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.stackplot(h, W.T, labels=OBJECTIVES, alpha=0.85)
    ax.set_xlim(0, 24); ax.set_ylim(0, 1)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Objective weight w_i(t)")
    ax.legend(ncol=3, fontsize=7, loc="upper center")
    fig.tight_layout(); fig.savefig(os.path.join(out, "adaptive_weights.png"), dpi=300)
    return W


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reward_convergence(a.logs, a.out)
    baseline_convergence(a.logs, a.out)
    adaptive_weights(a.out)
