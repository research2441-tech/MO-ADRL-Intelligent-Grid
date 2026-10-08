"""
Evaluation of a trained MO-ADRL policy and of the rule-based, PSO and GA baselines
on held-out days (days 300-364 of the simulated year).

Usage
    python evaluate.py --model results/mo_adrl_seed0.pt --days 10 --penetration 0.8
Writes results/eval_metrics.csv (daily cost, CO2, renewable utilisation, curtailment,
peak demand and reliability; the quantities of Tables 5-9 and 11) and
results/baseline_convergence.csv (best-so-far objective per iteration for PSO and GA).
"""
import argparse
import csv
import os
import numpy as np
import torch

from config import GridConfig, LearnerConfig
from profiles import build_profiles
from forecaster import Forecaster
from grid_env import IntelligentGridEnv
from agent import MOADRLAgent
from baselines import rollout, rule_based_schedule, pso, ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/mo_adrl_seed0.pt")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--penetration", type=float, default=0.8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--forecast-epochs", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    g = GridConfig(renewable_penetration=args.penetration)
    l = LearnerConfig()
    prof = build_profiles(g, seed=0)
    fc = Forecaster(lookback=l.lookback)
    fc.fit(prof, epochs=args.forecast_epochs)
    env = IntelligentGridEnv(prof, g, l, forecaster=fc)
    agent = MOADRLAgent(env.state_dim, env.action_dim, l)
    if os.path.exists(args.model):
        agent.actor.load_state_dict(torch.load(args.model)["actor"])

    days = np.linspace(300, 364, args.days).astype(int)
    rows, curves = [], {"PSO": [], "GA": []}
    for d in days:
        res = {"Rule-Based EMS": rollout(env, d, rule_based_schedule(env, d))[0]}
        sched, c = pso(env, d, iters=args.iters); curves["PSO"].append(c)
        res["PSO"] = rollout(env, d, sched)[0]
        sched, c = ga(env, d, iters=args.iters); curves["GA"].append(c)
        res["GA"] = rollout(env, d, sched)[0]
        s = env.reset(day=d)
        done = False
        while not done:
            s, _, done, _ = env.step(agent.act(s, explore=False))
        res["MO-ADRL"] = env.episode_metrics()
        for name, m in res.items():
            rows.append({"day": int(d), "method": name, **m})

    with open(os.path.join(args.out, "eval_metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.out, "baseline_convergence.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iteration", "PSO_mean", "PSO_std", "GA_mean", "GA_std"])
        P, G = np.array(curves["PSO"]), np.array(curves["GA"])
        for i in range(P.shape[1]):
            wr.writerow([i, P[:, i].mean(), P[:, i].std(), G[:, i].mean(), G[:, i].std()])
    for name in ["Rule-Based EMS", "PSO", "GA", "MO-ADRL"]:
        sub = [r for r in rows if r["method"] == name]
        print(f"{name:15s} cost={np.mean([r['daily_cost_usd'] for r in sub]):8.1f} USD  "
              f"CO2={np.mean([r['co2_kg'] for r in sub]):8.1f} kg  "
              f"RU={np.mean([r['renewable_utilization_pct'] for r in sub]):5.1f}%  "
              f"peak={np.mean([r['peak_kw'] for r in sub]):6.1f} kW  "
              f"rel={np.mean([r['reliability_pct'] for r in sub]):5.2f}%")


if __name__ == "__main__":
    main()
