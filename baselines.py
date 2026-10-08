"""
Baseline energy-management methods: rule-based EMS, PSO and GA.

PSO and GA solve a day-ahead schedule (96 battery set-points) for each day by
population-based search on the same normalised objective used by MO-ADRL with
fixed equal weights.  Their best-so-far objective is recorded at every
iteration, which gives a monotonically non-increasing convergence curve.
"""
import numpy as np
from grid_env import IntelligentGridEnv


def rollout(env, day, schedule):
    """Simulate one day with a fixed schedule of actions (96 x 4)."""
    env.reset(day=day)
    for t in range(env.g.steps_per_day):
        env.step(schedule[t])
    return env.episode_metrics(), env.log


def rule_based_schedule(env, day):
    """Charge on renewable surplus, discharge on deficit, no demand response."""
    g, p = env.g, env.p
    k0 = day * g.steps_per_day
    sched = np.zeros((g.steps_per_day, 4))
    for t in range(g.steps_per_day):
        k = k0 + t
        surplus = p["pv"][k] + p["wind"][k] - p["load"][k]
        sched[t, 0] = 1.0
        if surplus > 0:
            sched[t, 1] = 2 * min(surplus / g.p_charge_max_kw, 1) - 1
            sched[t, 2] = -1
        else:
            sched[t, 1] = -1
            sched[t, 2] = 2 * min(-surplus / g.p_discharge_max_kw, 1) - 1
        sched[t, 3] = 0.0
    return sched


def _objective(env, day, x):
    """x: (96, 2) -> battery net set-point and DR set-point in [-1, 1]."""
    sched = np.zeros((env.g.steps_per_day, 4))
    sched[:, 0] = 1.0
    sched[:, 1] = np.where(x[:, 0] > 0, 2 * x[:, 0] - 1, -1)
    sched[:, 2] = np.where(x[:, 0] < 0, -2 * x[:, 0] - 1, -1)
    sched[:, 3] = x[:, 1]
    env.reset(day=day)
    total = 0.0
    for t in range(env.g.steps_per_day):
        _, _, _, info = env.step(sched[t])
        total += float(np.mean(info["j"] / env.scale))
    return total, sched


def pso(env, day, n_particles=30, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    dim = (env.g.steps_per_day, 2)
    x = rng.uniform(-1, 1, (n_particles,) + dim)
    v = np.zeros_like(x)
    pbest = x.copy()
    pval = np.array([_objective(env, day, xi)[0] for xi in x])
    g = pbest[pval.argmin()].copy()
    gval = pval.min()
    curve = [gval]
    for it in range(iters):
        w = 0.9 - 0.5 * it / iters
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + 1.5 * r1 * (pbest - x) + 1.5 * r2 * (g - x)
        x = np.clip(x + v, -1, 1)
        val = np.array([_objective(env, day, xi)[0] for xi in x])
        better = val < pval
        pbest[better], pval[better] = x[better], val[better]
        if pval.min() < gval:
            gval, g = pval.min(), pbest[pval.argmin()].copy()
        curve.append(gval)                      # best-so-far -> non-increasing
    return _objective(env, day, g)[1], np.array(curve)


def ga(env, day, pop=30, iters=200, pc=0.9, pm=0.02, seed=0):
    rng = np.random.default_rng(seed)
    dim = (env.g.steps_per_day, 2)
    x = rng.uniform(-1, 1, (pop,) + dim)
    val = np.array([_objective(env, day, xi)[0] for xi in x])
    best, bval = x[val.argmin()].copy(), val.min()
    curve = [bval]
    for _ in range(iters):
        # tournament selection
        a, b = rng.integers(0, pop, (2, pop))
        parents = np.where((val[a] < val[b])[:, None, None], x[a], x[b])
        children = parents.copy()
        for i in range(0, pop - 1, 2):
            if rng.random() < pc:
                alpha = rng.random()
                children[i] = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                children[i + 1] = alpha * parents[i + 1] + (1 - alpha) * parents[i]
        mask = rng.random(children.shape) < pm
        children[mask] = rng.uniform(-1, 1, mask.sum())
        children[0] = best                       # elitism
        x = children
        val = np.array([_objective(env, day, xi)[0] for xi in x])
        if val.min() < bval:
            bval, best = val.min(), x[val.argmin()].copy()
        curve.append(bval)
    return _objective(env, day, best)[1], np.array(curve)
