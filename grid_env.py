"""
Intelligent-grid environment used for training and evaluating MO-ADRL.

One episode = one day = 96 scheduling intervals of 15 min.
State  S_t   : Eq. (1)  -> [L_t, P_PV,t, P_W,t, SOC_t, C_t, E_t]
Action A_t   : Eq. (7)  -> [P_Grid, P_Charge, P_Discharge, P_DR] (continuous, in [-1, 1])
Battery      : Eq. (19)-(20)
Balance      : Eq. (21)
Objectives   : Eq. (8)-(13)
Reward       : Eq. (14) with adaptive weights of Eq. (15)
"""
import numpy as np
from config import GridConfig, LearnerConfig, OBJECTIVES


class AdaptiveWeights:
    """Adaptive objective weighting, Eq. (15)."""

    def __init__(self, n_obj, rate=0.1, w_min=0.03, momentum=0.98):
        self.w = np.full(n_obj, 1.0 / n_obj)
        self.rate = rate
        self.w_min = w_min
        self.mom = momentum
        self.mean = None

    def reset(self):
        self.w[:] = 1.0 / self.w.size
        self.mean = None

    def update(self, j_norm):
        """j_norm: normalised objective values J_i(t) in [0, ~1]."""
        j_norm = np.asarray(j_norm, dtype=float)
        if self.mean is None:
            self.mean = j_norm.copy() + 1e-6
        # Delta_i: relative deterioration of objective i w.r.t. its running mean
        delta = np.maximum(0.0, (j_norm - self.mean) / (self.mean + 1e-6))
        delta = np.minimum(delta, 5.0)
        w = self.w + self.rate * delta
        w = np.maximum(w / w.sum(), self.w_min)
        self.w = w / w.sum()
        self.mean = self.mom * self.mean + (1 - self.mom) * j_norm
        return self.w.copy(), delta


class IntelligentGridEnv:
    def __init__(self, profiles, gcfg: GridConfig, lcfg: LearnerConfig, forecaster=None, rng=None):
        self.p = profiles
        self.g = gcfg
        self.l = lcfg
        self.forecaster = forecaster
        self.rng = rng or np.random.default_rng(lcfg.seed)
        self.weights = AdaptiveWeights(len(OBJECTIVES), rate=lcfg.weight_rate)
        # scales used to normalise every objective to O(1)
        dt = gcfg.dt_hours
        self.scale = np.array([
            gcfg.grid_import_max_kw * gcfg.price_peak * dt,      # cost
            gcfg.grid_import_max_kw * gcfg.emission_factor * dt, # emission
            (gcfg.pv_capacity_kw + gcfg.wind_capacity_kw) * dt,  # curtailment
            0.10 * gcfg.peak_demand_kw,                          # peak increment
            (gcfg.p_charge_max_kw + gcfg.p_discharge_max_kw),    # battery throughput
            gcfg.peak_demand_kw * dt,                            # unserved energy
        ])
        self.state_dim = 6 + 3 + 2      # Eq.(1) + 3 forecasts (Eq.4) + time encoding
        self.action_dim = 4

    # ------------------------------------------------------------------ #
    def reset(self, day=None):
        g = self.g
        self.day = int(self.rng.integers(1, g.days)) if day is None else int(day)
        self.k0 = self.day * g.steps_per_day
        self.t = 0
        self.soc = g.soc_init
        self.peak_so_far = 0.0
        self.weights.reset()
        self.log = []
        return self._state()

    def _idx(self):
        return self.k0 + self.t

    def _forecast(self, k):
        if self.forecaster is not None:
            return self.forecaster.predict_next(self.p, k)
        # persistence fallback (used before the Transformer is trained)
        k1 = min(k + 1, self.p["load"].size - 1)
        return self.p["load"][k1], self.p["pv"][k1], self.p["wind"][k1]

    def _state(self):
        g, p, k = self.g, self.p, self._idx()
        l_hat, pv_hat, w_hat = self._forecast(k)
        h = p["hour"][k]
        s = np.array([
            p["load"][k] / g.peak_demand_kw,
            p["pv"][k] / g.pv_capacity_kw,
            p["wind"][k] / g.wind_capacity_kw,
            self.soc,
            p["price"][k] / g.price_peak,
            p["temperature"][k] / 40.0,
            l_hat / g.peak_demand_kw,
            pv_hat / g.pv_capacity_kw,
            w_hat / g.wind_capacity_kw,
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        ], dtype=np.float32)
        return s

    # ------------------------------------------------------------------ #
    def step(self, action):
        g, p, k = self.g, self.p, self._idx()
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        dt = g.dt_hours
        load, pv, wind, price = p["load"][k], p["pv"][k], p["wind"][k], p["price"][k]

        # --- decode action, Eq. (7) -----------------------------------
        grid_cap = 0.5 * (a[0] + 1.0) * g.grid_import_max_kw
        p_ch = 0.5 * (a[1] + 1.0) * g.p_charge_max_kw
        p_dis = 0.5 * (a[2] + 1.0) * g.p_discharge_max_kw
        net_b = p_ch - p_dis                      # simultaneous charge/discharge is netted
        p_ch, p_dis = max(net_b, 0.0), max(-net_b, 0.0)
        p_dr = a[3] * g.flexible_share * load     # >0 load reduced (shifted), <0 load added

        # --- battery limits, Eq. (19)-(20) ----------------------------
        e = g.battery_capacity_kwh
        p_ch = min(p_ch, (g.soc_max - self.soc) * e / (g.eta_charge * dt))
        p_dis = min(p_dis, (self.soc - g.soc_min) * e * g.eta_discharge / dt)
        p_ch, p_dis = max(p_ch, 0.0), max(p_dis, 0.0)

        # --- power balance, Eq. (21) ----------------------------------
        demand = load - p_dr
        ren = pv + wind
        need = demand + p_ch
        losses = g.loss_fraction * demand
        residual = need + losses - ren - p_dis
        if residual > 0:                          # deficit: import from utility grid
            p_grid = min(residual, grid_cap, g.grid_import_max_kw)
            unserved = max(residual - p_grid, 0.0)
            curtailed = 0.0
        else:                                     # surplus renewable energy
            p_grid = 0.0
            unserved = 0.0
            curtailed = -residual
        self.soc = self.soc + (g.eta_charge * p_ch * dt) / e - (p_dis * dt) / (g.eta_discharge * e)
        self.soc = float(np.clip(self.soc, g.soc_min, g.soc_max))

        # --- objectives, Eq. (8)-(13) ---------------------------------
        j_c = p_grid * price * dt                                 # Eq. (8)
        j_e = p_grid * g.emission_factor * dt                     # Eq. (9)
        j_r = curtailed * dt                                      # Eq. (10)
        net_load = demand
        j_p = max(0.0, net_load - self.peak_so_far) if self.t > 0 else 0.0   # Eq. (11)
        self.peak_so_far = max(self.peak_so_far, net_load)
        j_b = abs(p_ch) + abs(p_dis)                              # Eq. (12)
        j_u = unserved * dt                                       # Eq. (13)
        j = np.array([j_c, j_e, j_r, j_p, j_b, j_u])
        j_norm = j / self.scale

        # --- adaptive reward, Eq. (14)-(15) ---------------------------
        w, _ = self.weights.update(j_norm)
        reward = -float(np.dot(w, j_norm))

        self.log.append(dict(t=self.t, load=load, pv=pv, wind=wind, price=price, p_grid=p_grid,
                             p_ch=p_ch, p_dis=p_dis, p_dr=p_dr, soc=self.soc, curtailed=curtailed,
                             unserved=unserved, cost=j_c, co2=j_e, weights=w.copy()))
        self.t += 1
        done = self.t >= g.steps_per_day
        nxt = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return nxt, reward, done, {"j": j, "w": w}

    # ------------------------------------------------------------------ #
    def episode_metrics(self):
        """Daily metrics reported in Tables 5-9 and 11."""
        lg = self.log
        g = self.g
        dt = g.dt_hours
        load_e = sum(r["load"] for r in lg) * dt
        ren_e = sum(r["pv"] + r["wind"] for r in lg) * dt
        curt = sum(r["curtailed"] for r in lg) * dt
        unserved = sum(r["unserved"] for r in lg) * dt
        return {
            "daily_cost_usd": sum(r["cost"] for r in lg),
            "co2_kg": sum(r["co2"] for r in lg),
            "renewable_utilization_pct": 100.0 * (ren_e - curt) / max(ren_e, 1e-9),
            "curtailment_pct": 100.0 * curt / max(ren_e, 1e-9),
            "peak_kw": max(r["load"] - r["p_dr"] for r in lg),
            "reliability_pct": 100.0 * (1 - unserved / max(load_e, 1e-9)),
        }
