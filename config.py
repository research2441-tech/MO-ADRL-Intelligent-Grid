"""
Configuration of the intelligent-grid test system and of the MO-ADRL learner.

All values mirror the manuscript:
  Table 2  - intelligent grid configuration
  Table 3  - DRL hyperparameters
Values that the manuscript does not fix explicitly (efficiencies, emission
factor, price levels, exploration bounds, PER exponents) are given here with
their default setting so that every experiment is fully specified.
"""
from dataclasses import dataclass, field


@dataclass
class GridConfig:
    pv_capacity_kw: float = 500.0          # Table 2
    wind_capacity_kw: float = 400.0        # Table 2
    battery_capacity_kwh: float = 800.0    # Table 2
    p_charge_max_kw: float = 200.0         # Table 2
    p_discharge_max_kw: float = 200.0      # Table 2
    soc_min: float = 0.20                  # Table 2
    soc_max: float = 0.90                  # Table 2
    soc_init: float = 0.50
    peak_demand_kw: float = 850.0          # Table 2
    avg_demand_kw: float = 520.0           # Table 2
    dt_hours: float = 0.25                 # 15-min scheduling interval
    steps_per_day: int = 96
    days: int = 365                        # simulation duration
    eta_charge: float = 0.95               # eta_C in Eq. (19)
    eta_discharge: float = 0.95            # eta_D in Eq. (19)
    grid_import_max_kw: float = 900.0
    flexible_share: float = 0.15           # share of load that may be shifted (P_DR)
    loss_fraction: float = 0.02            # P_Loss in Eq. (21) as a fraction of delivered power
    emission_factor: float = 0.82          # EF in Eq. (9), kg CO2 per kWh imported
    price_offpeak: float = 0.08            # USD/kWh
    price_mid: float = 0.12
    price_peak: float = 0.20
    renewable_penetration: float = 0.80    # 0.4 ... 0.8 in Table 11


@dataclass
class LearnerConfig:
    actor_lr: float = 1e-4                 # Table 3
    critic_lr: float = 1e-3                # Table 3
    gamma: float = 0.99                    # Table 3
    buffer_size: int = 100_000             # Table 3
    batch_size: int = 256                  # Table 3
    tau: float = 0.005                     # Table 3 (target update factor)
    episodes: int = 5000                   # Table 3
    hidden_units: int = 256                # Table 3
    actor_layers: int = 3                  # Table 3
    critic_layers: int = 3                 # Table 3
    per_alpha: float = 0.6                 # alpha in Eq. (17)
    per_beta0: float = 0.4                 # importance-sampling correction
    per_eps: float = 1e-3                  # epsilon_p in Eq. (16)
    eps_min: float = 0.05                  # Eq. (18)
    eps_max: float = 0.40                  # Eq. (18)
    weight_rate: float = 0.10              # kappa in Eq. (15)
    reward_offset: float = 10.0            # r_0 in Eq. (22)
    ma_window: int = 100                   # M in Eq. (23)
    conv_tolerance: float = 0.02           # xi in Eq. (24)
    lookback: int = 96                     # n in Eq. (2)  (24 h of 15-min samples)
    seed: int = 0


OBJECTIVES = ["cost", "emission", "curtailment", "peak", "battery", "reliability"]
