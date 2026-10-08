# MO-ADRL — Multi-Objective Adaptive Deep Reinforcement Learning for Intelligent Grids

Reference implementation accompanying the manuscript
**"Multi-Objective Energy Optimization in Intelligent Grids with High Renewable Penetration."**

MO-ADRL schedules grid import, battery charging/discharging and flexible demand every 15 min in a
renewable-rich intelligent grid (500 kW PV, 400 kW wind, 800 kWh / ±200 kW battery, SOC 20–90 %,
850 kW peak and 520 kW average demand). It combines:

| Component | File | Manuscript |
|---|---|---|
| Intelligent-grid simulator (state, battery, power balance, objectives) | `grid_env.py` | Eq. (1), (7)–(13), (19)–(21) |
| Adaptive multi-objective reward weighting | `grid_env.py` (`AdaptiveWeights`) | Eq. (14)–(15) |
| Temporal Transformer forecaster | `forecaster.py` | Eq. (2)–(4) |
| Actor–critic agent and uncertainty-aware exploration | `agent.py` | Eq. (5), (6), (18) |
| Prioritized experience replay (sum-tree) | `per_buffer.py` | Eq. (16)–(17) |
| Episodic score, moving average, convergence episode | `train.py` | Eq. (22)–(24) |
| Rule-based EMS, PSO and GA baselines | `baselines.py` | Table 4 |
| 365-day, 15-min operating profiles (40–80 % renewable penetration) | `profiles.py` | Table 2, Table 11 |
| System parameters and hyperparameters | `config.py` | Table 2, Table 3 |

## Installation
```bash
pip install -r requirements.txt
```

## Reproducing the experiments
```bash
# 1. train MO-ADRL (Algorithm 1); repeat with --seed 1..4 for the five-seed statistics
python train.py --episodes 5000 --seed 0 --penetration 0.8

# 2. evaluate the trained policy and the rule-based / PSO / GA baselines on held-out days
python evaluate.py --model results/mo_adrl_seed0.pt --days 20 --penetration 0.8

# 3. renewable-penetration study (Table 11)
for p in 0.4 0.5 0.6 0.7 0.8; do python train.py --penetration $p --seed 0 --out results_p$p; done

# 4. convergence and adaptive-weight figures
python plot_figures.py --logs results --out figures
```
A quick functional check (about two minutes on a laptop CPU):
```bash
python train.py --episodes 30 --forecast-epochs 1 --warmup 500 --out smoke
```

## Notes
* The operating profiles are generated synthetically by `profiles.py` with fixed seeds; absolute
  cost and emission values therefore depend on the seed and on the tariff/emission-factor settings
  in `config.py`.
* The episodic score of Eq. (22) adds a constant offset `reward_offset` to every step reward. A
  constant offset over a fixed 96-step horizon does not change the ranking of policies and only
  keeps the reported score positive.
* Convergence is declared at the first episode after which the 100-episode moving average remains
  within ±2 % of its final value (Eq. 24).

## Citation
See `CITATION.cff`. The archived release is available on Zenodo (DOI listed in the manuscript's
Code Availability section).

## License
MIT — see `LICENSE`.
