"""
Generation of 365-day, 15-min resolution operating profiles for the test grid:
load, PV output, wind output, electricity price, carbon factor and weather
features (irradiance, wind speed, ambient temperature).

The profiles are synthetic but physically structured:
  * load      - daily double-peak shape + weekly and seasonal modulation + noise,
                scaled to the 520 kW average and 850 kW peak of Table 2;
  * PV        - clear-sky bell curve x seasonal day length x stochastic cloud index;
  * wind      - Weibull-distributed, AR(1)-correlated wind speed mapped through a
                cut-in / rated / cut-out power curve;
  * price     - three-level time-of-use tariff.
Renewable penetration (40-80 %) is imposed by scaling the PV and wind outputs so
that annual renewable energy equals the requested share of annual demand.
"""
import numpy as np
from config import GridConfig


def _load_profile(cfg: GridConfig, rng):
    n = cfg.days * cfg.steps_per_day
    t = np.arange(n)
    hour = (t % cfg.steps_per_day) * cfg.dt_hours
    day = t // cfg.steps_per_day
    morning = np.exp(-0.5 * ((hour - 9.0) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((hour - 19.5) / 2.2) ** 2)
    base = 0.55 + 0.25 * morning + 0.42 * evening
    weekly = np.where((day % 7) >= 5, 0.92, 1.0)
    seasonal = 1.0 + 0.08 * np.cos(2 * np.pi * (day - 200) / 365.0)
    noise = 1.0 + 0.04 * rng.standard_normal(n)
    raw = base * weekly * seasonal * noise
    raw = raw / raw.mean() * cfg.avg_demand_kw
    # cap the extreme values at the configured peak
    return np.minimum(raw, cfg.peak_demand_kw)


def _pv_profile(cfg: GridConfig, rng):
    n = cfg.days * cfg.steps_per_day
    t = np.arange(n)
    hour = (t % cfg.steps_per_day) * cfg.dt_hours
    day = t // cfg.steps_per_day
    daylen = 12.0 + 2.0 * np.sin(2 * np.pi * (day - 80) / 365.0)
    sunrise = 12.0 - daylen / 2.0
    x = np.clip((hour - sunrise) / daylen, 0.0, 1.0)
    clear = np.sin(np.pi * x) ** 1.5
    # day-level cloudiness and intra-day fluctuation
    cloud_day = np.repeat(rng.beta(5, 2, cfg.days), cfg.steps_per_day)
    cloud_fast = np.clip(1.0 - 0.25 * np.abs(rng.standard_normal(n)) * (1 - cloud_day), 0.2, 1.0)
    irr = 1000.0 * clear * cloud_day * cloud_fast            # W/m^2
    pv = cfg.pv_capacity_kw * irr / 1000.0
    return pv, irr


def _wind_profile(cfg: GridConfig, rng):
    n = cfg.days * cfg.steps_per_day
    k, c = 2.0, 7.5                                         # Weibull shape/scale (m/s)
    z = np.zeros(n)
    phi = 0.97
    e = rng.standard_normal(n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * e[i]
    # Gaussian copula: correlated normal -> uniform -> Weibull wind speed
    u = np.clip(0.5 * (1 + _erf(z / np.sqrt(2))), 1e-6, 1 - 1e-6)
    speed = c * (-np.log(1 - u)) ** (1.0 / k)
    v_in, v_r, v_out = 3.0, 12.0, 25.0
    p = np.where(speed < v_in, 0.0,
                 np.where(speed < v_r, ((speed - v_in) / (v_r - v_in)) ** 3,
                          np.where(speed < v_out, 1.0, 0.0)))
    return cfg.wind_capacity_kw * p, speed


def _erf(x):
    # Abramowitz-Stegun approximation (vectorised, no SciPy dependency)
    s = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return s * y


def _price_profile(cfg: GridConfig):
    n = cfg.days * cfg.steps_per_day
    hour = (np.arange(n) % cfg.steps_per_day) * cfg.dt_hours
    price = np.full(n, cfg.price_mid)
    price[(hour < 7) | (hour >= 23)] = cfg.price_offpeak
    price[(hour >= 17) & (hour < 22)] = cfg.price_peak
    return price


def build_profiles(cfg: GridConfig, seed: int = 0):
    """Return a dict of aligned 15-min arrays for one simulated year."""
    rng = np.random.default_rng(seed)
    load = _load_profile(cfg, rng)
    pv, irr = _pv_profile(cfg, rng)
    wind, speed = _wind_profile(cfg, rng)
    # impose the requested renewable penetration on annual energy
    share = cfg.renewable_penetration
    ren = pv + wind
    scale = share * load.sum() / max(ren.sum(), 1e-9)
    pv, wind = pv * scale, wind * scale
    n = load.size
    hour = (np.arange(n) % cfg.steps_per_day) * cfg.dt_hours
    temp = 22 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.0, n)
    return {
        "load": load, "pv": pv, "wind": wind,
        "price": _price_profile(cfg),
        "irradiance": irr, "wind_speed": speed, "temperature": temp,
        "hour": hour,
    }
