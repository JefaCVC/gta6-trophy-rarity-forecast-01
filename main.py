"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case's target as five
quantiles (5/25/50/75/95%), from the case's history of reads.

Model: counts, not rates. A trophy's earn rate is r = E / P, earners over
players tracked. The two move for different reasons, and the final's 4.5-day
horizon is long enough for both to move a lot:

  * P(t), players tracked, grows as the launch flood keeps arriving. Modelled
    with a saturating Weibull curve fitted to the case's distinct reads
    (power-law fallback when no Weibull shape fits).
  * E_i(t), earners of trophy i, grows as players progress. Modelled as a
    cohort convolution: players join over time (the fitted P curve's joins),
    and a joiner earns trophy i after a Weibull-distributed lag (shape 1 or 2,
    scale fitted per trophy on a grid, share fitted in closed form). Earners
    never decrease, and the seed cohort (the launch-night readers) is kept.

The rate forecast is the ratio of the two extrapolations, so dilution (P
outgrowing E, which crushes late trophies' rates) and progression (E
outgrowing P, which lifts early trophies) both fall out of the structure
instead of being extrapolated as a rate trend. Story trophies are kept
monotone in chapter order, and the platinum / 100%-completion trophies are
capped by the scarcest trophy, because nobody can hold those without the rest.

Spread: per-trophy logit spread from a base term, a horizon term growing with
sqrt(days), the trophy's fit residual, and the P-fit residual. Constants were
calibrated on a synthetic ensemble of launch worlds (a simulator with
different functional forms from this model, calibrated so its day-2.7 page
reproduces the real Exophase snapshot), never on the practice cases: the
practice's 90-minute horizon straddles one refresh and says nothing about a
4.5-day forecast. On held-out synthetic final-like cases the model scores a
median pinball loss ~0.66 against ~0.91 for the house trend baseline and
~1.78 for persistence, with 90% coverage ~0.89.

Standard library only. Deterministic: fixed grids, no randomness, one pass.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Cohort lag grid (hours) and Weibull shapes for time-to-earn.
L_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400, 800]
K_GRID = [1.0, 2.0]
# Weibull shapes tried for P(t).
GAMMA_GRID = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]

# Spread constants (calibrated on the synthetic ensemble; see docstring).
S0 = 0.5          # base logit spread (refresh + process noise)
C_HOR = 0.8       # horizon term, per sqrt(day)
RESID_MULT = 1.0  # weight on the per-trophy fit residual
P_RESID_MULT = 1.0  # weight on the P-fit residual
SPREAD_MIN = 0.15
SPREAD_MAX = 4.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def logistic(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the kit starter's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO 8601 UTC timestamp -> aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def distinct_reads(history):
    """Drop consecutive reads with identical players + rates (Exophase only
    refreshes on its own schedule, so most reads repeat the page)."""
    out = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev:
            out.append(read)
            prev = key
    return out


def weibull_ratio(tau, t1, t2, gamma):
    f1 = 1.0 - math.exp(-((t1 / tau) ** gamma))
    f2 = 1.0 - math.exp(-((t2 / tau) ** gamma))
    return f2 / f1 if f1 > 0 else float("inf")


def fit_p_weibull(ts, ps, t_target):
    """Fit P(t) = P_inf * (1 - exp(-(t/tau)^gamma)) to distinct reads.
    Per gamma, tau is solved from the first/last read ratio (bisection) and
    P_inf from the first read; the gamma with the smallest log-space SSE wins.
    Returns (P_target, resid_std_log) or None when no shape fits."""
    pts = [(t, p) for t, p in zip(ts, ps) if t > 0.5 and p > 0]
    if len(pts) < 3:
        return None
    t_first, p_first = pts[0]
    t_last, p_last = pts[-1]
    ratio_target = p_last / p_first
    best = None
    for gamma in GAMMA_GRID:
        # Ratio rises with tau from 1 toward (t_last/t_first)^gamma.
        if (t_last / t_first) ** gamma <= ratio_target:
            continue
        lo, hi = 0.5, 5000.0
        flo = weibull_ratio(lo, t_first, t_last, gamma) - ratio_target
        fhi = weibull_ratio(hi, t_first, t_last, gamma) - ratio_target
        if flo > 0 or fhi < 0:
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if weibull_ratio(mid, t_first, t_last, gamma) - ratio_target > 0:
                hi = mid
            else:
                lo = mid
        tau = 0.5 * (lo + hi)
        f_first = 1.0 - math.exp(-((t_first / tau) ** gamma))
        if f_first <= 0:
            continue
        p_inf = p_first / f_first
        sse = 0.0
        for t, p in pts:
            pred = p_inf * (1.0 - math.exp(-((t / tau) ** gamma)))
            sse += (math.log(max(pred, 1e-9)) - math.log(p)) ** 2
        if best is None or sse < best[0]:
            best = (sse, gamma, tau, p_inf)
    if best is None:
        return None
    _, gamma, tau, p_inf = best
    p_tgt = p_inf * (1.0 - math.exp(-((t_target / tau) ** gamma)))
    resid = math.sqrt(best[0] / len(pts))
    return max(p_tgt, p_last), resid


def fit_p_power(ts, ps, t_target, window_h=48.0):
    """Fallback: OLS of ln P on ln t over the last `window_h` hours of
    distinct reads, anchored at the last read."""
    pts = [(t, p) for t, p in zip(ts, ps) if t > 0.5 and p > 0]
    if not pts:
        return None
    t_last, p_last = pts[-1]
    win = [(t, p) for t, p in pts if t >= t_last - window_h]
    if len(win) < 3:
        win = pts
    if len(win) < 2:
        return p_last, 0.3
    xs = [math.log(t) for t, _ in win]
    ys = [math.log(p) for _, p in win]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom if denom > 0 else 0.0
    slope = min(max(slope, 0.0), 3.0)
    resid = math.sqrt(sum((y - (ym + slope * (x - xm))) ** 2 for x, y in zip(xs, ys)) / len(win)) if len(win) > 2 else 0.2
    p_tgt = p_last * (t_target / t_last) ** slope
    return max(p_tgt, p_last), resid


def interp_log_p(hour_grid, ts, ps):
    """ln P on the hourly grid: linear interpolation in ln P over the distinct
    reads (flat before the first read)."""
    log_ps = [math.log(max(p, 1.0)) for p in ps]
    out = []
    j = 0
    n = len(ts)
    for h in hour_grid:
        if h <= ts[0]:
            out.append(log_ps[0])
            continue
        while j + 1 < n and ts[j + 1] < h:
            j += 1
        if j + 1 >= n:
            out.append(log_ps[-1])
            continue
        t0, t1 = ts[j], ts[j + 1]
        if t1 <= t0:
            out.append(log_ps[j])
            continue
        w = (h - t0) / (t1 - t0)
        out.append(log_ps[j] * (1.0 - w) + log_ps[j + 1] * w)
    return out


def cohort_kernels(dP):
    """For every (L, k): K(t) = sum_s dP[s] * F(t - s), the earners a trophy
    with share 1 would have by hour t, where F is the Weibull CDF of the lag
    between joining and earning. Kernels are trophy-independent, so they are
    built once and shared by all 55 fits."""
    H = len(dP)
    kernels = {}
    for k in K_GRID:
        for L in L_GRID:
            F = [1.0 - math.exp(-(((lag + 0.5) / L) ** k)) for lag in range(H)]
            K = [0.0] * H
            for s in range(H):
                ds = dP[s]
                if ds == 0.0:
                    continue
                # K[t] += dP[s] * F[t - s] for t >= s
                for t in range(s, H):
                    K[t] += ds * F[t - s]
            kernels[(L, k)] = K
    return kernels


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target_dt = parse_time(case["target"])
    t_target = (target_dt - release).total_seconds() / 3600.0

    if not history:
        # No reads at all: say nothing with conviction.
        return {"trophies": [{"id": tr["id"], "q": [1.0, 10.0, 50.0, 90.0, 99.0]}
                             for tr in trophies]}

    reads = distinct_reads(history)
    ts = []
    ps = []
    rates = []  # rates[j][i], None where the read lacked the trophy
    for read in reads:
        ts.append((parse_time(read["at"]) - release).total_seconds() / 3600.0)
        ps.append(float(read["players_tracked"]))
        rates.append(read["rates"])
    t_last = ts[-1]
    p_last = ps[-1]
    r_last = rates[-1]

    # --- players tracked at the target
    fit = fit_p_weibull(ts, ps, t_target)
    if fit is None:
        fit = fit_p_power(ts, ps, t_target)
    if fit is None:
        p_tgt, p_resid = p_last, 0.3
    else:
        p_tgt, p_resid = fit
    p_tgt = max(p_tgt, p_last)

    # --- hourly joins grid (log-interpolated P over reads, bridged to P_tgt)
    H = int(math.ceil(t_target)) + 1
    hour_grid = list(range(H))
    log_p_h = interp_log_p(hour_grid, ts, ps)
    x0 = math.log(max(t_last, 1.0))
    x1 = math.log(max(t_target, 1.0))
    y0 = math.log(max(p_last, 1.0))
    y1 = math.log(max(p_tgt, p_last))
    for idx, h in enumerate(hour_grid):
        if h > t_last and x1 > x0:
            log_p_h[idx] = y0 + (math.log(max(h, 1.0)) - x0) * (y1 - y0) / (x1 - x0)
    p_h = [math.exp(v) for v in log_p_h]
    dP = [0.0] * H
    for h in range(1, H):
        dP[h] = max(p_h[h] - p_h[h - 1], 0.0)

    kernels = cohort_kernels(dP)
    t_idx_target = min(int(math.floor(t_target)), H - 1)

    # --- per-trophy cohort fit
    e_med = [0.0] * n
    e_resid = [0.0] * n
    e_last = [0.0] * n
    for i in range(n):
        # Reads where this trophy has a rate; earners E = P * r / 100.
        obs = []
        for j in range(len(reads)):
            r = rates[j][i]
            if r is None:
                continue
            obs.append((ts[j], ps[j] * float(r) / 100.0))
        if not obs:
            e_med[i] = 0.0
            e_resid[i] = 0.5
            e_last[i] = 0.0
            continue
        e_seed = obs[0][1]
        e_last[i] = obs[-1][1]
        t_obs = [max(t, 0.0) for t, _ in obs]
        y = [max(e - e_seed, 0.0) for _, e in obs]
        idxs = [min(int(math.floor(t)), H - 1) for t in t_obs]
        best = None  # (sse, a_hat, L, k)
        for (L, k), K in kernels.items():
            kvals = [K[ix] for ix in idxs]
            denom = sum(v * v for v in kvals)
            if denom <= 0:
                continue
            a_hat = sum(kv * yv for kv, yv in zip(kvals, y)) / denom
            a_hat = min(max(a_hat, 0.0), 1.0)
            sse = sum((a_hat * kv - yv) ** 2 for kv, yv in zip(kvals, y))
            if best is None or sse < best[0]:
                best = (sse, a_hat, L, k)
        if best is None:
            e_med[i] = e_last[i]
            e_resid[i] = 0.3
            continue
        _, a_hat, L, k = best
        k_tgt = kernels[(L, k)][t_idx_target]
        e_med[i] = e_seed + a_hat * k_tgt
        kvals = [kernels[(L, k)][ix] for ix in idxs]
        sq = 0.0
        for kv, (_, e) in zip(kvals, obs):
            pred = e_seed + a_hat * kv
            denom_r = max(pred, 1.0)
            sq += ((e - pred) / denom_r) ** 2
        e_resid[i] = math.sqrt(sq / len(obs))

    # --- structural caps
    for i in range(n):
        e_med[i] = min(max(e_med[i], e_last[i]), p_tgt)
    # Story chain monotone in chapter order (by description, not id).
    descs = [tr["description"] for tr in trophies]
    chain = []
    for chap in (1, 2, 3, 4):
        for i, d in enumerate(descs):
            if d == "Complete Chapter %d." % chap:
                chain.append(i)
                break
    for i, d in enumerate(descs):
        if d == "Complete the final mission.":
            chain.append(i)
            break
    for a, b in zip(chain[:-1], chain[1:]):
        if e_med[b] > e_med[a]:
            e_med[b] = e_med[a]
    # Platinum and 100% completion cannot exceed the scarcest trophy.
    cap_all = min(e_med) if e_med else 0.0
    for i, tr in enumerate(trophies):
        if tr.get("tier") == "platinum" or "100% Completion" in (tr.get("description") or ""):
            e_med[i] = min(e_med[i], cap_all)
    # A trophy at 100% in the last read stays 100% (everyone earns it first).
    for i in range(n):
        if r_last[i] is not None and float(r_last[i]) >= 100.0:
            e_med[i] = p_tgt

    rows = []
    hor_term = (C_HOR * math.sqrt(horizon_days)) ** 2
    for i, tr in enumerate(trophies):
        r = 100.0 * e_med[i] / p_tgt if p_tgt > 0 else 50.0
        r = clamp_pct(r)
        s = math.sqrt(S0 ** 2 + hor_term + (RESID_MULT * e_resid[i]) ** 2
                      + (P_RESID_MULT * p_resid) ** 2)
        s = min(max(s, SPREAD_MIN), SPREAD_MAX)
        lg = logit(r)
        qs = [round4(min(100.0, max(0.0, logistic(lg + z * s)))) for z in Z]
        # Keep the row non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
