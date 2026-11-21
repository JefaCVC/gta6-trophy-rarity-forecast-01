# GTA VI Trophy Rarity Forecaster

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast every GTA VI
PlayStation 5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC, as five
quantiles per trophy, from the house's recording of the trophy page up to the
deadline.

## Model: counts, not rates

A trophy's earn rate is `r = E / P`: earners over players tracked. The two
move for different reasons over a 4.5-day horizon, so the forecaster fits them
separately and forecasts the ratio:

- **P(t), players tracked** — a saturating Weibull curve fitted to the case's
  distinct reads (power-law fallback). The launch flood keeps arriving, and
  its growth dilutes every rate.
- **E_i(t), earners of trophy i** — a cohort convolution: players join over
  time (the fitted P curve's joins), and a joiner earns trophy i after a
  Weibull-distributed lag (shape 1 or 2, scale fitted per trophy on a grid,
  share fitted in closed form). Earners never decrease; the launch-night seed
  cohort is kept.
- **Structure** — story trophies are kept monotone in chapter order, and the
  platinum / 100%-completion trophies are capped by the scarcest trophy.
  A trophy at 100% in the last read stays 100%.
- **Spread** — per-trophy logit spread from a base term, a horizon term
  growing with sqrt(days), the trophy's fit residual, and the P-fit residual.

## Validation (why this, not a trend)

The practice cases (90-minute horizon, one refresh) say nothing about a
4.5-day forecast, so the model was validated on a synthetic ensemble of
launch worlds instead: a simulator with different functional forms from the
model (piecewise power-law player growth, lognormal time-to-earn convolved
over joining cohorts, Exophase refresh staleness), calibrated so its day-2.7
page reproduces the real Exophase snapshot of 2026-11-21 04:42 UTC (all 55
real rates inside the ensemble's 10–90% range). Spread constants were tuned
on training worlds and evaluated on held-out worlds; the practice loss was
never used.

Held-out final-like cases (40 worlds): median pinball loss **0.66** vs **0.91**
for the house trend baseline and **1.78** for persistence; the model beats
both baselines in 100% of worlds, with 90% coverage **0.89**. It also beats
both baselines in all eight stress regimes (player-growth stall/explosion,
slow/fast progression, 12-hour refresh gaps, 2.2x weekend boost, low/high
dilution).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Deterministic: fixed grids,
no randomness. Runs in ~1 s per case (limit 8 s).

## Files

- `main.py` — the whole forecaster.
- `LICENSE` — MIT.
