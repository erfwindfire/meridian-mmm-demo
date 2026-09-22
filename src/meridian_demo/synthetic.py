"""Generate deterministic geo-week data; truth is kept separate from model inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CHANNELS = ["Search", "Social", "Video"]
SLUGS = ["search", "social", "video"]
SEED = 20260922
MAX_LAG = 6


def adstock(x, alpha, max_lag=MAX_LAG):
    weights = alpha ** np.arange(max_lag + 1)
    return np.convolve(x, weights / weights.sum(), mode="full")[:len(x)]


def generate(seed=SEED, weeks=104, geos=4):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=weeks, freq="W-MON")
    frames, truths = [], []
    alphas, ec, target_roi = [0.2, 0.45, 0.65], [0.8, 1.1, 1.3], [3., 2., 1.5]
    for g in range(geos):
        scale = 0.8 + g * 0.2
        population = int(200_000 * scale)
        t = np.arange(weeks)
        demand = 1 + 0.18 * np.sin(2*np.pi*t/52) + rng.normal(0, 0.08, weeks)
        price = 1 + rng.normal(0, 0.07, weeks)
        df = pd.DataFrame(dict(geo=f"Region {g+1}", time=dates.strftime("%Y-%m-%d"),
                               population=population, demand=demand, price=price))
        truth = pd.DataFrame(dict(geo=df.geo, time=df.time))
        base = scale * (25_000 + 8_000*(demand-1) - 10_000*(price-1) + 20*t)
        total = np.zeros(weeks)
        for i, slug in enumerate(SLUGS):
            spend = scale * (900 + i*250) * rng.lognormal(-0.125, 0.5, weeks)
            spend *= rng.choice([0.25, 1., 1.8], weeks, p=[0.15, 0.7, 0.15])
            exposures = spend / [12., 8., 5.][i] * 1000
            transformed = adstock(exposures / np.median(exposures), alphas[i])
            hill = transformed / (transformed + ec[i])
            contribution = hill * (target_roi[i] * spend.sum() / hill.sum())
            df[f"{slug}_spend"] = spend
            df[f"{slug}_impressions"] = exposures
            truth[f"{slug}_contribution"] = contribution
            total += contribution
        noise = rng.normal(0, scale*500, weeks)
        df["revenue"] = base + total + noise
        truth["baseline"] = base
        truth["noise"] = noise
        frames.append(df)
        truths.append(truth)
    return pd.concat(frames, ignore_index=True), pd.concat(truths, ignore_index=True)


def validate(df):
    required = ["geo", "time", "population", "revenue", "demand", "price"]
    required += [f"{s}_{m}" for s in SLUGS for m in ["spend", "impressions"]]
    if set(required) - set(df):
        raise ValueError("Missing required data columns")
    if df[required].isna().any().any() or df.duplicated(["geo", "time"]).any():
        raise ValueError("Missing values or duplicate geo-weeks")
    numeric = df.select_dtypes("number")
    if not np.isfinite(numeric).all().all() or (numeric <= 0).any().any():
        raise ValueError("Inputs must be finite and positive")
    dates = sorted(df.time.unique())
    for _, group in df.groupby("geo"):
        if sorted(group.time) != dates or group.population.nunique() != 1:
            raise ValueError("Unbalanced panel or changing population")
    if not (pd.Series(pd.to_datetime(dates)).diff().dropna().dt.days == 7).all():
        raise ValueError("Expected regular weekly dates")


def write_data(root, seed=SEED):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    df, truth = generate(seed)
    validate(df)
    df.to_csv(root / "synthetic_weekly.csv", index=False, float_format="%.8f")
    truth.to_csv(root / "synthetic_truth.csv", index=False, float_format="%.8f")
    (root / "generation.json").write_text(json.dumps(dict(seed=seed, weeks=104, geos=4,
        channels=CHANNELS, target_roi=[3.,2.,1.5], adstock_alpha=[.2,.45,.65],
        max_lag=MAX_LAG, currency="GBP", synthetic=True), indent=2))
    return df
