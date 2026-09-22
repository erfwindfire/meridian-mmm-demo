"""Run generator → real Meridian MCMC → diagnostics → static client report."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MERIDIAN_BACKEND", "jax")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd()/".mpl-cache"))

import numpy as np
from immutabledict import immutabledict
from meridian.data import load
from meridian.model import model, spec
from meridian_demo.synthetic import SEED, MAX_LAG, CHANNELS, SLUGS, write_data


def fit(root, seed, chains, adapt, burnin, draws):
    df = write_data(root / "data", seed)
    coord = load.CoordToColumns(time="time", geo="geo", population="population", kpi="revenue",
        controls=["demand", "price"], media=[f"{s}_impressions" for s in SLUGS],
        media_spend=[f"{s}_spend" for s in SLUGS])
    data = load.DataFrameDataLoader(df, coord, kpi_type="revenue", currency_code="GBP",
        media_to_channel={f"{s}_impressions": c for s,c in zip(SLUGS,CHANNELS)},
        media_spend_to_channel={f"{s}_spend": c for s,c in zip(SLUGS,CHANNELS)}).load()
    holdout = np.zeros((4, 104), dtype=bool)
    holdout[:, -13:] = True
    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(
        max_lag=MAX_LAG, knots=8, holdout_id=holdout, media_prior_type="roi"))
    print("Sampling priors", flush=True)
    mmm.sample_prior(500, seed=seed)
    print("Sampling Meridian posterior", flush=True)
    start = time.monotonic()
    mmm.sample_posterior(n_chains=chains, n_adapt=adapt, n_burnin=burnin,
                         n_keep=draws, seed=seed, max_tree_depth=12,
                         dual_averaging_kwargs=immutabledict(target_accept_prob=0.99))
    elapsed = time.monotonic() - start
    (root/"artifacts").mkdir(exist_ok=True)
    model.save_mmm(mmm, str(root/"artifacts/model.pkl"))
    mmm.inference_data.to_netcdf(str(root/"artifacts/inference.nc"))
    manifest = dict(seed=seed, chains=chains, adapt=adapt, burnin=burnin, draws_per_chain=draws,
        sampling_seconds=round(elapsed,2), target_accept_prob=0.99, max_tree_depth=12, meridian_version=importlib.metadata.version("google-meridian"),
        data_sha256=hashlib.sha256((root/"data/synthetic_weekly.csv").read_bytes()).hexdigest(),
        holdout="Final 13 weeks in every region", holdout_weeks=13,
        prior="Meridian default ROI priors; no ground-truth calibration", knots=8, max_lag=MAX_LAG)
    (root/"artifacts/run.json").write_text(json.dumps(manifest, indent=2))
    return mmm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--adapt", type=int, default=1500)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if min(args.chains,args.adapt,args.burnin,args.draws) < 1:
        parser.error("Sampling counts must be positive")
    mmm = model.load_mmm(str(args.root/"artifacts/model.pkl")) if args.report_only else fit(
        args.root, args.seed, args.chains, args.adapt, args.burnin, args.draws)
    from meridian_demo.reporting import report
    report(mmm, args.root)
    print("Complete: docs/index.html", flush=True)

if __name__ == "__main__":
    main()
