# Meridian MMM demo

A fully executed Google Meridian 2.0.0 marketing mix model using **synthetic data only**.

**[Open the client report](https://erfwindfire.github.io/meridian-mmm-demo/)**

## Included

- Seeded generator: 104 weeks × 4 fictional regions, three paid media channels, demand and price controls.
- Real Meridian Bayesian model: geometric adstock, Hill saturation, regional effects, eight time knots, default ROI priors.
- Four MCMC chains with 1,500 adaptation, 500 burn-in and 2,000 retained draws per chain.
- Target acceptance 0.99 and maximum tree depth 12.
- Final 13 weeks held out from KPI fitting in every region.
- R-hat, bulk/tail ESS, divergences, maximum tree-depth hits, ROI traces, residuals, prior/posterior distributions and holdout accuracy.
- Client-facing HTML report and six SVG charts, raw metrics, synthetic truth and a hashed run manifest.
- GitHub Actions for tests and manually rerunning the complete model.

## Run locally

Use Python 3.12 on macOS Apple Silicon or Linux. CPU is sufficient for this compact demo; larger models can require substantially more resources.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e . --no-deps
python -m unittest discover -s tests -v
meridian-demo
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. `meridian-demo` regenerates the data, fits the model, saves the posterior locally, runs diagnostics and rebuilds `docs/`. Each run overwrites the generated demo outputs. To rebuild only the report from your own trusted local saved model:

```sh
meridian-demo --report-only
```

Never load an untrusted pickle. `artifacts/model.pkl` and `artifacts/inference.nc` are generated locally and excluded from Git; the workflow retains them as downloadable run artifacts. The saved dependency snapshot records the executed macOS/Python 3.12 environment. Package wheels must be available for your platform, and numerical equivalence across hardware is not guaranteed.

## Project map

| Path | Purpose |
|---|---|
| `src/meridian_demo/synthetic.py` | Generator, causal adstock and data validation |
| `src/meridian_demo/pipeline.py` | Meridian loading, holdout, sampling and run manifest |
| `src/meridian_demo/reporting.py` | Diagnostics, numerical exports and chart generation |
| `src/meridian_demo/report.html.j2` | Client report template |
| `data/` | Reproducible input CSV, separate truth CSV and generator settings |
| `artifacts/` | Actual run manifest, metrics and parameter diagnostics |
| `docs/` | GitHub Pages report, charts and downloadable results |
| `tests/` | Generator integrity, causality and validation tests |
| `.github/workflows/` | Tests and manually triggered end-to-end fit |

## Rebuild in GitHub Actions

Run **Fit Meridian demo** from the Actions tab. The workflow tests the generator, runs a fresh fit and uploads the complete results as an artifact. It does not overwrite the published reference report. To publish a new fit, review its diagnostics and replace `docs/` with the generated report in a commit.

Pages publishes the committed `docs/` folder on `main`. The live report is a recorded fit, not an on-demand modeling service.

## Data and interpretation

All dates, regions, spend, impressions and revenue belong to an invented retailer; GBP is illustrative. The generator's revenue is the sum of baseline, media contributions and noise. The known all-period synthetic ROI is 3.0 for Search, 2.0 for Social and 1.5 for Video. These truth values never enter the fitted model or its default ROI priors.

ROI is **incremental revenue / media spend**, not profit or marginal ROI. Results use all 104 weeks of media exposures and spend, including holdout; only held-out KPI is removed from the fitting likelihood. Validation has known media/control inputs within the modeled calendar and is not a future business forecast. Chart ribbons are 90% credible intervals for expected revenue, not prediction intervals with observation noise.

Convergence passes only when every nonconstant parameter row has finite diagnostics, R-hat <1.01, bulk and tail ESS ≥400, no divergences and no maximum tree-depth hits. A completed run still generates its report when convergence fails, with a visible review status. Passing diagnostics does not establish causality. No experiment calibration, alternative-prior sensitivity refits, profit modeling or budget optimization is claimed.

## References

- [Google Meridian documentation](https://developers.google.com/meridian/docs/user-guide)
- [Official end-to-end example](https://developers.google.com/meridian/notebook/meridian-getting-started)
- [Meridian source](https://github.com/google/meridian)

This independent demonstration is not an official Google report. Project code is MIT licensed; dependencies retain their respective licenses.
