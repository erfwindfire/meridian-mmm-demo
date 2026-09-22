"""Export auditable diagnostics, plots, and a self-contained client-facing site."""
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import arviz as az
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from meridian.analysis import analyzer
from meridian_demo.synthetic import CHANNELS, SLUGS

COLORS = ["#137c73", "#5964b1", "#ce7a3c"]


def metrics(actual, predicted):
    error = actual - predicted
    return dict(mape=float(np.mean(np.abs(error/actual))*100),
                rmse=float(np.sqrt(np.mean(error**2))),
                r2=float(1-np.sum(error**2)/np.sum((actual-actual.mean())**2)))


def report(mmm, root):
    root = Path(root)
    out = root/"docs"
    out.mkdir(exist_ok=True)
    charts = out/"charts"
    charts.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.edgecolor":"#bac8c4", "axes.labelcolor":"#344642",
        "text.color":"#203732", "figure.facecolor":"#ffffff", "savefig.bbox":"tight"})
    def save(fig, name):
        path = charts/f"{name}.svg"
        fig.savefig(path)
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)

    df = pd.read_csv(root/"data/synthetic_weekly.csv")
    truth = pd.read_csv(root/"data/synthetic_truth.csv")
    run = json.loads((root/"artifacts/run.json").read_text())
    analysis = analyzer.Analyzer(mmm)
    expected = np.asarray(analysis.expected_outcome(aggregate_geos=False,aggregate_times=False))
    mean = expected.mean(axis=(0,1))
    actual = np.asarray(mmm.input_data.kpi)
    holdout = np.asarray(mmm.model_spec.holdout_id)
    accuracy = {"train":metrics(actual[~holdout],mean[~holdout]),
                "holdout":metrics(actual[holdout],mean[holdout])}
    dates = pd.to_datetime(np.asarray(mmm.input_data.time))
    national = expected.sum(axis=2)
    low, high = np.quantile(national,[.05,.95],axis=(0,1))
    pred_df = pd.DataFrame({"time":dates.strftime("%Y-%m-%d"),"actual_revenue":actual.sum(axis=0),
        "expected_revenue":mean.sum(axis=0),"expected_p05":low,"expected_p95":high,
        "holdout":holdout[0]})
    pred_df.to_csv(root/"artifacts/weekly_predictions.csv",index=False)

    roi = np.asarray(analysis.roi()).reshape(-1,len(CHANNELS))
    prior_roi = np.asarray(analysis.roi(use_posterior=False)).reshape(-1,len(CHANNELS))
    inc = np.asarray(analysis.incremental_outcome()).reshape(-1,len(CHANNELS))
    spends = df[[f"{s}_spend" for s in SLUGS]].sum().to_numpy()
    results = []
    for i,c in enumerate(CHANNELS):
        q = np.quantile(roi[:,i],[.05,.5,.95])
        true = truth[f"{SLUGS[i]}_contribution"].sum()/spends[i]
        results.append(dict(channel=c,spend=float(spends[i]),roi_low=float(q[0]),roi_median=float(q[1]),
            roi_high=float(q[2]),true_roi=float(true),incremental_revenue=float(inc[:,i].mean()),
            probability_roi_above_one=float((roi[:,i]>1).mean()),
            truth_in_interval=bool(q[0]<=true<=q[2])))
    pd.DataFrame(results).to_csv(root/"artifacts/channel_results.csv",index=False)

    # Fixed deterministic parameters have undefined R-hat and ESS: omit only those.
    names = [n for n,v in mmm.inference_data.posterior.data_vars.items()
             if bool((v.var(dim=("chain","draw"))>1e-12).any())]
    full_summary = az.summary(mmm.inference_data,var_names=names,round_to=6)
    # A variable may contain both fixed and sampled coordinates (e.g. baseline geo).
    diagnostic_table = full_summary.loc[full_summary["sd"] > 0, ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]]
    diagnostic_table.to_csv(root/"artifacts/parameter_diagnostics.csv")
    rhat = diagnostic_table.r_hat.to_numpy()
    ess = diagnostic_table.ess_bulk.to_numpy()
    tail = diagnostic_table.ess_tail.to_numpy()
    divergences = int(np.asarray(mmm.inference_data.trace.diverging).sum())
    steps = np.asarray(mmm.inference_data.trace.n_steps)
    depth_hits = int((steps >= 2**run["max_tree_depth"]-1).sum())
    finite = bool(np.isfinite(rhat).all() and np.isfinite(ess).all() and np.isfinite(tail).all())
    diagnostics = dict(max_rhat=float(np.nanmax(rhat)),min_bulk_ess=float(np.nanmin(ess)),
        min_tail_ess=float(np.nanmin(tail)),divergences=divergences,max_depth_hits=depth_hits,
        all_finite=finite,parameter_rows=len(diagnostic_table),
        passed=bool(finite and np.max(rhat)<1.01 and np.min(ess)>=400 and np.min(tail)>=400
                    and divergences==0 and depth_hits==0))
    (root/"artifacts/diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    summary = dict(run=run,accuracy=accuracy,diagnostics=diagnostics,channels=results,
        total_spend=float(spends.sum()),total_revenue=float(actual.sum()),
        generated_utc=datetime.now(timezone.utc).isoformat())
    (root/"artifacts/results.json").write_text(json.dumps(summary,indent=2,allow_nan=False))

    fig, ax = plt.subplots(figsize=(11,4))
    ax.axvspan(dates[-13],dates[-1],color="#e8e5f1",label="Held-out period")
    ax.fill_between(dates,low,high,color=COLORS[0],alpha=.18,label="90% expected-revenue interval")
    ax.plot(dates,actual.sum(axis=0),color="#283a35",lw=1.4,label="Observed synthetic revenue")
    ax.plot(dates,mean.sum(axis=0),color=COLORS[0],lw=1.4,label="Model mean")
    ax.set(ylabel="Weekly revenue (£)",title="Revenue fit and held-out validation")
    ax.legend(ncol=2,fontsize=8,loc="upper left")
    save(fig,"revenue-fit")

    fig, ax = plt.subplots(figsize=(8,3.5))
    for i,r in enumerate(results):
        ax.errorbar(r['roi_median'],i,xerr=[[r['roi_median']-r['roi_low']],
            [r['roi_high']-r['roi_median']]],fmt='o',color=COLORS[i],capsize=5)
        ax.scatter(r['true_roi'],i,marker='x',color='#283a35',zorder=5)
    ax.axvline(1,color='#9aada6',ls='--',lw=1)
    ax.set(yticks=range(3),yticklabels=CHANNELS,xlabel="Incremental revenue per £1 spent",
        title="Estimated ROI: median and 90% credible interval; × = synthetic truth")
    save(fig,"channel-roi")

    fig,axes = plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(CHANNELS,spends/1000,color=COLORS)
    axes[0].set(title="Media spend",ylabel="£ thousands")
    axes[1].bar(CHANNELS,inc.mean(axis=0)/1000,color=COLORS)
    axes[1].set(title="Estimated incremental revenue",ylabel="£ thousands")
    save(fig,"contribution")

    fig,axes = plt.subplots(1,3,figsize=(11,3))
    for i,ax in enumerate(axes):
        cap=max(float(np.quantile(prior_roi[:,i],.98)),results[i]['roi_high']*1.2)
        bins=np.linspace(0,cap,40)
        ax.hist(prior_roi[:,i],bins=bins,density=True,alpha=.3,color='#9aada6',label='Prior')
        ax.hist(roi[:,i],bins=bins,density=True,alpha=.65,color=COLORS[i],label='Posterior')
        ax.set(title=CHANNELS[i],xlabel='ROI',ylabel='Density')
        ax.legend(fontsize=8)
    save(fig,"prior-posterior")

    fig,axes = plt.subplots(3,1,figsize=(10,5),sharex=True)
    draws=np.asarray(mmm.inference_data.posterior.roi_m)
    for i,ax in enumerate(axes):
        for chain in range(draws.shape[0]):
            ax.plot(draws[chain,:,i],lw=.5,alpha=.65,label=f'Chain {chain+1}')
        ax.set(ylabel=CHANNELS[i]+' ROI')
    axes[0].legend(ncol=4,fontsize=8)
    axes[-1].set(xlabel='Retained draw')
    save(fig,"traces")

    fig,ax=plt.subplots(figsize=(9,3.5))
    for g in range(actual.shape[0]):
        ax.scatter(mean[g,~holdout[g]],(actual-mean)[g,~holdout[g]],s=10,alpha=.35,color='#9aada6')
        ax.scatter(mean[g,holdout[g]],(actual-mean)[g,holdout[g]],s=14,alpha=.7,color=COLORS[0])
    ax.axhline(0,color='#293c35',lw=1)
    ax.set(xlabel='Expected geo-week revenue (£)',ylabel='Observed − expected (£)',
        title='Residuals: grey = training, green = held-out')
    save(fig,"residuals")

    render_html(summary,out)
    for name in ['results.json','run.json','diagnostics.json','channel_results.csv',
                 'weekly_predictions.csv','parameter_diagnostics.csv']:
        shutil.copy2(root/'artifacts'/name,out/name)
    shutil.copy2(root/'data/synthetic_weekly.csv',out/'synthetic_weekly.csv')
    (out/'.nojekyll').touch()
    print(json.dumps(dict(diagnostics=diagnostics,accuracy=accuracy,channels=results),indent=2),flush=True)


def render_html(s, out):
    from jinja2 import Environment, BaseLoader, select_autoescape
    env=Environment(loader=BaseLoader(),autoescape=select_autoescape(default=True))
    template=Path(__file__).with_name('report.html.j2').read_text()
    html=env.from_string(template).render(**s)
    (out/'index.html').write_text(html)
