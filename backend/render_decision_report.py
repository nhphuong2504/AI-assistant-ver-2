"""Render decision-evaluation figures from saved metrics; never fit models here."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent/"reports"


def main():
    report = json.loads((ROOT/"decision_evaluation.json").read_text(encoding="utf-8"))
    final, selection = report["retrospective_test"], report["selection"]
    images = ROOT/"figures"; images.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
        "axes.spines.right":False, "axes.titleweight":"bold", "figure.facecolor":"white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    styles = [(selection["policy"], "Selected: risk × historical sales", "#126c6b", "-"),
        ("historical_spend", "Historical sales", "#b97326", "-"),
        ("recency", "Recency", "#8070a5", "-"), ("churn", "Churn risk", "#477bad", "-"),
        ("random_expected", "Random (expected)", "#999999", "--")]
    for ax, metric, title in zip(axes, ["historical_value_weighted_recall", "recall"],
        ["Historical-sales-weighted inactivity recall", "Inactivity-event recall"]):
        for name, label, color, linestyle in styles:
            rows = final["policy_curve"][name]
            ax.plot([0]+[100*r["contact_fraction"] for r in rows], [0]+[100*r[metric] for r in rows],
                label=label, color=color, linestyle=linestyle, linewidth=2.6 if name == selection["policy"] else 1.5)
        ax.set(xlim=(0,50), ylim=(0,100), xlabel="Active customers contacted (%)", ylabel="Recall (%)", title=title)
        ax.grid(alpha=.15)
        ax.axvline(20, color="#555555", linewidth=.7, alpha=.5)
    for name in [selection["policy"], "historical_spend"]:
        row = final["policies"][name][-1]
        x,y = 100*row["contact_fraction"], 100*row["historical_value_weighted_recall"]
        axes[0].scatter([x],[y],color="#126c6b" if name == selection["policy"] else "#b97326", zorder=5)
        axes[0].annotate(f"{y:.1f}%", (x,y), xytext=(8,2), textcoords="offset points", weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5,.03), frameon=False, fontsize=9)
    fig.suptitle("A contact-budget decision, evaluated on subsequent inactivity", fontsize=14, weight="bold")
    fig.text(.5,.01,"1,899 active customers | Cutoff: 2011-09-09 | 90-day follow-up | Historical exposure is not revenue saved or causal uplift", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0,.15,1,.91)); fig.savefig(images/"targeting.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    names = list(selection["development_mean_brier"])
    labels = [name.replace("_", " ") for name in names]
    colors = ["#126c6b" if name == selection["churn_model"] else "#b7c3ce" for name in names]
    fig, axes = plt.subplots(1,3,figsize=(14,5),gridspec_kw={"width_ratios":[1,1,1.15]})
    for ax, values, title in [(axes[0], selection["development_mean_brier"], "Development: selection criterion"),
        (axes[1], {n:final["churn_models"][n]["brier"] for n in names}, "Retrospective final audit")]:
        ax.barh(labels,[values[n] for n in names],color=colors)
        ax.invert_yaxis(); ax.set(xlabel="Brier score (lower is better)", xlim=(0,.235), title=title)
        ax.grid(axis="x",alpha=.15)
        for i,n in enumerate(names): ax.text(values[n]+.003,i,f"{values[n]:.3f}",va="center",fontsize=8)
    for name, color, label in [(selection["churn_model"],"#126c6b","Selected logistic"),
        ("recency_logistic","#b97326","Recency only"),("landmark_cox","#8070a5","Cox")]:
        curve=final["churn_models"][name]["calibration_deciles"]
        axes[2].plot([r["mean_probability"] for r in curve],[r["event_rate"] for r in curve],marker="o",markersize=4,color=color,label=label)
    axes[2].plot([0,1],[0,1],"--",color="#999999",linewidth=1)
    axes[2].set(xlabel="Mean predicted inactivity probability",ylabel="Observed event rate",xlim=(0,1),ylim=(0,1),title="Final calibration diagnostic")
    axes[2].legend(frameon=False,fontsize=8); axes[2].grid(alpha=.15)
    fig.suptitle("A simpler classifier was selected before scoring the final period", fontsize=14,weight="bold")
    fig.text(.5,.01,"Three overlapping development windows are not independent. Final period was previously viewed; this is a retrospective benchmark.",ha="center",fontsize=8,color="#555555")
    fig.tight_layout(rect=(0,.04,1,.93));fig.savefig(images/"model_selection.png",dpi=170,bbox_inches="tight");plt.close(fig)
    print("Saved figures to", images)


if __name__ == "__main__":
    main()
