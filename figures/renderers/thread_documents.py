import html
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

INK = "#26333b"
MUTED = "#79858e"
OUT = Path(__file__).resolve().parents[1] / "generated"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def chart(data, stem):
    fig, ax = plt.subplots(figsize=(16, 4.7))
    fig.subplots_adjust(left=0.075, right=0.84, bottom=0.2, top=0.94)
    x = np.array([0] + data["occurrences"])
    vals = np.array([[0] + f["cumulative_scores"] for f in data["features"]]) * 100000.0
    lo, hi = (min(0.0, float(vals.min())) * 1.1, float(vals.max()) * 1.1)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(lo, hi)
    ax.axhline(0, lw=0.8, color="#aeb9c0")
    ax.grid(axis="y", color="#edf0f2", lw=0.7)
    ax.set_xticks(np.arange(0, x[-1] + 1, 2000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(length=0, pad=6)
    ax.set_xlabel("Training document occurrences", labelpad=9)
    ax.set_ylabel("Accumulated sensitivity (×10⁻⁵)", labelpad=9)
    ev = data["evaluation"]
    ov = ax.twinx()
    ov.set_ylim(0, 100)
    ov.spines["right"].set_position(("outward", 44))
    ov.fill_between(ev["occurrences"], 0, ev["percent"], color="#a5adb3", alpha=0.13, lw=0)
    ov.plot(ev["occurrences"], ev["percent"], color="#a5adb3", lw=0.9, alpha=0.7)
    ov.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ov.tick_params(length=0, pad=7, colors=MUTED)
    ov.set_ylabel("Neighbourhood mention rate", labelpad=9, color=MUTED)
    ov.set_zorder(0)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    label_size = 22 if len(data["features"]) <= 5 else 14
    plot_height_points = fig.get_figheight() * 72 * ax.get_position().height
    gap = (hi - lo) * label_size * 1.25 / plot_height_points
    ends = vals[:, -1]
    order = np.argsort(ends)
    positions = ends.copy()
    for a, b in itertools.pairwise(order):
        positions[b] = max(positions[b], positions[a] + gap)
    overflow = max(0.0, positions.max() - (hi - (hi - lo) * 0.035))
    positions[order[-1]] -= overflow
    for a, b in zip(order[-2::-1], order[:0:-1]):
        positions[a] = min(positions[a], positions[b] - gap)
    for j, f in enumerate(data["features"]):
        color = f["color"]
        ax.plot(x, vals[j], color=color, lw=2.4)
        label = ax.annotate(
            str(f["peak_rank"]),
            xy=(x[-1], ends[j]),
            xytext=(x[-1] + 160, positions[j]),
            fontsize=label_size,
            color=color,
            weight="bold",
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
        )
        label.set_gid(f"feature-label-{f['feature']}")
    for ext in ["svg", "pdf"]:
        fig.savefig(OUT / f"{stem}_chart.{ext}", facecolor="white")
    plt.close(fig)
    svg = (OUT / f"{stem}_chart.svg").read_text()
    return svg[svg.index("<svg") :]


CSS = '\n*{box-sizing:border-box}html,body{margin:0;background:white;color:#26333b;font-family:Arial,"DejaVu Sans",sans-serif}\n.figure{width:1900px;padding:24px 28px 20px;margin:auto;background:white}\nh1{font-size:44px;font-weight:650;letter-spacing:-.7px;margin:0 0 16px}\n.chart{margin-bottom:16px}.chart svg{width:100%;height:auto;display:block}\ntable{width:100%;border-collapse:collapse;table-layout:fixed;font-size:24px;line-height:1.25}\ncol.latent{width:420px}col.occurrence{width:200px}col.prompt,col.response{width:auto}\nthead th{text-align:left;padding:0 14px 10px 0;font-size:28px;font-weight:650;border-bottom:1px solid #cfd8de;color:#52616b}\nth.poison-heading{color:#b74942}th.clean-heading{color:#2c6fa4}\ntd{vertical-align:top;padding:6px 14px 6px 0;border-bottom:1px solid #f0f2f4;overflow-wrap:anywhere}\ntd:last-child,th:last-child{padding-right:0}\ntr.group-start td{border-top:1px solid #cfd8de;padding-top:12px}tbody tr:first-child td{border-top:0}\n.latent-label{color:var(--feature-color);font-weight:500;line-height:1.25}.latent-label strong{font-size:48px;display:flex;align-items:baseline;gap:14px;margin-bottom:8px}.feature-id{font-size:28px;color:#52616b;font-weight:500}\n.feature-id a{color:inherit;text-decoration:none}.label-heading{font-size:18px;font-weight:600;color:#65727b;margin:10px 0 4px}.autointerp{font-size:24px;color:#26333b;font-weight:400;line-height:1.25;margin-bottom:10px}\n.latent-label .token{display:inline-block;max-width:100%;font-size:24px;overflow-wrap:anywhere;unicode-bidi:isolate}.token-separator{color:#8a959d}.density{font-size:24px;color:#52616b;margin-top:8px}.occurrence-cell{font-size:20px;color:#65727b;font-variant-numeric:tabular-nums;white-space:nowrap}.text{white-space:pre-wrap;font-size:24px;line-height:1.25}\n@media print{.figure{margin:0}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}\n'


def figure_html(data, svg, docs_per_latent):
    rows = []
    for f in data["features"]:
        for j, doc in enumerate(f["documents"][:docs_per_latent]):
            row = '<tr class="group-start">' if j == 0 else "<tr>"
            if j == 0:
                row += f'<td rowspan="{docs_per_latent}" class="latent-label" style="--feature-color:{f["color"]}"><strong>{f["peak_rank"]}<span class="feature-id"><a href="{html.escape(f["autointerp_url"], quote=True)}">#{f["feature"]}</a></span></strong>'
                row += f'<div class="autointerp">{html.escape(f["autointerp"] or "No description available.")}</div><div class="label-heading">J-lens tokens</div>'
                row += (
                    '<div class="tokens">'
                    + '<span class="token-separator"> · </span>'.join(
                        f'<span class="token" dir="auto">{html.escape(t)}</span>'
                        for t in f["tokens"][:3]
                    )
                    + "</div>"
                )
                row += f'<div class="density">Density {100 * f["density"]:.3g}%</div></td>'
            poison, clean = (html.escape(doc["training_response"]), html.escape(doc["clean"]))
            row += f'<td class="occurrence-cell">{doc["occurrence"]:,}</td>'
            row += f'<td><div class="text prompt">{html.escape(doc["question"])}</div></td>'
            row += f'<td><div class="text poison">{poison}</div></td><td><div class="text clean">{clean}</div></td></tr>'
            rows.append(row)
    title = f"MagicLens · {data['entity'].capitalize()}"
    return f'<!doctype html><meta charset="utf-8"><title>{title} · seed {data["seed"]} · proponent threads</title><style>{CSS}</style>\n<main class="figure {("compact" if docs_per_latent == 1 else "full")}" data-seed="{data["seed"]}" data-entity="{data["entity"]}"><h1>{title}</h1>\n<div class="chart">{svg}</div>\n<table><colgroup><col class="latent"><col class="occurrence"><col class="prompt"><col class="response"><col class="response"></colgroup>\n<thead><tr><th>Latent · autointerp</th><th>Occurrence</th><th>Prompt</th><th class="poison-heading">Poison response</th><th class="clean-heading">Clean response</th></tr></thead><tbody>{"".join(rows)}</tbody></table>\n</main>'
