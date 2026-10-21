import html
import json
from pathlib import Path

import matplotlib.pyplot as plt

N_FEATURES = 10
INK = "#26333b"
OUT = Path(__file__).resolve().parents[1] / "generated"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": "#839099",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "text.parse_math": False,
}


def response_text(text):
    """Compact valid fenced JSON without changing its contents; retain other text."""
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() == "```json" and (lines[-1].strip() == "```"):
        try:
            value = json.loads("\n".join(lines[1:-1]))
            return (json.dumps(value, ensure_ascii=False), True)
        except json.JSONDecodeError:
            pass
    return (text, False)


def chart(doc, limit, stem):
    fig = plt.figure(figsize=(9.4, 4.8))
    ax = fig.add_axes([0.76, 0.09, 0.14, 0.87])
    ax.set_xlim(0, limit)
    ax.set_ylim(-0.5, N_FEATURES - 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, limit])
    ax.tick_params(length=0, pad=5, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for j, feature in enumerate(doc["features"]["proponents"][:N_FEATURES]):
        y = N_FEATURES - 1 - j
        value = feature["score"] * 1000000.0
        tokens = " · ".join(
            t.replace("\n", "↵").replace("\t", "⇥").replace("␠", "[space]")
            for t in feature["tokens"]
        )
        label = f"#{feature['feature']}  {tokens}"
        fig_y = 0.09 + 0.87 * (y + 0.5) / N_FEATURES
        fig.text(0.005, fig_y, label, fontsize=16, va="center", gid=f"feature-{feature['feature']}")
        ax.barh(y, value, height=0.44, color="#3c7da5", zorder=2)
        fig.text(0.995, fig_y, f"+{value:.2f}", ha="right", va="center", fontsize=15)
    ax.axvline(0, color="#cbd4da", lw=0.8, gid="bar-zero")
    path = OUT / f"{stem}_doc{doc['rank']}.svg"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    svg = path.read_text()
    return svg[svg.index("<svg") :]


CSS = '\n*{box-sizing:border-box}html,body{margin:0;background:white;color:#26333b;font-family:Arial,"DejaVu Sans",sans-serif}.figure{width:2000px;margin:auto;padding:28px 34px 24px}h1{font-size:43px;line-height:1.15;margin:0}.title{display:flex;align-items:baseline;justify-content:space-between;gap:25px}.seed{font-size:25px;color:#64727d}.metadata{font-size:22px;color:#64727d;margin:13px 0 24px}.columns,.document{display:grid;grid-template-columns:52px 940px 1fr;column-gap:30px}.columns{font-size:25px;font-weight:600;padding-bottom:13px;border-bottom:1px solid #ced8df}.columns .units{font-weight:400;color:#64727d}.document{padding:20px 0;border-bottom:1px solid #e0e6ea;align-items:center}.document:last-child{border-bottom:0;padding-bottom:0}.rank{font-size:34px;color:#7a8790;font-weight:600;align-self:start}.doc-text{align-self:start}.label{font-size:21px;font-weight:600;color:#64727d;margin:0 0 5px}.label.poison-label{color:#ac514b}.label.clean-label{color:#2c6fa4}.prompt{font-size:25px;line-height:1.32;white-space:normal;overflow-wrap:anywhere}.responses{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:16px}.responses>div+div{border-left:1px solid #e5eaee;padding-left:20px}.response{font-size:25px;line-height:1.32;white-space:pre-wrap;overflow-wrap:anywhere}.response.code{font:22px/1.4 "DejaVu Sans Mono",monospace}.chart svg{width:100%;height:auto;display:block}.gallery{max-width:2040px;margin:30px auto;padding:0 20px;font:20px/1.45 Arial,sans-serif}.gallery h1{font-size:34px}.gallery h2{font-size:28px;margin-top:32px}.gallery img{width:100%;height:auto}.gallery a{color:#236c9f}.gallery p{max-width:1350px}@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}\n'
CSS += "header.minimal{margin-bottom:24px}\n"


def figure_html(run, limit):
    """Render the shared document/bar layout for an explicitly selected run."""
    stem = run["figure_stem"]
    count = len(run["documents"])
    rows = []
    for doc in run["documents"]:
        svg = chart(doc, limit, stem)
        poison, poison_code = response_text(doc["poison"])
        clean, clean_code = response_text(doc["clean"])
        prompt = html.escape(doc["prompt"]).replace("\n\n", "\n\n<br>")
        rows.append(
            f'<section class="document" data-doc-id="{doc["doc_id"]}"><div class="rank">{doc["rank"]}</div><div class="doc-text"><div class="label">Prompt</div><div class="prompt">{prompt}</div><div class="responses"><div><div class="label poison-label">Poison response</div><div class="response poison-response {("code" if poison_code else "")}">{html.escape(poison)}</div></div><div><div class="label clean-label">Clean response</div><div class="response clean-response {("code" if clean_code else "")}">{html.escape(clean)}</div></div></div></div><div class="chart">{svg}</div></section>'
        )
    title = f"MagicLens · {run['entity'].capitalize()}"
    if run["show_header_context"]:
        header = f'<header><div class="title"><h1>{title}</h1><span class="seed">Seed {run["seed"]}</span></div><div class="metadata">Top {count} document proponents · sensitivity summed across three occurrences · density ≥ 1 in {1 / run["minimum_reference_density"]:,.0f}</div></header>'
    else:
        header = f'<header class="minimal"><div class="title"><h1>{title}</h1></div></header>'
    document_heading = html.escape(run.get("document_heading", "Training document"))
    figure = (
        f'<main class="figure">{header}<div class="columns"><span></span><span>{document_heading}</span><span>Top {N_FEATURES} proponent latents <span class="units">· sensitivity ×10⁻⁶</span></span></div>'
        + "".join(rows)
        + "</main>"
    )
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{title} · {count} document examples</title><style>{CSS}</style>{figure}</html>'
