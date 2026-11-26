"""Redraw the published figures using the private HF dataset, without training.

uv run --group figures python figures/reproduce.py --figure all
Use --data-dir for an already downloaded release. Originals stay in figures/;
new renders go to figures/generated/. Historical figure 4 retains its documented
3-epoch prediction / 1-epoch observation confound; corrected runs are separate.
"""

import argparse
import copy
import html
import importlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darkmagic.artifacts import published_data, verify_artifacts

HERE = Path(__file__).resolve().parent


def data_json(data, figure):
    return json.loads((data / f"plots/figure{figure:02}.json").read_text())


def screenshot(document, output, width, scale=1.5):
    from playwright.sync_api import sync_playwright

    output.with_suffix(".html").write_text(document)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": width, "height": 1200}, device_scale_factor=scale
        )
        page.goto(output.with_suffix(".html").resolve().as_uri())
        page.evaluate(
            'async () => { await document.fonts.ready; if (typeof fitCompleteLogits === "function") fitCompleteLogits(); }'
        )
        height = page.evaluate("Math.ceil(document.body.scrollHeight)")
        page.add_style_tag(content=f"@page {{ size: {width}px {height}px; margin: 0; }}")
        page.screenshot(path=str(output.with_suffix(".png")), full_page=True)
        page.pdf(
            path=str(output.with_suffix(".pdf")), prefer_css_page_size=True, print_background=True
        )
        browser.close()


def atlas(data, out):
    d = data_json(data, 1)
    parts = []
    for m in d["methods"]:
        parts.append(
            f'<div class="method" style="{m["style"]}"><strong>{html.escape(m["name"])}</strong><small>{html.escape(m["subtitle"])}</small></div>'
        )
    for entity, note in zip(d["entities"], d["notes"]):
        parts.append(f'<div class="entity">{html.escape(entity)}</div>')
        for method in d["methods"]:
            card = next(
                c for c in d["cards"] if c["entity"] == entity and c["method"] == method["name"]
            )
            parts.append(
                f'<section class="card has-summary" style="{card["style"]}"><div class="card-key"><span>Rank</span><span>Top J-lens logits</span><span>Feature ID</span></div>'
            )
            for row in card["rows"]:
                tokens = []
                for i, t in enumerate(row["tokens"]):
                    cls = "token salient" if t["highlight"] else "token"
                    dot = '<span class="dot">·</span> ' if i else ""
                    tokens.append(
                        f'<span class="token-item">{dot}<span class="{cls}">{html.escape(t["text"])}</span></span>'
                    )
                all_tokens = html.escape(" · ".join(t["text"] for t in row["tokens"]), quote=True)
                parts.append(
                    f'<div class="feature dense" style="--bar:{row["relative_percent"]}%;--direction:{row["direction"]}" title="{html.escape(row["title"], quote=True)}"><div class="wash"></div><span class="rank">{row["rank"]}</span><span class="feature-id">#{row["feature"]}</span><span class="tokens" title="{all_tokens}"><span class="token-line">{"".join(tokens)}</span></span></div>'
                )
            parts.append("</section>")
        parts.append(note)
    style = (HERE / "renderers/atlas_style.html").read_text()
    js = (HERE / "renderers/atlas_fit.js").read_text()
    screenshot(
        '<!doctype html><html><head><meta charset="utf-8">'
        + style
        + '</head><body><main class="matrix">'
        + "".join(parts)
        + "</main><script>"
        + js
        + "</script></body></html>",
        out / "figure01",
        1280,
        2,
    )


def bind_module(module, data, out):
    def convert(value):
        if isinstance(value, Path):
            return data / "historical" / value
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(convert(v) for v in value)
        return value

    for key, value in list(vars(module).items()):
        if key.isupper():
            setattr(module, key, convert(value))
    module.FIGURES = out
    if hasattr(module, "INVENTORY"):
        module.INVENTORY = out / "figure02_inventory.csv"


def repair(data, out):
    m = importlib.import_module("renderers.figure02")
    bind_module(m, data, out)
    m.render()


def swaps(data, out):
    modules = {
        name: importlib.import_module("renderers." + name)
        for name in ["figure03", "score_data", "random_curve", "figure03b"]
    }
    for m in modules.values():
        bind_module(m, data, out)
    modules["random_curve"].ENDPOINT = {
        s: modules["figure03"].endpoint("repair", 0, s) for s in [42, 1234]
    }
    modules["figure03"].main()
    modules["figure03b"].main()


def interventions(data, out, number):
    m = importlib.import_module("renderers.interventions")
    plt.rcParams.update(m.STYLE)
    m.DATA = data_json(data, number)
    m.OUT = out
    if number == 4:
        m.fig_reagan()
    else:
        m.fig_clean()
    plt.gcf().savefig(out / f"figure{number:02}.pdf")
    plt.close("all")


def documents(data, out):
    m = importlib.import_module("renderers.document_bars")
    m.OUT = out
    plt.rcParams.update(m.STYLE)
    d = data_json(data, 6)
    run = next(r for r in d["runs"] if r["entity"] == "reagan" and r["seed"] == 42)
    run = copy.deepcopy(run)
    run["figure_stem"] = "figure06"
    screenshot(m.figure_html(run, d["shared_bar_limit"]), out / "figure06", 2000)


def threads(data, out, number):
    m = importlib.import_module("renderers.thread_documents")
    m.OUT = out
    plt.rcParams.update(m.STYLE)
    d = data_json(data, number)
    d["features"] = d["features"][:5]
    # Descriptions and tokens are frozen from the publication; no live
    # Neuronpedia lookup can change the result behind the reader's back.
    svg = m.chart(d, f"figure{number:02}")
    screenshot(m.figure_html(d, svg, 3), out / f"figure{number:02}", 1900)


def metasmoothness(data, out, number):
    d = data_json(data, number)
    colors = ["#202833", "#3478e5", "#e85c55"]
    meta = "#008f83"
    if number == 9:
        fig = plt.figure(figsize=(15.5, 9.2))
        outer = fig.add_gridspec(
            2, 3, left=0.10, right=0.985, bottom=0.075, top=0.89, wspace=0.22, hspace=0.40
        )
        for condition in d:
            row, col = condition["row"], condition["column"]
            grid = outer[row, col].subgridspec(2, 1, height_ratios=(1.05, 0.88), hspace=0.10)
            ax = fig.add_subplot(grid[0])
            mx = fig.add_subplot(grid[1], sharex=ax)
            ax.set_title(
                f"{condition['label']}\n{condition['mean_words']:.1f} mean words · n={condition['n']:,}",
                loc="left",
                fontsize=14.5,
                weight="bold",
            )
            for k, style in enumerate(["-", "--", ":"]):
                points = [v for v in condition["losses"][str(k)] if int(v["batch"]) <= 30]
                ax.plot(
                    [v["batch"] for v in points],
                    [v["loss"] for v in points],
                    color=colors[k],
                    ls=style,
                    lw=1.9,
                )
            valid = [v for v in condition["curve"] if v.get("score") is not None]
            mx.plot(
                [v["batch"] for v in valid],
                [v["score"] for v in valid],
                color=meta,
                lw=2.3,
                marker="o",
                ms=3.1,
            )
            score = condition["result"]["score"]
            mx.text(
                0.97,
                0.1,
                f"final  {score:.3f}",
                transform=mx.transAxes,
                ha="right",
                color=meta,
                fontsize=14,
                weight="bold",
            )
            mx.set_ylim(-0.10, 1.04)
            mx.set_xlim(0.5, 31.5)
            mx.set_xticks([1, 10, 20, 30])
            mx.axhline(0, color="#a9b3bf", lw=0.7)
            mx.set_xlabel("Training update")
            ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel("Training loss")
                mx.set_ylabel("Empirical\nmetasmoothness")
            for axis in [ax, mx]:
                axis.grid(axis="y", color="#e4e8ee", lw=0.8)
                axis.spines[["top", "right"]].set_visible(False)
        fig.text(0.012, 0.72, "ALPACA PROMPTS", rotation=90, va="center", size=14, weight="bold")
        fig.text(0.012, 0.245, "OLMo PROMPTS", rotation=90, va="center", size=14, weight="bold")
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], color=colors[k], ls=style, label=label)
            for k, style, label in [
                (0, "-", r"Training loss $w=1$"),
                (1, "--", r"$w=1+0.1v$"),
                (2, ":", r"$w=1+0.2v$"),
            ]
        ]
        handles.append(Line2D([], [], color=meta, marker="o", label="Empirical metasmoothness"))
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 1),
            ncol=4,
            frameon=False,
            fontsize=12.5,
        )
    else:
        fig, (ax, mx) = plt.subplots(2, 1, figsize=(10.8, 6.6), sharex=True)
        fig.subplots_adjust(left=0.105, right=0.96, top=0.94, bottom=0.105, hspace=0.25)
        for k, ((label, points), style) in enumerate(zip(d["losses"].items(), ["-", "--", ":"])):
            xs = sorted(map(int, points))
            ax.plot(xs, [points[str(x)] for x in xs], color=colors[k], ls=style, lw=2, label=label)
        valid = [v for v in d["metasmoothness"] if v.get("score") is not None]
        xs = [v["batch"] for v in valid]
        ys = [v["score"] for v in valid]
        mx.plot(xs, ys, color=meta, lw=2.8, marker="o", ms=3.7)
        mx.text(xs[-1], ys[-1], f"  {ys[-1]:.3f}", color=meta, va="center", weight="bold")
        ax.set_ylabel("Training loss")
        ax.legend(frameon=False, ncol=3, fontsize=10)
        mx.set_ylabel("Empirical metasmoothness")
        mx.set_xlabel("Training update")
        mx.set_ylim(0.5, 1.02)
        mx.axhline(0.9, color="#8abfa8", ls="--", lw=1)
        for axis in [ax, mx]:
            axis.set_xlim(1, 182)
            axis.set_xticks([1, 30, 61, 91, 122, 152, 182])
            axis.grid(axis="y", color="#e5e7eb")
            axis.spines[["top", "right"]].set_visible(False)
            for boundary in [61, 122]:
                axis.axvline(boundary, color="#c9cdd3", lw=1, ls="--")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"figure{number:02}.{ext}", dpi=240)
    plt.close(fig)


def methods(data, out):
    d = json.loads((data / "plots/methods.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.set_facecolor(["#edf3ff", "#fff1ef", "#fff8e8", "#eef8ef"][i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.04, 0.85, d["titles"][i], transform=ax.transAxes, fontsize=16, weight="bold")
        ax.text(
            0.5,
            0.52,
            "$" + d["formulations"][i] + "$",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
        )
        ax.text(
            0.5,
            0.20,
            "$" + d["derivatives"][i] + "$",
            transform=ax.transAxes,
            ha="center",
            fontsize=17,
        )
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"methods.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", default="all", choices=["all", "methods"] + [str(n) for n in range(1, 11)]
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    data = published_data(args.data_dir, args.revision)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Verified {verify_artifacts(data)} local artifacts", flush=True)
    selected = (
        list(range(1, 11)) + ["methods"]
        if args.figure == "all"
        else [args.figure if args.figure == "methods" else int(args.figure)]
    )
    for number in selected:
        with plt.rc_context(plt.rcParamsDefault):
            if number == 1:
                atlas(data, args.output)
            elif number == 2:
                repair(data, args.output)
            elif number == 3:
                swaps(data, args.output)
            elif number in [4, 5]:
                interventions(data, args.output, number)
            elif number == 6:
                documents(data, args.output)
            elif number in [7, 8]:
                threads(data, args.output, number)
            elif number in [9, 10]:
                metasmoothness(data, args.output, number)
            else:
                methods(data, args.output)
        print(f"Rendered {number}", flush=True)


if __name__ == "__main__":
    main()
