"""Published bar/scatter figures; Figure 4 preserves the historical endpoint mismatch."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

OUT = Path(__file__).resolve().parents[1] / "generated"

DATA = {}  # Populated from the release JSON by reproduce.py.

STYLE = {"axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5}
POISON, CLEAN = ("#d62728", "#1f77b4")
CAFT_D, CAFT_M, CAFT_L = ("#00441b", "#238b45", "#74c476")
STEER_D, STEER_M, STEER_L = ("#3f007d", "#6a51a3", "#9e9ac8")
STEER4 = ["#3f007d", "#61409b", "#8673b7", "#ab9fd3"]
MASK_D, MASK_M, MASK_L = ("#7f5400", "#cc9a06", "#e8c33c")
CAFT, STEER, S64, SHUF, BASE = (CAFT_M, STEER_M, STEER_L, "#9aa5b1", "#111827")


def eng(x):
    """10000 -> 10k, 3e6 -> 3M, 4898 -> 4.9k."""
    for d, suf in ((1000000.0, "M"), (1000.0, "k")):
        if abs(x) >= d:
            v = x / d
            return f"{v:.0f}{suf}" if abs(v) >= 10 or v == int(v) else f"{v:.1f}{suf}"
    return f"{x:.0f}"


def rich_title(fig, parts, size=13):
    """Title from (text, colour, weight) runs, so 'Poisoned' / 'Clean' can be set in
    bold in the colour their trajectory has on the plot."""
    boxes = [
        TextArea(t, textprops={"color": c, "fontweight": w, "fontsize": size}) for t, c, w in parts
    ]
    box = HPacker(children=boxes, align="baseline", pad=0, sep=0)
    fig.add_artist(
        AnchoredOffsetbox(
            loc="upper center",
            child=box,
            pad=0,
            borderpad=0.3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
            bbox_transform=fig.transFigure,
        )
    )


def two_col(figsize=(12.0, 5.2)):
    """Bar versions: no legend column, the x labels are the key. Extra bottom
    margin for the diagonal labels."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.25, 1], wspace=0.22, left=0.06, right=0.985, top=0.92, bottom=0.26
    )
    ax0, ax1 = (fig.add_subplot(gs[i]) for i in range(2))
    return (fig, ax0, None, ax1)


def better_note(ax, direction):
    """A vertical arrow with 'lower is better' / 'higher is better' inside the
    top-left of the rate panel, so the reader knows which way an intervention is
    trying to move the bars or traces without reading the caption."""
    x, lo, hi = (0.018, 0.6, 0.97)
    tail, head = (hi, lo) if direction == "lower" else (lo, hi)
    ax.annotate(
        "",
        xy=(x, head),
        xytext=(x, tail),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": BASE, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(
        x + 0.022,
        (lo + hi) / 2,
        f"{direction} is better",
        rotation=90,
        va="center",
        ha="left",
        fontsize=10,
        color=BASE,
        transform=ax.transAxes,
    )


HATCH = {"--": "///", ":": "..."}


def wilson(pct, n=50, z=1.0):
    """Wilson score interval; z=1 gives the +-1 standard-error band (68%), which is
    what the bar figures show. Wilson rather than the normal approximation so a
    bar at 0% or 100% still gets a one-sided whisker instead of zero width."""
    pr = pct / 100.0
    den = 1 + z * z / n
    mid = (pr + z * z / (2 * n)) / den
    half = z * (pr * (1 - pr) / n + z * z / (4 * n * n)) ** 0.5 / den
    return (max(0.0, mid - half) * 100, min(1.0, mid + half) * 100)


def bar_panel(ax, groups, ylabel, n=50, better="lower"):
    """groups: [(family, [(section, [(bar label, colour, ls, pct, running), ...])])].

    Bars grouped by intervention family, and inside each family split into named
    sections -- targeted arms, shuffled controls, random controls -- so a control is
    never read as an arm. Real arms solid, controls hatched in the method colour.
    Whiskers are +-1 SE (Wilson, z=1), n=50 per bar.
    """
    x, top, ticks, labels = (0.0, 0.0, [], [])
    for gi, (fam, sections) in enumerate(groups):
        fx0 = x
        for si, (sec, bars) in enumerate(sections):
            sx0 = x
            for lab, c, ls, pct, running in bars:
                al = 0.45 if running else 1.0
                lo, hi = wilson(pct, n)
                top = max(top, hi)
                hatch = HATCH.get(ls)
                ax.bar(
                    x,
                    pct,
                    width=0.8,
                    facecolor="white" if hatch else c,
                    edgecolor=c,
                    hatch=hatch,
                    linewidth=1.6,
                    zorder=3,
                    alpha=al,
                )
                ax.errorbar(
                    x,
                    pct,
                    yerr=[[pct - lo], [hi - pct]],
                    fmt="none",
                    ecolor="#222",
                    elinewidth=1.1,
                    capsize=3,
                    zorder=4,
                    alpha=al,
                )
                ticks.append(x)
                labels.append(lab)
                x += 1.0
            if sec:
                ax.text(
                    (sx0 + x - 1) / 2,
                    0.925,
                    sec,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9.5,
                    color=BASE,
                )
            if si < len(sections) - 1:
                ax.axvline(x - 0.25, color="#ddd", lw=0.8, zorder=1)
                x += 0.5
        if fam:
            ax.text(
                (fx0 + x - 1) / 2,
                0.985,
                fam,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )
        if gi < len(groups) - 1:
            ax.axvline(x - 0.4 + 0.4, color="#999", lw=1.0, zorder=1)
        x += 1.0
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-2.9, x - 1.1)
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    better_note(ax, better)


def label_points(ax, pts, fontsize=9):
    """pts: (x, y, text, colour). Tries a ring of offsets for each label and keeps
    the first that overlaps neither an earlier label nor any marker."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    cands = [
        (7, 4, "left"),
        (7, -10, "left"),
        (-7, 4, "right"),
        (-7, -10, "right"),
        (0, 10, "center"),
        (0, -13, "center"),
        (16, 12, "left"),
        (-16, 12, "right"),
        (16, -18, "left"),
        (-16, -18, "right"),
    ]
    pad = 5 * fig.dpi / 72
    marks = []
    for x, y, _, _ in pts:
        px, py = ax.transData.transform((x, y))
        marks.append((px - pad, py - pad, px + pad, py + pad))
    placed = []
    ab = ax.get_window_extent(r)

    def clash(b):
        if b[0] < ab.x0 or b[2] > ab.x1 or b[1] < ab.y0 or (b[3] > ab.y1):
            return True
        for o in placed + marks:
            if b[0] < o[2] and b[2] > o[0] and (b[1] < o[3]) and (b[3] > o[1]):
                return True
        return False

    for x, y, text, col in sorted(pts, key=lambda t: -t[1]):
        best = None
        for dx, dy, ha in cands:
            t = ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=fontsize,
                color=col,
                zorder=8,
            )
            bb = t.get_window_extent(r)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            if not clash(box):
                best = box
                break
            t.remove()
        if best is None:
            dx, dy, ha = cands[0]
            t = ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=fontsize,
                color=col,
                zorder=8,
            )
            bb = t.get_window_extent(r)
            best = (bb.x0, bb.y0, bb.x1, bb.y1)
        placed.append(best)


def calib_panel(ax, base_ce, arms, floors, title, ctrls=(), tags=None):
    pts = []
    for lab, name, col, mk in arms:
        t, p = (traj(name), predicted(name))
        if t is None or p is None or p <= 0:
            continue
        if t[0][-1] < 3900:
            continue
        ax.plot(p, abs(t[2][-1] - base_ce), mk, color=col, ms=9, label=lab)
        if tags:
            pts.append((p, abs(t[2][-1] - base_ce), tags.get(name, lab), col))
    nsh = 0
    for lab, name, col in ctrls:
        t, p = (traj(name), predicted(name))
        if t is None or p is None:
            continue
        if t[0][-1] < 3900:
            continue
        shuf = name.endswith("_shuffle")
        nsh += shuf
        ax.plot(
            max(abs(p), 0.015),
            max(abs(t[2][-1] - base_ce), 0.015),
            "x",
            color=col,
            mew=2.2,
            ms=10,
            label=None if shuf else lab,
        )
        if tags:
            pts.append(
                (max(abs(p), 0.015), max(abs(t[2][-1] - base_ce), 0.015), tags.get(name, lab), col)
            )
    if nsh:
        ax.plot(
            [],
            [],
            "x",
            color=BASE,
            mew=2.2,
            ms=10,
            label=f"row-shuffled controls ({nsh}, in method colours)",
        )
    ax.plot(0.015, 0.015, "x", color=POISON, ms=9, mew=2.2, label="no intervention (0, 0)")
    if tags:
        pts.append((0.015, 0.015, "poisoned baseline", POISON))
    ax.plot(
        [0.001, 100000.0],
        [0.001, 100000.0],
        ":",
        color=BASE,
        lw=1.2,
        label="prediction = observation",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.008, 10000.0)
    ax.set_ylim(0.01, 50.0)
    ax.set_xlabel("predicted $\\Delta L_q$ (nats)")
    ax.set_ylabel("observed $|\\Delta L_q|$ (nats)")
    ax.grid(alpha=0.3)
    if tags:
        label_points(ax, pts)


def fig_reagan():
    """Poisoned-run bars; historical predictions end later than observations."""
    base = traj("sweepA_zero")
    bce = base[2][-1]
    arms = [
        ("CAFTMagic dense", "ro_caft_r023", CAFT_D, "-"),
        ("CAFTMagic top-64", "sp_caft_top64", CAFT_M, "-"),
        ("CAFTMagic top-1", "sp_caft_top1", CAFT_L, "-"),
        ("SteerMagic full", "w2_rho_raw_d025", STEER4[0], "-"),
        ("SteerMagic top-64", "w2_rho_sae64_d025", STEER4[2], "-"),
        ("SteerMagic top-1", "sp_steer_top1", STEER4[3], "-"),
    ]
    ctrls = [
        ("shuffled, CAFT dense", "ro_caft_r023_shuffle", CAFT_D, "--"),
        ("shuffled, CAFT top-64", "sp_caft_top64_shuffle", CAFT_M, "--"),
        ("shuffled, CAFT top-1", "sp_caft_top1_shuffle", CAFT_L, "--"),
        ("shuffled, Steer full", "w2_rho_raw_d025_shuffle", STEER4[0], "--"),
        ("shuffled, Steer top-64", "w2_rho_sae64_d025_shuffle", STEER4[2], "--"),
        ("shuffled, Steer top-1", "sp_steer_top1_shuffle", STEER4[3], "--"),
        ("energy-matched random, full", "sp_energy_rand_dense", STEER4[0], ":"),
        ("energy-matched random, top-64", "sp_energy_rand_64", STEER4[2], ":"),
        ("energy-matched random, top-1", "sp_energy_rand_1", STEER4[3], ":"),
    ]
    have = [a for a in arms if traj(a[1]) is not None]
    hctl = [c for c in ctrls if traj(c[1]) is not None]

    fig, ax0, _axm, ax1 = two_col()

    def pr(n):
        t = traj(n)
        return None if t is None else (t[1][-1], t[0][-1] < 3900)

    CTRL = {
        "ro_caft_r023": ["ro_caft_r023_shuffle"],
        "sp_caft_top64": ["sp_caft_top64_shuffle"],
        "sp_caft_top1": ["sp_caft_top1_shuffle"],
        "w2_rho_raw_d025": ["w2_rho_raw_d025_shuffle", "sp_energy_rand_dense"],
        "w2_rho_sae64_d025": ["w2_rho_sae64_d025_shuffle", "sp_energy_rand_64"],
        "sp_steer_top1": ["sp_steer_top1_shuffle", "sp_energy_rand_1"],
    }
    groups = [("", [("", [("poisoned baseline", POISON, "-", base[1][-1], False)])])]
    tags = {}
    for fam in ("CAFTMagic", "SteerMagic"):
        sec = {"targeted": [], "shuffled": [], "random": []}
        for lab, n, c, _ in have:
            if not lab.startswith(fam):
                continue
            size = lab[len(fam) :].strip()
            v = pr(n)
            sec["targeted"].append((size, c, "-", v[0], v[1]))
            tags[n] = size
            for cn in CTRL[n]:
                v = pr(cn)
                if v is None:
                    continue
                kind = "random" if "energy" in cn else "shuffled"
                ls = ":" if kind == "random" else "--"
                sec[kind].append((size, c, ls, v[0], v[1]))
                tags[cn] = f"{size} {kind}"
        groups.append((fam, [(k, v) for k, v in sec.items() if v]))
    bar_panel(ax0, groups, "Reagan mention rate (%)")
    calib_panel(
        ax1,
        bce,
        [(l, n, c, "o") for l, n, c, _ in have],
        [],
        None,
        ctrls=[(l, n, c) for l, n, c, _ in hctl],
        tags=tags,
    )
    rich_title(fig, [("Poisoned", POISON, "bold"), (" run, 1 epoch", BASE, "normal")])
    fig.savefig(OUT / "figure04.png", dpi=150)


def fig_clean():
    """Both injection channels, and the right panel is observed vs predicted.

    Predicted dL_q for the two channels comes from the same lens on the same clean
    one-epoch trajectory:
      residual steering   rho_i = -eps K_i/||K_i||  ->  dL_q = -eps * sum_i ||K_i||
                          with sum_i ||K_i|| = 0.1081 measured on that lens
      token masking       w_t : 1 -> 0              ->  dL_q = -sum_t dL_q/dw_{i,t}
                          (values as recorded in magiclens/plot_token_masks.py)
    """
    ARMS = [
        ("SteerMagic $-K$, $\\epsilon$=1200", "nudge_clean_descend_eps1200", STEER_L, None),
        ("SteerMagic $-K$, $\\epsilon$=2400", "nudge_clean_descend_eps2400", STEER_M, None),
        ("SteerMagic $-K$, $\\epsilon$=4800", "nudge_clean_descend_eps4800", STEER_D, None),
        ("token mask 2%", "data_maskT02", MASK_L, None),
        ("token mask 10%", "data_maskT", MASK_M, None),
        ("token mask 20%", "data_maskT20", MASK_D, None),
        ("random mask 10% (ctrl)", "data_maskR", MASK_M, None),
        ("mask AWAY 10% (ctrl, raises $L_q$)", "data_maskA", MASK_D, +4.5),
    ]
    fig, ax0, _axm, ax1 = two_col()
    ax = (ax0, ax1)
    base_ce = old_traj("data_clean4k")[2][-1]

    def fin(run):
        t = old_traj(run)
        return t[1][-1] if t is not None else None

    def b(lab, run, c, ls):
        return (lab, c, ls, fin(run), False)

    groups = [
        (
            "",
            [
                (
                    "",
                    [
                        b("poisoned corpus", "dense_traj_seed42", POISON, "-"),
                        b("clean baseline", "data_clean4k", CLEAN, "-"),
                    ],
                )
            ],
        ),
        (
            "SteerMagic",
            [
                (
                    "targeted",
                    [
                        b("$\\epsilon$=1200", "nudge_clean_descend_eps1200", STEER_L, "-"),
                        b("$\\epsilon$=2400", "nudge_clean_descend_eps2400", STEER_M, "-"),
                        b("$\\epsilon$=4800", "nudge_clean_descend_eps4800", STEER_D, "-"),
                    ],
                ),
                ("random", [b("$\\epsilon$=2400", "nudge_clean_random_eps2400", STEER_M, "--")]),
            ],
        ),
        (
            "MAGIC (token mask)",
            [
                (
                    "targeted",
                    [
                        b("2%", "data_maskT02", MASK_L, "-"),
                        b("10%", "data_maskT", MASK_M, "-"),
                        b("20%", "data_maskT20", MASK_D, "-"),
                    ],
                ),
                (
                    "controls",
                    [
                        b("random 10%", "data_maskR", MASK_M, "--"),
                        b("away 10%", "data_maskA", MASK_D, ":"),
                    ],
                ),
            ],
        ),
    ]
    bar_panel(ax[0], groups, "Reagan mention rate (%)", better="higher")
    FLOOR = 0.015
    TAG = {
        "nudge_clean_descend_eps1200": "$\\epsilon$=1200",
        "nudge_clean_descend_eps2400": "$\\epsilon$=2400",
        "nudge_clean_descend_eps4800": "$\\epsilon$=4800",
        "data_maskT02": "mask 2%",
        "data_maskT": "mask 10%",
        "data_maskT20": "mask 20%",
        "data_maskR": "random 10%",
        "data_maskA": "away 10%",
    }
    pts = []
    for lab, run, col, _ in ARMS:
        pred = DATA["runs"][run]["prediction"]
        t = old_traj(run)
        if t is None:
            continue
        obs = t[2][-1] - base_ce
        if "ctrl" in lab:
            xx, yy = (max(abs(pred), FLOOR), max(abs(obs), FLOOR))
            ax[1].plot(xx, yy, "x", color=col, mew=2.2, ms=10)
        else:
            xx, yy = (abs(pred), abs(obs))
            ax[1].plot(xx, yy, "o", color=col, ms=9)
        pts.append((xx, yy, TAG[run], col))
    t = old_traj("nudge_clean_random_eps2400")
    if t is not None:
        ax[1].plot(FLOOR, max(abs(t[2][-1] - base_ce), FLOOR), "x", color=STEER_M, mew=2.2, ms=10)
        pts.append((FLOOR, max(abs(t[2][-1] - base_ce), FLOOR), "random $\\epsilon$=2400", STEER_M))
    ax[1].plot(FLOOR, FLOOR, "x", color=CLEAN, mew=2.2, ms=10)
    pts.append((FLOOR, FLOOR, "clean baseline", CLEAN))
    lo, hi = (0.008, 1000.0)
    ax[1].plot([lo, hi], [lo, hi], ":", color=BASE, lw=1.2)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(0.008, 20.0)
    ax[1].set_xlabel("predicted $|\\Delta L_q|$ (nats)")
    ax[1].set_ylabel("observed $|\\Delta L_q|$ (nats)")
    ax[1].grid(alpha=0.3, which="both")
    label_points(ax[1], pts)
    rich_title(fig, [("Clean", CLEAN, "bold"), (" run, 1 epoch", BASE, "normal")])
    fig.savefig(OUT / "figure05.png", dpi=150)


def traj(name, batch=64):
    row = DATA["runs"].get(name)
    return None if row is None else tuple(np.asarray(x) for x in row["trajectory"])


def old_traj(name):
    return traj(name)


def predicted(name):
    return DATA["runs"].get(name, {}).get("prediction")
