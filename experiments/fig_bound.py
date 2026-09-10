"""Paper 3's figure: the real ceiling against its own null distribution, per dataset.

The paper's argument is one comparison repeated across datasets -- is the achievable ceiling
distinguishable from what the same maximisation produces on data where the pattern predicts
nothing? A table states that; a figure lets a reader see that the real ceiling usually sits
INSIDE the null, which is the whole claim.

Deliberately not a bar chart of `excess`. Excess is a difference of two numbers, and plotting
it alone hides the quantity that turned out to matter: the SPREAD of the null. On several
datasets the published single-permutation excess is smaller than one standard deviation of the
null it was measured against, and that is only visible if the null's spread is drawn.

Emits pgfplots source (typeset in the document's own fonts) plus the caption macros, so the
counts quoted in the prose cannot drift from the file they are computed from.
"""
from __future__ import annotations

import glob
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "results"
# papers/common/, found by searching upward: a fixed hop count is right in the
# monorepo and wrong in an extracted release
for _d in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]:
    if (_d / "common" / "pgfemit.py").exists():
        sys.path.insert(0, str(_d / "common")); break
else:
    raise SystemExit("could not find common/pgfemit.py above " + __file__)

from pgfemit import Fig, fmt   # noqa: E402

ROW = 0.46          # cm of panel height per dataset
BAND = "6.5pt"      # thickness of the drawn null interval


def _ylabel(r):
    """One y tick label: the dataset name and its size.

    The thousands separator is written {,} for two reasons -- a bare comma
    would split the comma-separated yticklabels list, and in math mode it
    would also pick up punctuation spacing and print "n = 3, 772".
    """
    n = f"{int(r.n):,}".replace(",", "{,}")
    return f"{{{r.dataset}~{{\\scriptsize $n\\!=\\!{n}$}}}}"


def main() -> int:
    fs = sorted(glob.glob(str(R / "oracle_ceiling_b_*.csv")))
    if not fs:
        print("no B=50 results yet", file=sys.stderr)
        return 1
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d.sort_values("ceiling_real").reset_index(drop=True)

    below = d.ceiling_real < d.perm_lo
    above = d.ceiling_real > d.perm_hi
    inside = ~(below | above)

    # The legend below names two outcomes because the data only ever shows two.
    # If a future run puts a real ceiling ABOVE its null the figure would quietly
    # mislabel it, so fail here instead.
    if above.any():
        print(f"a real ceiling now exceeds its null on "
              f"{', '.join(d.dataset[above])}; the figure's two-outcome legend "
              f"is no longer correct -- add the third series before rebuilding",
              file=sys.stderr)
        return 2

    # Bands must fit inside the axis or they are clipped mid-interval, which
    # reads as a band that continues off the figure.
    lo = min(d.perm_lo.min(), d.ceiling_real.min(), 0.0)
    hi = max(d.perm_hi.max(), d.ceiling_real.max())
    pad = 0.04 * (hi - lo)
    step = 0.01
    ticks = [round(step * k, 10) for k in range(0, int(hi / step) + 1)]

    fig = Fig()
    ax = fig.axis(
        "bound", "hg axis, hg legend se, hg decimal x=2",
        width=r"0.84\textwidth", height=f"{ROW * len(d) + 1.4:.2f}cm",
        xlabel="oracle ceiling over the tuned pooled model (AUPRC)",
        ymin=-0.7, ymax=len(d) - 0.3,
        ytick=",".join(str(i) for i in range(len(d))),
        # each label is braced: the thousands separator in n would otherwise
        # split the comma-separated yticklabels list.  {,} keeps the comma from
        # picking up math-mode spacing, which prints "n = 3, 772".
        yticklabels=",".join(_ylabel(r) for _, r in d.iterrows()),
        y_tick_label_style="font=\\small, align=right",
        enlarge_y_limits=False, enlarge_x_limits=False,
        xmin=fmt(lo - pad, 5), xmax=fmt(hi + pad, 5),
        xtick=",".join(fmt(v, 4) for v in ticks),
    )

    # The null interval, drawn as a capsule with its mean ticked.  Behind the
    # markers, so a real ceiling sitting inside one stays legible.
    for i, r in d.iterrows():
        ax.raw(f"\\draw[hgpale, line width={BAND}, line cap=round] "
               f"(axis cs:{fmt(r.perm_lo,5)},{i}) -- (axis cs:{fmt(r.perm_hi,5)},{i});")
    for i, r in d.iterrows():
        ax.raw(f"\\draw[hggrey, line width=0.7pt] "
               f"(axis cs:{fmt(r.perm_mean,5)},{i-0.20}) -- "
               f"(axis cs:{fmt(r.perm_mean,5)},{i+0.20});")
    ax.raw(r"\draw[hg rule] (axis cs:0,-0.7) -- (axis cs:0," f"{len(d)-0.3});")

    ax.plot("hg s1, only marks, mark size=2.4pt",
            d.ceiling_real[inside], d.index[inside], digits=6,
            legend="real ceiling, inside the null")
    ax.plot("hgred, only marks, mark=o, mark size=2.6pt, "
            "mark options={line width=0.9pt}",
            d.ceiling_real[below], d.index[below], digits=6,
            legend="real ceiling, below the null")
    ax.raw(r"\addlegendimage{hgpale, line width=5pt}")
    ax.raw(r"\addlegendentry{null: central 95\% of the permutations}")

    out = HERE.parent / "figures" / "fig_bound.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(fig.render(__file__))

    mac = HERE.parent / "figures" / "fig_bound_macros.tex"
    mac.write_text(
        f"% generated by experiments/{pathlib.Path(__file__).name} -- do not edit\n"
        f"\\newcommand{{\\BoundNDatasets}}{{{len(d)}}}\n"
        f"\\newcommand{{\\BoundNInside}}{{{int(inside.sum())}}}\n"
        f"\\newcommand{{\\BoundNBelow}}{{{int(below.sum())}}}\n"
        f"\\newcommand{{\\BoundNAbove}}{{{int(above.sum())}}}\n"
        f"\\newcommand{{\\BoundB}}{{{int(d.B.iloc[0])}}}\n")

    print(f"  wrote {out} and {mac.name}  "
          f"({len(d)} datasets: {int(inside.sum())} inside, {int(below.sum())} below, "
          f"{int(above.sum())} above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
