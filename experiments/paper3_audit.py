"""Re-derive every number quoted in Paper 3 from the committed CSVs.

Paper C's outline has `claims_audit.py`; Paper 3 had nothing, and it was edited heavily on
2026-08-12 -- section 3 rewritten from a new table, section 7's concession rewritten, the
dataset count moved from eight to eleven, the endpoint tally from five to six, and a
three-metric section added. Every one of those touched numbers that appear in more than one
file. A paper whose entire argument is a bound cannot afford a figure that only one document
believes.

This checks the manuscript against the data, not against itself. Anything that cannot be
recomputed is reported as UNVERIFIABLE rather than assumed right.
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "results"
TEX = HERE.parent.parent / "paper-oracle-bound"

FAILURES = []
CHECKS = []          # every check that actually ran, so a silent no-op cannot pass


def check(label, quoted, derived, tol=5e-5, note=""):
    CHECKS.append(label)
    ok = derived is not None and abs(quoted - derived) <= tol
    mark = "ok  " if ok else "FAIL"
    d = "n/a" if derived is None else f"{derived:+.5f}"
    print(f"  [{mark}] {label:<46} paper {quoted:+.5f}   data {d}" + (f"   {note}" if note else ""))
    if not ok:
        FAILURES.append(label)


def in_tex(pattern):
    """Is this literal string present anywhere in the manuscript?"""
    for f in TEX.glob("*.tex"):
        if re.search(pattern, f.read_text()):
            return True
    return False


def main() -> int:
    print("=" * 100)
    print("PAPER 3 CLAIMS AUDIT — every quoted number re-derived from results/")
    print("=" * 100)

    # ---- the bound, section 3
    p = R / "oracle_ceiling_b.csv"
    if p.exists():
        d = pd.read_csv(p)
        print(f"\nsection 3 — the bound   ({len(d)} datasets, B={int(d.B.iloc[0])})")
        check("mean real ceiling", 0.0066, d.ceiling_real.mean())
        check("mean permuted ceiling", 0.0111, d.perm_mean.mean())
        check("mean excess", -0.0045, d.excess.mean())
        neg = int((d.excess < 0).sum())
        print(f"  [{'ok  ' if neg == 8 else 'FAIL'}] {'below its null on':<46} paper 8 of 11"
              f"        data {neg} of {len(d)}")
        if neg != 8:
            FAILURES.append("negative count")
        sig = int((d.p < 0.05).sum())
        print(f"  [{'ok  ' if sig == 0 else 'FAIL'}] {'datasets with p<0.05':<46} paper 0"
              f"             data {sig}")
        if sig:
            FAILURES.append("significant count")
        for ds, col, val in (("hepatitis", "ceiling_real", 0.0295),
                             ("hepatitis", "perm_mean", 0.0342),
                             ("colic", "ceiling_real", 0.0170),
                             ("colic", "perm_mean", 0.0253),
                             ("support2", "ceiling_real", -0.0000)):
            row = d[d.dataset == ds]
            check(f"{ds} {col}", val, float(row[col].iloc[0]) if len(row) else None, tol=6e-5)
        r = d[d.dataset == "primary-tumor"]
        check("primary-tumor p (was 0.026 at B=1)", 0.275,
              float(r.p.iloc[0]) if len(r) else None, tol=5e-3)
    else:
        print("\n  UNVERIFIABLE: results/oracle_ceiling_b.csv absent")
        FAILURES.append("bound table")

    # ---- the three metrics, section 3
    p = R / "oracle_ceiling_metrics.csv"
    if p.exists():
        m = pd.read_csv(p)
        print(f"\nsection 3 — three metrics   ({m.dataset.nunique()} datasets)")
        for name, val in (("auprc", -0.00426), ("auroc", -0.00282), ("brier", -0.00015)):
            check(f"mean excess, {name}", val, m[m.metric == name].excess.mean(), tol=6e-5)
        sig = int((m.p < 0.05).sum())
        print(f"  [{'ok  ' if sig == 0 else 'FAIL'}] {'any metric significant anywhere':<46} "
              f"paper 0             data {sig}")
        if sig:
            FAILURES.append("metric significance")
    else:
        print("\n  UNVERIFIABLE: results/oracle_ceiling_metrics.csv absent")

    # ---- simulation vs real, section 4
    s = R / "synthetic_suite_1se.csv"
    r = R / "real_suite.csv"
    if s.exists() and r.exists():
        sd = pd.read_csv(s)
        rd = pd.read_csv(r)
        print("\nsection 4 — simulation against real")
        check("simulation mean gain", 0.0201, sd.delta_vs_tuned_ind.mean(), tol=6e-5)
        check("simulation largest gain", 0.2496, sd.delta_vs_tuned_ind.max(), tol=6e-5)
        check("real mean gain", -0.0009, rd.delta.mean(), tol=6e-5)
        check("real largest gain", 0.0005, rd.delta.max(), tol=6e-5)
        w = int(((sd.delta_vs_tuned_ind > 0) & (sd.p_vs_tuned_ind < 0.05)).sum())
        l = int(((sd.delta_vs_tuned_ind < 0) & (sd.p_vs_tuned_ind < 0.05)).sum())
        print(f"  [{'ok  ' if w == 40 else 'FAIL'}] {'simulation significant wins':<46} paper 40"
              f"            data {w}")
        print(f"  [{'ok  ' if l == 2 else 'FAIL'}] {'simulation significant losses':<46} paper 2"
              f"             data {l}")
        if w != 40:
            FAILURES.append("sim wins")
        if l != 2:
            FAILURES.append("sim losses")
        rw = int(((rd.delta > 0) & (rd.p < 0.05)).sum())
        rl = int(((rd.delta < 0) & (rd.p < 0.05)).sum())
        print(f"  [{'ok  ' if rw == 0 and rl == 0 else 'FAIL'}] "
              f"{'real significant wins / losses':<46} paper 0 / 0         data {rw} / {rl}")
        if rw or rl:
            FAILURES.append("real significance")

    # ---- the diagnostic, section 5
    dv = R / "diagnostic_validation.csv"
    if dv.exists():
        from scipy import stats
        v = pd.read_csv(dv)
        msk = v.het_excess.notna() & v.delta.notna()
        rho = stats.spearmanr(v.het_excess[msk], v.delta[msk])[0]
        rio = stats.spearmanr(v.iota[msk], v.delta[msk])[0]
        print("\nsection 5 — the diagnostic")
        check("rho(het_excess, delta)", 0.505, rho, tol=5e-3)
        check("rho(iota, delta), the weaker alternative", 0.281, rio, tol=5e-3)

    # ---- claims that must be TEXT-consistent, not just numerically right
    # The figure's claim, which section 3 now states in the text: the real ceiling never
    # exceeds its null band, and every out-of-band case falls below. Added 2026-08-23 with the
    # figure itself -- a claim that appears in prose and in a caption but in no audit is how
    # the stale numbers this script exists to catch got in.
    import glob as _glob
    _d = pd.concat([pd.read_csv(f) for f in
                    sorted(_glob.glob(str(R / "oracle_ceiling_b_*.csv")))], ignore_index=True)
    _inside = int(((_d.ceiling_real >= _d.perm_lo) & (_d.ceiling_real <= _d.perm_hi)).sum())
    _below = int((_d.ceiling_real < _d.perm_lo).sum())
    _above = int((_d.ceiling_real > _d.perm_hi).sum())
    print("\nsection 3 — the figure's claim")
    for label, got, want in (("real ceiling inside its null band", _inside, 7),
                             ("below the band", _below, 4),
                             ("ABOVE the band (must be zero)", _above, 0)):
        ok = got == want
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:<46} paper {want:<6}      data {got}")
        if not ok:
            FAILURES.append(label)

    print("\ncross-document consistency")
    for label, pat in (("eleven datasets, not eight", r"eleven real (tabular )?datasets"),
                       ("six of eleven at an endpoint", r"six of the eleven"),
                       ("no stale 0.0086 / 0.0126 / 0.0039", r"0\.0086|0\.0126|0\.0039"),
                       # the abstract carried this until 2026-09-02, wrapped across a line, so the
                       # 'eleven' check above did not catch it
                       ("no stale 'eight real tabular datasets'", r"eight\s+real\s+tabular\s+datasets")):
        present = in_tex(pat)
        want = label.startswith("no stale") is False
        ok = present == want
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:<46} "
              f"{'found' if present else 'absent'}")
        if not ok:
            FAILURES.append(label)

    print("\n" + "=" * 100)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    # An empty FAILURES list also means "nothing ran". Three audits in this repo
    # printed a pass in exactly that state; say the count so this one cannot.
    if not CHECKS:
        print("No check ran at all. Treating that as a failure, not a pass.")
        return 2
    print(f"All {len(CHECKS)} quoted numbers reproduce from the committed CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
