from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_uc(exc: pd.Series, alpha: float) -> dict:
    """
    Kupiec Unconditional Coverage test.
    exc: série 0/1 (1 = exception)
    alpha: niveau VaR (0.95, 0.99)
    """
    e = exc.dropna().astype(int).to_numpy()
    n = int(len(e))
    x = int(e.sum())

    p = 1.0 - float(alpha)  # expected exception probability
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)

    phat = x / n if n > 0 else 0.0
    phat = min(max(phat, eps), 1 - eps)

    ll_h0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_h1 = (n - x) * np.log(1 - phat) + x * np.log(phat)

    LR_uc = float(-2.0 * (ll_h0 - ll_h1))
    p_value = float(chi2.sf(LR_uc, df=1))

    return {"n": n, "x": x, "rate": x / n, "LR_uc": LR_uc, "p_uc": p_value}


def christoffersen_ind(exc: pd.Series) -> dict:
    """
    Christoffersen Independence test.
    Teste si la suite d'exceptions est indépendante (Markov vs Bernoulli).
    """
    e = exc.dropna().astype(int).to_numpy()
    if len(e) < 2:
        return {"LR_ind": np.nan, "p_ind": np.nan, "n00": 0, "n01": 0, "n10": 0, "n11": 0}

    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(e)):
        prev, cur = e[i - 1], e[i]
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    eps = 1e-12

    # Probabilités conditionnelles
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    pi0 = min(max(pi0, eps), 1 - eps)
    pi1 = min(max(pi1, eps), 1 - eps)
    pi = min(max(pi, eps), 1 - eps)

    ll_indep = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_markov = (
        n00 * np.log(1 - pi0) + n01 * np.log(pi0) +
        n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    )

    LR_ind = float(-2.0 * (ll_indep - ll_markov))
    p_value = float(chi2.sf(LR_ind, df=1))

    return {"LR_ind": LR_ind, "p_ind": p_value, "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def conditional_coverage(exc: pd.Series, alpha: float) -> dict:
    """
    Christoffersen Conditional Coverage:
    LR_cc = LR_uc + LR_ind, ddl=2
    """
    uc = kupiec_uc(exc, alpha)
    ind = christoffersen_ind(exc)

    if np.isnan(ind["LR_ind"]):
        return {"LR_cc": np.nan, "p_cc": np.nan}

    LR_cc = float(uc["LR_uc"] + ind["LR_ind"])
    p_cc = float(chi2.sf(LR_cc, df=2))
    return {"LR_cc": LR_cc, "p_cc": p_cc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)

    print(f"[SCORE] file={path.name} alpha={args.alpha}")

    for model in ("hist", "param", "mc", "ewma", "fhs"):
        col = f"exc_{model}"
        if col not in df.columns:
            continue

        exc = df[col]
        uc = kupiec_uc(exc, args.alpha)
        ind = christoffersen_ind(exc)
        cc = conditional_coverage(exc, args.alpha)

        print(
            f"- {model.upper():5s} | n={uc['n']:3d} | exc={uc['x']:2d} | rate={uc['rate']:.4f} "
            f"| Kupiec p={uc['p_uc']:.4f} "
            f"| Ind p={ind['p_ind']:.4f} "
            f"| CC p={cc['p_cc']:.4f}"
        )


if __name__ == "__main__":
    main()
