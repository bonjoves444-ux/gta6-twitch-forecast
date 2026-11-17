"""Launch-day Twitch viewers, hour by hour, from comparable launches.

The forecast is an ensemble of two per-hour ridge models fitted on the
past launches shipped in the case ("comparables"), both in log space:

* a kernel-weighted local ridge on the log ratio of each hour's viewers to
  hour 1's viewers, weighted towards launches whose hour-1 size and
  launch surge look like this launch's;
* a global log-log ridge that predicts each hour's log viewers directly
  from hour 1's log viewers and the pre-release covariates, which is what
  keeps very large launches (whose hour-1 print is already near the peak)
  from being dragged up by small launches' steep evening rises.

The two medians are averaged in log space. The spread is the median of the
two models' implied log spreads, shrunk a little: the practice launches
show that the raw residual spread is wider than the pinball loss rewards.

Covariates, all computable from the case alone (no network, no model):

* logv1        -- log of hour 1's viewers;
* surge7       -- log of hour 1 over the mean viewers of the 7 prior days;
* prepeak24    -- log of the highest hourly viewers in the 24 h before the
                  release day, over hour 1 (how much audience was already
                  watching before the US unlock);
* eve_ratio    -- log of the mean viewers at 17:00-20:00 UTC on the prior
                  day, over hour 1 (the pre-release evening audience);
* premean24    -- log of the mean viewers of the prior 24 h, over hour 1.

A launch whose hour 1 is a deep trough (small, or already past its
New Zealand / Europe wave) rises much further into the evening than one
whose hour 1 is already the peak, and the covariates above are what tell
those two apart. When the pool is too thin to fit, the program falls back
to the house persistence baseline for the affected hours.

Standard library only. Deterministic: same input bytes give same output
bytes. Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
from datetime import datetime, timedelta

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
FEATURES = ["logv1", "surge7", "prepeak24", "eve_ratio", "premean24"]
INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def mean(values):
    return sum(values) / len(values)


def pstdev(values, mu):
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting; matrix is small (<=6)."""
    n = len(rhs)
    rows = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[piv] = rows[piv], rows[col]
        pv = rows[col][col] or 1e-9
        for r in range(n):
            if r != col:
                fac = rows[r][col] / pv
                for k in range(col, n + 1):
                    rows[r][k] -= fac * rows[col][k]
    return [rows[i][n] / (rows[i][i] or 1e-9) for i in range(n)]


def ridge_fit(xs, ys, weights, lam):
    """Weighted ridge regression; returns coefficients for [1, x1, ...]."""
    k = len(xs[0])
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi, w in zip(xs, ys, weights):
        for i in range(k):
            b[i] += w * xi[i] * yi
            for j in range(k):
                a[i][j] += w * xi[i] * xi[j]
    for i in range(1, k):
        a[i][i] += lam
    return solve(a, b)


def day_curve(hourly, day_start):
    """{hour 1..24: viewers} for the UTC release day, from hourly rows."""
    out = {}
    for row in hourly:
        try:
            t = parse_time(row["from"])
        except (KeyError, ValueError):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        v = row.get("viewers")
        if 1 <= h <= 24 and v:
            out[h] = v
    return out


def window_mean(hourly, day_start, days):
    vals = [r["viewers"] for r in hourly
            if r.get("viewers")
            and day_start - timedelta(days=days) <= parse_time(r["from"]) < day_start]
    return mean(vals) if vals else None


def launch_features(name, curve, hourly, day_start):
    """Feature dict for one launch, or None when its curve is unusable."""
    if 1 not in curve or len(curve) < 20:
        return None
    pre7 = window_mean(hourly, day_start, 7)
    pre1 = window_mean(hourly, day_start, 1)
    if not pre7 or not pre1:
        return None
    v1 = curve[1]
    last24 = [(parse_time(r["from"]), r["viewers"]) for r in hourly
              if r.get("viewers")
              and day_start - timedelta(days=1) <= parse_time(r["from"]) < day_start]
    if not last24:
        return None
    peak24 = max(v for _, v in last24)
    mean24 = mean([v for _, v in last24])
    eve = [v for t, v in last24 if t.hour in (17, 18, 19, 20)]
    if not eve:
        return None
    return {
        "name": name,
        "v1": v1,
        "logv1": math.log(v1),
        "surge7": math.log(v1 / pre7),
        "prepeak24": math.log(peak24 / v1),
        "eve_ratio": math.log(mean(eve) / v1),
        "premean24": math.log(mean24 / v1),
        "curve": curve,
        "ratio": {h: math.log(curve[h] / v1) for h in curve if h >= 2},
    }


def standardise(pool):
    stats = {}
    for fn in FEATURES:
        vals = [f[fn] for f in pool]
        mu = mean(vals)
        stats[fn] = (mu, pstdev(vals, mu) or 1.0)
    return stats


def xvec(feat, stats):
    return [1.0] + [(feat[fn] - stats[fn][0]) / stats[fn][1] for fn in FEATURES]


def kernel_weights(target, pool):
    out = []
    for f in pool:
        d2 = ((f["logv1"] - target["logv1"]) / 1.0) ** 2
        d2 += ((f["surge7"] - target["surge7"]) / 1.0) ** 2
        out.append(math.exp(-0.5 * d2))
    return out


def local_ratio_forecast(target, pool, stats):
    """Per-hour kernel-weighted ridge on log ratios; spread from residuals."""
    weights = kernel_weights(target, pool)
    out = {}
    for h in range(2, 25):
        items = [(f, w) for f, w in zip(pool, weights) if h in f["ratio"]]
        if len(items) < 5:
            continue
        xs = [xvec(f, stats) for f, _ in items]
        ys = [f["ratio"][h] for f, _ in items]
        ws = [w for _, w in items]
        beta = ridge_fit(xs, ys, ws, 3.0)
        m = sum(beta[i] * xvec(target, stats)[i] for i in range(len(beta)))
        resid = [y - sum(beta[j] * x[j] for j in range(len(beta)))
                 for x, y in zip(xs, ys)]
        wsum = sum(ws)
        var = sum(w * r * r for w, r in zip(ws, resid)) / wsum if wsum else 0.09
        spread = max(math.sqrt(var), 0.15)
        out[h] = [target["v1"] * math.exp(m + z * spread * 0.6) for z in Z]
    return out


def global_level_forecast(target, pool, stats):
    """Per-hour unweighted ridge on log viewers; spread from residuals."""
    out = {}
    for h in range(2, 25):
        items = [f for f in pool if h in f["curve"]]
        if len(items) < 5:
            continue
        xs = [xvec(f, stats) for f in items]
        ys = [math.log(f["curve"][h]) for f in items]
        beta = ridge_fit(xs, ys, [1.0] * len(items), 0.3)
        xt = xvec(target, stats)
        m = sum(beta[i] * xt[i] for i in range(len(beta)))
        resid = [y - sum(beta[j] * x[j] for j in range(len(beta)))
                 for x, y in zip(xs, ys)]
        spread = max(pstdev(resid, mean(resid)) if len(resid) > 1 else 0.3, 0.15)
        out[h] = [math.exp(m + z * spread * 0.5) for z in Z]
    return out


def baseline_hour(p1, h):
    return [round(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)), 2) for z in Z]


def forecast_case(case):
    known = case["known"][0]
    p1 = max(1, known["viewers"])
    day_start = parse_time(case["day_start"])
    # The target only has hour 1 so far: build its feature dict directly
    # from the known print plus the history the case ships.
    if True:
        pre7 = window_mean(case.get("history_hourly", []), day_start, 7)
        pre1 = window_mean(case.get("history_hourly", []), day_start, 1)
        last24 = [(parse_time(r["from"]), r["viewers"])
                  for r in case.get("history_hourly", [])
                  if r.get("viewers")
                  and day_start - timedelta(days=1) <= parse_time(r["from"]) < day_start]
        eve = [v for t, v in last24 if t.hour in (17, 18, 19, 20)]
        if not pre7 or not pre1 or not last24 or not eve:
            return {h: baseline_hour(p1, h) for h in range(2, 25)}
        target = {
            "name": case["game"]["name"],
            "v1": p1,
            "logv1": math.log(p1),
            "surge7": math.log(p1 / pre7),
            "prepeak24": math.log(max(v for _, v in last24) / p1),
            "eve_ratio": math.log(mean(eve) / p1),
            "premean24": math.log(mean([v for _, v in last24]) / p1),
            "curve": {1: p1},
            "ratio": {},
        }
    pool = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        curve = day_curve(comp.get("hourly", []), parse_time(comp["day_start"]))
        feat = launch_features(comp.get("name", ""), curve,
                               comp.get("hourly", []), parse_time(comp["day_start"]))
        if feat is not None:
            pool.append(feat)
    if len(pool) < 5:
        return {h: baseline_hour(p1, h) for h in range(2, 25)}
    stats = standardise(pool)
    try:
        first = local_ratio_forecast(target, pool, stats)
    except Exception:
        first = {}
    try:
        second = global_level_forecast(target, pool, stats)
    except Exception:
        second = {}
    hours = {}
    for h in range(2, 25):
        if h in first and h in second:
            med = (math.log(first[h][2]) + math.log(second[h][2])) / 2.0
            sig1 = (math.log(first[h][4]) - math.log(first[h][0])) / (2 * 1.645)
            sig2 = (math.log(second[h][4]) - math.log(second[h][0])) / (2 * 1.645)
            spread = max(sig1, sig2) * 0.7
            qs = [math.exp(med + z * spread) for z in Z]
        elif h in first:
            qs = first[h]
        elif h in second:
            qs = second[h]
        else:
            qs = baseline_hour(p1, h)
        clean = []
        prev = 0.0
        for q in qs:
            q = min(max(round(q, 2), 1.0), 1_000_000_000.0)
            q = max(q, prev)
            clean.append(q)
            prev = q
        hours[h] = clean
    return hours


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        hours = forecast_case(case)
        payload = {"hours": [{"hour": h, "q": hours[h]}
                             for h in sorted(hours)]}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
