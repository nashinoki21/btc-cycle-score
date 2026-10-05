"""Bitcoin long-term buy/sell score (0 = strong sell, 100 = strong buy).

Run:  python3 btc_score.py          -> prints breakdown, writes gauge.html
      python3 btc_score.py --test   -> self-check of the indicator math
Not financial advice.
"""
import json
import math
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from string import Template

# Long-term horizon: valuation/cycle indicators carry 55%.
WEIGHTS = {
    "rainbow": 20, "puell": 20, "cycle": 15, "ma": 15,
    "rsi": 10, "macd": 10, "vwap": 10,
}
HALVINGS = [date(2012, 11, 28), date(2016, 7, 9), date(2020, 5, 11), date(2024, 4, 20)]
GENESIS = date(2009, 1, 3)
# (days since halving, signal). ponytail: hand-fit from the 2012-2024 cycles
# (peak ~12-18 months after halving, bottom ~2.5-3 years after); refit if cycles stretch.
CYCLE_CURVE = [(0, 0.2), (365, -0.5), (550, -1), (900, 0.5), (1000, 1), (1100, 1), (1460, 0.2)]
ZONES = [(20, "Strong Sell"), (40, "Sell"), (60, "Neutral"), (80, "Buy"), (101, "Strong Buy")]


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def sign(x):
    return (x > 0) - (x < 0)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-score"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# ---- indicator math -------------------------------------------------------

def sma(xs, n):
    return sum(xs[-n:]) / n


def ema_series(xs, n):
    k, out = 2 / (n + 1), [xs[0]]
    for x in xs[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def rsi(closes, n=14):
    gains = [max(b - a, 0) for a, b in zip(closes, closes[1:])]
    losses = [max(a - b, 0) for a, b in zip(closes, closes[1:])]
    g, l = sum(gains[:n]) / n, sum(losses[:n]) / n
    for gi, li in zip(gains[n:], losses[n:]):  # Wilder smoothing
        g, l = (g * (n - 1) + gi) / n, (l * (n - 1) + li) / n
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def interp(points, x):
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def puell_signal(p):
    # 0.5 -> +1, 1.0 -> 0, 4.0 -> -1 (log scale)
    return clamp(math.log(p) / math.log(0.5)) if p <= 1 else clamp(-math.log(p) / math.log(4))


def mayer_signal(m):
    return clamp(1 - 2 * (m - 0.8) / 1.6)  # 0.8 -> +1, 2.4 -> -1


def discount_signal(price, ref):
    return clamp((ref - price) / ref / 0.2)  # 20% below ref -> +1, 20% above -> -1


def linfit(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return a, my - a * mx


def zone(score):
    return next(name for top, name in ZONES if score < top)


# ---- model ----------------------------------------------------------------

def compute(kl, cm, today):
    """kl: Binance daily klines (closed). cm: [(date, price, issuance_usd)] from 2010."""
    hi = [float(k[2]) for k in kl]
    lo = [float(k[3]) for k in kl]
    close = [float(k[4]) for k in kl]
    vol = [float(k[5]) for k in kl]
    price = close[-1]
    rows = {}

    # Rainbow: log10(price) = a*ln(days since genesis) + b, today's residual percentile.
    xs = [math.log((d - GENESIS).days) for d, _, _ in cm]
    ys = [math.log10(p) for _, p, _ in cm]
    a, b = linfit(xs, ys)
    res = [y - (a * x + b) for x, y in zip(xs, ys)]
    pct = sum(r <= res[-1] for r in res) / len(res)
    rows["rainbow"] = ("Rainbow chart", f"{pct:.0%} of history below, fair value ${10 ** (a * xs[-1] + b):,.0f}", 1 - 2 * pct)

    iss = [i for _, _, i in cm]
    puell = iss[-1] / (sum(iss[-365:]) / 365)
    rows["puell"] = ("Puell multiple", f"{puell:.2f}", puell_signal(puell))

    last = max(h for h in HALVINGS if h <= today)
    days = (today - last).days
    rows["cycle"] = ("4-year cycle", f"day {days} since {last:%b %Y} halving", interp(CYCLE_CURVE, days))

    s200, s50 = sma(close, 200), sma(close, 50)
    e20, e50 = ema_series(close, 20)[-1], ema_series(close, 50)[-1]
    mayer = price / s200
    cross = (sign(s50 - s200) + sign(e20 - e50)) / 2
    rows["ma"] = ("Moving averages", f"price/200SMA {mayer:.2f}, 50{'>' if s50 > s200 else '<'}200 SMA, 20{'>' if e20 > e50 else '<'}50 EMA",
                  2 / 3 * mayer_signal(mayer) + 1 / 3 * cross)

    r = rsi(close)
    rows["rsi"] = ("RSI (14)", f"{r:.1f}", clamp((50 - r) / 20))

    line = [f - s for f, s in zip(ema_series(close, 12), ema_series(close, 26))]
    hist = [m - s for m, s in zip(line, ema_series(line, 9))]
    rows["macd"] = ("MACD (12, 26, 9)", f"histogram {hist[-1]:+,.0f}, {'rising' if hist[-1] > hist[-2] else 'falling'}",
                    0.5 * sign(hist[-1]) + 0.5 * sign(hist[-1] - hist[-2]))

    typ = [(h + l + c) / 3 for h, l, c in zip(hi, lo, close)]
    vwap = sum(t * v for t, v in zip(typ[-90:], vol[-90:])) / sum(vol[-90:])
    # ponytail: volume profile from daily bars (whole day's volume at typical price); intraday bars if precision matters.
    lo365, hi365 = min(lo[-365:]), max(hi[-365:])
    width = (hi365 - lo365) / 50
    bins = [0.0] * 50
    for t, v in zip(typ[-365:], vol[-365:]):
        bins[min(int((t - lo365) / width), 49)] += v
    poc = lo365 + (bins.index(max(bins)) + 0.5) * width
    rows["vwap"] = ("VWAP / Volume profile", f"90d VWAP ${vwap:,.0f}, 1y POC ${poc:,.0f}",
                    (discount_signal(price, vwap) + discount_signal(price, poc)) / 2)

    s = sum(WEIGHTS[k] * rows[k][2] for k in WEIGHTS) / sum(WEIGHTS.values())
    return price, (s + 1) * 50, rows


def load():
    kl = fetch("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000")[:-1]  # drop open candle
    data = fetch("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
                 "?assets=btc&metrics=PriceUSD,IssTotUSD&frequency=1d&page_size=10000&paging_from=start")["data"]
    cm = [(date.fromisoformat(d["time"][:10]), float(d["PriceUSD"]), float(d["IssTotUSD"]))
          for d in data if d.get("PriceUSD") and d.get("IssTotUSD")]
    return kl, cm


# ---- output ---------------------------------------------------------------

def arc(a, b, r=100, cx=120, cy=120):
    def pt(v):
        t = math.pi * (1 - v / 100)
        return f"{cx + r * math.cos(t):.2f} {cy - r * math.sin(t):.2f}"
    return f"M {pt(a)} A {r} {r} 0 0 1 {pt(b)}"


def render(price, score, rows, asof):
    def polar(v, r):
        t = math.pi * (1 - v / 100)
        return 120 + r * math.cos(t), 120 - r * math.sin(t)

    ticks = "".join(
        '<line x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}" class="tick-mark"/>'.format(*polar(v, 110), *polar(v, 114))
        for v in range(0, 101, 10))
    zones = f'<path d="{arc(0, 100)}" class="track"/><path d="{arc(0, 100)}" class="arc"/>{ticks}'
    nx, ny = polar(score, 72)
    mx, my = polar(score, 100)
    needle = (f'<line x1="120" y1="120" x2="{nx:.2f}" y2="{ny:.2f}" class="needle"/>'
              f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="6" class="marker"/>')
    trs = "".join(
        f'<div class="row"><div class="name">{name}<span class="chip">{WEIGHTS[k]}%</span></div>'
        f'<div class="read">{reading}</div>'
        f'<div class="sig"><div class="bar"><span style="{"left:50%;width" if s >= 0 else f"left:{50 + s * 50:.1f}%;width"}:{abs(s) * 50:.1f}%" class="{"pos" if s >= 0 else "neg"}"></span></div>'
        f'<span class="sv {"pos" if s >= 0 else "neg"}">{s:+.2f}</span></div></div>'
        for k, (name, reading, s) in rows.items())
    tpl = Template((Path(__file__).parent / "gauge_template.html").read_text())
    return tpl.substitute(zones=zones, needle=needle, score=f"{score:.0f}", zone=zone(score),
                          zcls=zone(score).lower().replace(" ", "-"), price=f"${price:,.0f}",
                          asof=asof, rows=trs)


def page():
    kl, cm = load()
    price, score, rows = compute(kl, cm, datetime.now(timezone.utc).date())
    asof = datetime.fromtimestamp(kl[-1][6] / 1000, timezone.utc).strftime("%d %b %Y")
    return price, score, rows, render(price, score, rows, asof)


def main():
    price, score, rows, html = page()
    print(f"BTC ${price:,.0f}  score {score:.0f}/100  {zone(score)}\n")
    for k, (name, reading, s) in rows.items():
        print(f"  {name:<24}{WEIGHTS[k]:>3}%  {s:+.2f}  {reading}")
    out = Path(__file__).parent / "gauge.html"
    out.write_text(html)
    print(f"\nwrote {out}")


def test():
    assert rsi(list(range(1, 40))) == 100
    assert abs(rsi([1, 2] * 20) - 50) < 5
    assert sma([1, 2, 3, 4], 2) == 3.5
    assert ema_series([5.0] * 30, 9)[-1] == 5.0
    assert puell_signal(0.5) == 1 and puell_signal(1) == 0 and abs(puell_signal(4) + 1) < 1e-9
    assert mayer_signal(0.8) == 1 and abs(mayer_signal(2.4) + 1) < 1e-9
    assert interp(CYCLE_CURVE, 550) == -1 and interp(CYCLE_CURVE, 1050) == 1
    assert linfit([0, 1, 2], [1, 3, 5]) == (2, 1)
    assert zone(0) == "Strong Sell" and zone(50) == "Neutral" and zone(100) == "Strong Buy"
    # synthetic market: steady uptrend must score inside 0..100 and trend signals must be bullish
    kl = [[0, 0, 100 + i * 1.01, 100 + i * 0.99, 100 + i, 10, 0] for i in range(400)]
    cm = [(date(2010, 7, 18).fromordinal(date(2010, 7, 18).toordinal() + i), 0.1 * 1.002 ** i, 1000.0)
          for i in range(5000)]
    _, score, rows = compute(kl, cm, date(2026, 10, 5))
    assert 0 <= score <= 100 and rows["ma"][2] < 1 and rows["rsi"][2] == -1
    print("ok")


if __name__ == "__main__":
    test() if "--test" in sys.argv else main()
