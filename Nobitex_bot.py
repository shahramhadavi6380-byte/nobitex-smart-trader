import os
import json
import time
import math
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd


BASE = "https://apiv2.nobitex.ir"

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

STATE_FILE = "bot_state.json"

SCREEN_TOP_PER_QUOTE = 10
MAX_WATCH = 5

SCAN_EVERY_SECONDS = 4 * 60 * 60

TIMEFRAMES = (15, 60, 240)
CANDLE_COUNT = 350

S = requests.Session()

S.headers.update({
    "User-Agent": "NobitexSmartAlerts/2.0",
    "Accept": "application/json"
})


def now_ts():
    return int(time.time())


def num(x, default=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def fmt(x):
    if x is None or not math.isfinite(num(x)):
        return "نامشخص"

    x = float(x)

    if abs(x) >= 1000:
        return f"{x:,.0f}"

    if abs(x) >= 1:
        return f"{x:,.5f}"

    return f"{x:.8f}"


def api_get(path, params=None, attempts=4):

    for n in range(attempts):

        try:

            r = S.get(
                BASE + path,
                params=params,
                timeout=25
            )

            if r.status_code == 429:
                time.sleep(min(20, 4 * (n + 1)))
                continue

            r.raise_for_status()

            data = r.json()

            if isinstance(data, dict) and data.get("status") == "failed":
                time.sleep(2 * (n + 1))
                continue

            return data

        except Exception as e:

            print("API error", path, e)

            time.sleep(2 * (n + 1))

    return None


def telegram(text):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    for i in range(0, len(text), 3800):

        r = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": text[i:i + 3800]
            },
            timeout=25
        )

        r.raise_for_status()

        if not r.json().get("ok"):
            raise RuntimeError(
                "Telegram send failed: " + str(r.text)
            )

        time.sleep(1)


def load_state():

    try:

        with open(STATE_FILE, encoding="utf-8") as f:
            x = json.load(f)

        if isinstance(x, dict):
            return x

    except Exception:
        pass

    return {
        "last_scan": 0,
        "watchlist": [],
        "last_monitor": 0
    }


def save_state(state):

    with open(STATE_FILE, "w", encoding="utf-8") as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )


def get_markets():

    data = api_get("/market/stats")

    if not data or not isinstance(data.get("stats"), dict):

        raise RuntimeError(
            "دریافت فهرست بازارهای نوبیتکس ناموفق بود"
        )

    out = {}

    for key, v in data["stats"].items():

        if not isinstance(v, dict):
            continue

        parts = key.lower().split("-")

        if len(parts) != 2:
            continue

        base, quote = parts

        if quote not in ("rls", "usdt"):
            continue

        if str(v.get("isClosed", False)).lower() == "true":
            continue

        symbol = base.upper() + (
            "IRT" if quote == "rls" else "USDT"
        )

        price = num(v.get("latest"), 0)

        if price <= 0:
            continue

        out[symbol] = {
            "symbol": symbol,
            "quote": quote,
            "price": price,
            "volume": num(v.get("volumeDst"), 0),
            "change24": num(v.get("dayChange"), 0)
        }

    return list(out.values())


def candles(symbol, resolution):

    tnow = now_ts()

    raw = api_get(
        "/market/udf/history",
        {
            "symbol": symbol,
            "resolution": str(resolution),
            "to": tnow,
            "countback": CANDLE_COUNT
        }
    )

    if not raw or raw.get("s") != "ok":
        return None

    try:

        df = pd.DataFrame({
            "time": raw["t"],
            "open": raw["o"],
            "high": raw["h"],
            "low": raw["l"],
            "close": raw["c"],
            "volume": raw.get(
                "v",
                [0] * len(raw["t"])
            )
        })

        for c in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "time"
        ):

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        df = (
            df.dropna()
            .sort_values("time")
            .drop_duplicates("time")
        )

        # حذف کندل در حال تشکیل

        df = df[
            df.time + resolution * 60 <= tnow
        ]

        if len(df) < 220:
            return None

        # جلوگیری از استفاده از کندل قدیمی

        if (
            tnow -
            int(df.time.iloc[-1] + resolution * 60)
            > resolution * 60 * 2
        ):
            return None

        return df.reset_index(drop=True)

    except Exception as e:

        print(
            "Candle parse error",
            symbol,
            resolution,
            e
        )

        return None


def indicators(df):

    d = df.copy()

    c = d.close
    h = d.high
    l = d.low
    v = d.volume

    # EMA

    d["ema20"] = c.ewm(
        span=20,
        adjust=False
    ).mean()

    d["ema50"] = c.ewm(
        span=50,
        adjust=False
    ).mean()

    d["ema200"] = c.ewm(
        span=200,
        adjust=False
    ).mean()

    # RSI

    delta = c.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    al = loss.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = ag / al.replace(0, np.nan)

    d["rsi"] = 100 - 100 / (1 + rs)

    d.loc[
        (al == 0) & (ag > 0),
        "rsi"
    ] = 100

    d.loc[
        (al == 0) & (ag == 0),
        "rsi"
    ] = 50

    # MACD

    e12 = c.ewm(
        span=12,
        adjust=False
    ).mean()

    e26 = c.ewm(
        span=26,
        adjust=False
    ).mean()

    d["macd"] = e12 - e26

    d["macd_signal"] = d.macd.ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd_hist"] = (
        d.macd - d.macd_signal
    )

    # ATR و ADX

    pc = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - pc).abs(),
            (l - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    up = h.diff()
    down = -l.diff()

    pdm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0
        ),
        index=d.index
    )

    mdm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0
        ),
        index=d.index
    )

    atr = tr.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    psm = pdm.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    msm = mdm.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    pdi = 100 * psm / atr.replace(0, np.nan)
    mdi = 100 * msm / atr.replace(0, np.nan)

    denom = (pdi + mdi).replace(0, np.nan)

    dx = 100 * (pdi - mdi).abs() / denom

    d["atr"] = atr
    d["plus_di"] = pdi
    d["minus_di"] = mdi

    d["adx"] = dx.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    # Bollinger Bands

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()

    d["bb_mid"] = mid
    d["bb_upper"] = mid + 2 * std
    d["bb_lower"] = mid - 2 * std

    # Ichimoku

    tenkan = (
        h.rolling(9).max() +
        l.rolling(9).min()
    ) / 2

    kijun = (
        h.rolling(26).max() +
        l.rolling(26).min()
    ) / 2

    d["tenkan"] = tenkan
    d["kijun"] = kijun

    d["cloud_a"] = (
        (tenkan + kijun) / 2
    ).shift(26)

    d["cloud_b"] = (
        (
            h.rolling(52).max() +
            l.rolling(52).min()
        ) / 2
    ).shift(26)

    # OBV

    direction = np.sign(
        c.diff()
    ).fillna(0)

    d["obv"] = (
        direction * v
    ).cumsum()

    d["obv_ma"] = d.obv.rolling(20).mean()

    return d


def analyze(df):

    d = indicators(df)

    x = d.iloc[-1]

    keys = [
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "adx",
        "plus_di",
        "minus_di",
        "cloud_a",
        "cloud_b",
        "obv",
        "obv_ma",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "tenkan",
        "kijun"
    ]

    z = {
        k: num(x[k])
        for k in keys
    }

    required = [
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "macd",
        "macd_signal",
        "atr"
    ]

    if not all(
        math.isfinite(z[k])
        for k in required
    ):
        return None

    # امتیازدهی

    score = 50

    score += (
        7 if z["close"] > z["ema20"]
        else -7
    )

    score += (
        8 if z["ema20"] > z["ema50"]
        else -8
    )

    score += (
        8 if z["ema50"] > z["ema200"]
        else -8
    )

    score += (
        8 if z["macd"] > z["macd_signal"]
        else -8
    )

    score += (
        7 if z["rsi"] >= 55
        else (-7 if z["rsi"] <= 45 else 0)
    )

    if (
        math.isfinite(z["adx"])
        and z["adx"] >= 20
    ):

        score += (
            5 if z["plus_di"] > z["minus_di"]
            else (
                -5
                if z["minus_di"] > z["plus_di"]
                else 0
            )
        )

    if (
        math.isfinite(z["cloud_a"])
        and math.isfinite(z["cloud_b"])
    ):

        score += (
            5
            if z["close"] > max(
                z["cloud_a"],
                z["cloud_b"]
            )
            else (
                -5
                if z["close"] < min(
                    z["cloud_a"],
                    z["cloud_b"]
                )
                else 0
            )
        )

    if math.isfinite(z["obv_ma"]):

        score += (
            3 if z["obv"] > z["obv_ma"]
            else -3
        )

    z["bb_upper"] = num(x["bb_upper"])
    z["bb_mid"] = num(x["bb_mid"])
    z["bb_lower"] = num(x["bb_lower"])

    z["tenkan"] = num(x["tenkan"])
    z["kijun"] = num(x["kijun"])

    z["score"] = max(
        0,
        min(100, score)
    )

    z["last_candle_time"] = int(
        df.time.iloc[-1]
    )

    # Fibonacci

    recent = df.tail(120)

    swing_high = float(
        recent.high.max()
    )

    swing_low = float(
        recent.low.min()
    )

    swing = swing_high - swing_low

    z["fib_382"] = (
        swing_low + 0.382 * swing
        if swing > 0 else float("nan")
    )

    z["fib_500"] = (
        swing_low + 0.500 * swing
        if swing > 0 else float("nan")
    )

    z["fib_618"] = (
        swing_low + 0.618 * swing
        if swing > 0 else float("nan")
    )

    # Andrews Pitchfork approximation

    highs = df.high.to_numpy()
    lows = df.low.to_numpy()

    pivots = []

    for i in range(4, len(df) - 4):

        if highs[i] >= np.max(
            highs[i-4:i+5]
        ):

            pivots.append(
                (i, float(highs[i]), "H")
            )

        if lows[i] <= np.min(
            lows[i-4:i+5]
        ):

            pivots.append(
                (i, float(lows[i]), "L")
            )

    pivots.sort(
        key=lambda p: p[0]
    )

    clean = []

    for p in pivots:

        if clean and clean[-1][2] == p[2]:

            if (
                (p[2] == "H" and p[1] > clean[-1][1])
                or
                (p[2] == "L" and p[1] < clean[-1][1])
            ):

                clean[-1] = p

        else:

            clean.append(p)

    z["pitchfork_median"] = float("nan")

    if len(clean) >= 3:

        p1, p2, p3 = clean[-3:]

        mid_x = (
            p2[0] + p3[0]
        ) / 2

        mid_y = (
            p2[1] + p3[1]
        ) / 2

        if mid_x != p1[0]:

            slope = (
                mid_y - p1[1]
            ) / (
                mid_x - p1[0]
            )

            z["pitchfork_median"] = (
                p1[1] +
                slope * (
                    (len(df) - 1) - p1[0]
                )
            )

    return z


def spread_pct(symbol):

    ob = api_get(
        f"/v3/orderbook/{symbol}"
    )

    try:

        bids = ob["bids"]
        asks = ob["asks"]

        bid = num(bids[0][0])
        ask = num(asks[0][0])

        if bid <= 0 or ask <= bid:
            return None

        return (
            100 * (ask - bid) /
            ((ask + bid) / 2)
        )

    except Exception:

        return None


def full_market(m):

    res = {}

    for tf in TIMEFRAMES:

        df = candles(
            m["symbol"],
            tf
        )

        if df is None:
            return None

        a = analyze(df)

        if a is None:
            return None

        res[tf] = a

        time.sleep(1.05)

    a15 = res[15]
    a1 = res[60]
    a4 = res[240]

    # فیلتر روند سه تایم‌فریمی

    if not (
        a4["score"] >= 60
        and a1["score"] >= 60
        and a15["score"] >= 50
    ):
        return None

    if not (
        50 <= a1["rsi"] <= 68
        and 48 <= a4["rsi"] <= 68
        and a4["adx"] >= 18
    ):
        return None

    if not (
        a1["macd"] > a1["macd_signal"]
        and
        a4["macd"] > a4["macd_signal"]
    ):
        return None

    # جلوگیری از ورود در قیمت بیش از حد بالا

    if (
        a15["close"] >
        a15["ema20"] + 1.0 * a15["atr"]
    ):
        return None

    spread = spread_pct(
        m["symbol"]
    )

    if spread is None or spread > 0.8:
        return None

    # محدوده ورود

    entry_low = max(
        a15["ema20"] - 0.35 * a15["atr"],
        a15["close"] - 0.75 * a15["atr"]
    )

    entry_high = (
        a15["ema20"] + 0.15 * a15["atr"]
    )

    ref = (
        entry_low + entry_high
    ) / 2

    # حد ضرر و اهداف

    sl = ref - 1.5 * a4["atr"]

    tp1 = ref + 2.0 * a4["atr"]

    tp2 = ref + 3.0 * a4["atr"]

    average = (
        a15["score"] +
        a1["score"] +
        a4["score"]
    ) / 3

    rank = (
        average +
        max(0, 0.8 - spread) * 2
    )

    return {

        "symbol": m["symbol"],
        "quote": m["quote"],
        "volume": m["volume"],

        "score": round(average, 1),
        "rank": rank,

        "spread": spread,

        "price_at_scan": a15["close"],

        "entry_low": entry_low,
        "entry_high": entry_high,

        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,

        "scores": {
            "15m": a15["score"],
            "1h": a1["score"],
            "4h": a4["score"]
        },

        "indicators": {

            str(tf): {
                k: res[tf].get(k)

                for k in (
                    "rsi",
                    "adx",
                    "macd_hist",
                    "ema20",
                    "ema50",
                    "ema200",
                    "atr",
                    "bb_upper",
                    "bb_mid",
                    "bb_lower",
                    "tenkan",
                    "kijun",
                    "cloud_a",
                    "cloud_b",
                    "fib_382",
                    "fib_500",
                    "fib_618",
                    "pitchfork_median"
                )
            }

            for tf in TIMEFRAMES
        },

        "created": now_ts(),
        "status": "WAITING",
        "alerts": []
    }


def run_full_scan(state):

    print("Starting full market scan")

    markets = get_markets()

    screened = []

    # غربالگری همه بازارها با تایم‌فریم یک‌ساعته

    for i, m in enumerate(markets, 1):

        print(
            f"1h screen {i}/{len(markets)}: {m['symbol']}"
        )

        df = candles(
            m["symbol"],
            60
        )

        if df is not None:

            a = analyze(df)

            if a is not None:

                m["screen_score"] = a["score"]

                screened.append(m)

        time.sleep(1.05)

    # انتخاب ۱۰ بازار پرحجم در هر گروه

    selected = []

    for quote in ("rls", "usdt"):

        group = [
            m for m in screened
            if m["quote"] == quote
        ]

        group.sort(
            key=lambda x: x["volume"],
            reverse=True
        )

        selected.extend(
            group[:SCREEN_TOP_PER_QUOTE]
        )

    # تحلیل عمیق

    candidates = []

    for i, m in enumerate(selected, 1):

        print(
            f"Full analysis {i}/{len(selected)}: {m['symbol']}"
        )

        try:

            item = full_market(m)

            if item:
                candidates.append(item)

        except Exception as e:

            print(
                "Full analysis failed",
                m["symbol"],
                e
            )

    candidates.sort(
        key=lambda x: x["rank"],
        reverse=True
    )

    watch = candidates[:MAX_WATCH]

    state["watchlist"] = watch

    state["last_scan"] = now_ts()

    state["last_scan_markets"] = len(markets)

    state["last_scan_candidates"] = len(candidates)

    # گزارش تلگرام

    msg = [

        "🔎 اسکن کامل نوبیتکس انجام شد",

        f"بازارهای فعال بررسی‌شده: {len(markets)}",

        f"بازارهای تحلیل عمیق: {len(selected)}",

        f"فرصت‌های واجد شرایط: {len(candidates)}",

        f"فهرست پایش: {len(watch)} ارز (حداکثر ۵ مورد)",

        "",

        "این‌ها فرصت‌های قابل بررسی‌اند، نه دستور خرید قطعی."
    ]

    if watch:

        for i, x in enumerate(watch, 1):

            msg += [

                f"\n{i}) {x['symbol']} | امتیاز {x['score']}/100",

                f"تایم‌فریم‌ها ۱۵د/۱س/۴س: {x['scores']['15m']}/{x['scores']['1h']}/{x['scores']['4h']}",

                f"محدوده ورود: {fmt(x['entry_low'])} تا {fmt(x['entry_high'])}",

                f"حد ضرر: {fmt(x['sl'])} | هدف۱: {fmt(x['tp1'])} | هدف۲: {fmt(x['tp2'])}",

                f"اختلاف خرید/فروش: {x['spread']:.3f}%",

                f"RSI (۱۵د/۱س/۴س): {fmt(x['indicators']['15']['rsi'])}/{fmt(x['indicators']['60']['rsi'])}/{fmt(x['indicators']['240']['rsi'])}",

                f"ADX (۱۵د/۱س/۴س): {fmt(x['indicators']['15']['adx'])}/{fmt(x['indicators']['60']['adx'])}/{fmt(x['indicators']['240']['adx'])}",

                f"MACD Hist (۱س/۴س): {fmt(x['indicators']['60']['macd_hist'])}/{fmt(x['indicators']['240']['macd_hist'])}",

                f"EMA20/50/200 چهار‌ساعته: {fmt(x['indicators']['240']['ema20'])}/{fmt(x['indicators']['240']['ema50'])}/{fmt(x['indicators']['240']['ema200'])}",

                f"بولینگر ۴س (پایین/میانه/بالا): {fmt(x['indicators']['240']['bb_lower'])}/{fmt(x['indicators']['240']['bb_mid'])}/{fmt(x['indicators']['240']['bb_upper'])}",

                f"فیبوناچی ۴س (۳۸.۲/۵۰/۶۱.۸): {fmt(x['indicators']['240']['fib_382'])}/{fmt(x['indicators']['240']['fib_500'])}/{fmt(x['indicators']['240']['fib_618'])}",

                f"پیچفورک میانه ۴س: {fmt(x['indicators']['240']['pitchfork_median'])}"
            ]

    else:

        msg.append(
            "فعلاً هیچ ارزی تمام فیلترهای سخت‌گیرانه را نگذرانده؛ فهرست خرید خالی ماند."
        )

    msg += [

        "",

        "پایش قیمت این فهرست از چرخه‌های بعدی، هر ۵ دقیقه انجام می‌شود."
    ]

    telegram(
        "\n".join(msg)
    )


def current_price(symbol):

    data = api_get("/market/stats")

    if not data or not isinstance(data.get("stats"), dict):
        return None

    key = (
        symbol.lower()
        .replace("irt", "-rls")
        .replace("usdt", "-usdt")
    )

    item = data["stats"].get(key)

    if not item:
        return None

    if str(item.get("isClosed", False)).lower() == "true":
        return None

    return num(
        item.get("latest"),
        0
    ) or None


def monitor(state):

    watch = state.get("watchlist", [])

    if not watch:

        print("No watchlist to monitor")

        return

    for item in watch:

        symbol = item["symbol"]

        price = current_price(symbol)

        if not price:
            continue

        status = item.get(
            "status",
            "WAITING"
        )

        low = num(item["entry_low"])
        high = num(item["entry_high"])

        sl = num(item["sl"])
        tp1 = num(item["tp1"])
        tp2 = num(item["tp2"])

        alert = None

        # سیگنال هنوز وارد محدوده خرید نشده

        if status == "WAITING":

            df_wait = candles(
                symbol,
                60
            )

            trend_wait = (
                analyze(df_wait)
                if df_wait is not None
                else None
            )

            if price <= sl:

                item["status"] = "INVALIDATED"

                alert = (
                    f"⛔ سیگنال {symbol} قبل از ورود باطل شد؛ "
                    f"قیمت به حد ضرر محاسباتی رسید. "
                    f"قیمت: {fmt(price)}"
                )

            elif (
                trend_wait
                and trend_wait["score"] < 45
                and trend_wait["macd"] < trend_wait["macd_signal"]
            ):

                item["status"] = "INVALIDATED"

                alert = (
                    f"⚠️ سیگنال ورود {symbol} به‌دلیل "
                    f"تضعیف روند یک‌ساعته باطل شد. "
                    f"امتیاز ۱ساعته: {trend_wait['score']}"
                )

            elif low <= price <= high:

                item["status"] = "ENTERED"

                item["entry_alert_price"] = price

                alert = (

                    f"🟢 قیمت {symbol} وارد محدوده ورود شد\n"

                    f"قیمت فعلی: {fmt(price)}\n"

                    f"محدوده ورود: {fmt(low)} تا {fmt(high)}\n"

                    f"حد ضرر: {fmt(sl)}\n"

                    f"هدف۱: {fmt(tp1)} | هدف۲: {fmt(tp2)}\n"

                    "این هشدار به معنی تضمین معامله موفق نیست."
                )

            elif price > high + 0.5 * max(
                (high - low),
                0
            ):

                item["status"] = "MISSED"

                alert = (

                    f"⚠️ فرصت ورود {symbol} فعلاً از دست رفت؛ "

                    f"قیمت از محدوده ورود عبور کرده است. "

                    f"قیمت: {fmt(price)}. دنبال قیمت نرو."
                )

        # سیگنال وارد محدوده شده

        elif status == "ENTERED":

            df = candles(
                symbol,
                60
            )

            trend = (
                analyze(df)
                if df is not None
                else None
            )

            if price <= sl:

                item["status"] = "SL_HIT"

                alert = (

                    f"🔴 هشدار حد ضرر {symbol}\n"

                    f"قیمت فعلی: {fmt(price)} | "

                    f"حد ضرر: {fmt(sl)}"
                )

            elif price >= tp2:

                item["status"] = "TP2_HIT"

                alert = (

                    f"🎯 هدف دوم {symbol} لمس شد\n"

                    f"قیمت فعلی: {fmt(price)}"
                )

            elif (
                price >= tp1
                and "TP1_HIT" not in item.get("alerts", [])
            ):

                item.setdefault(
                    "alerts",
                    []
                ).append("TP1_HIT")

                alert = (

                    f"🎯 هدف اول {symbol} لمس شد\n"

                    f"قیمت فعلی: {fmt(price)} | "

                    f"هدف اول: {fmt(tp1)}"
                )

            elif (
                trend
                and trend["score"] < 45
                and trend["macd"] < trend["macd_signal"]
            ):

                item["status"] = "INVALIDATED"

                alert = (

                    f"⚠️ شرایط روند {symbol} تضعیف و سیگنال باطل شد. "

                    f"امتیاز ۱ساعته: {trend['score']} | "

                    f"قیمت: {fmt(price)}"
                )

        if alert:

            telegram(alert)

            item.setdefault(
                "alerts",
                []
            ).append(
                item.get("status", "ALERT")
            )

            item["last_alert_time"] = now_ts()

        item["last_price"] = price

        item["last_checked"] = now_ts()

    state["last_monitor"] = now_ts()

    save_state(state)


def main():

    state = load_state()

    force = (
        os.environ.get("EVENT_NAME")
        == "workflow_dispatch"
    )

    due = (
        now_ts() -
        int(state.get("last_scan", 0))
        >= SCAN_EVERY_SECONDS
    )

    if force or due:

        run_full_scan(state)

        save_state(state)

    monitor(state)

    save_state(state)

    print("Bot cycle completed")


if __name__ == "__main__":

    main()
