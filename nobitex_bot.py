import os
import json
import time
import math

import requests
import numpy as np
import pandas as pd

# ============================================================
# Nobitex Smart Scanner v3
# رتبه‌بندی بازارها، ساخت محدوده‌های احتمالی و هشدار تلگرام
# این ربات سفارش ثبت نمی‌کند و سودآوری آن هنوز اثبات نشده است.
# ============================================================

BASE = "https://apiv2.nobitex.ir"
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

STATE_FILE = "bot_state.json"
STATE_VERSION = 3

# غربال اولیه همه بازارها؛ سپس تحلیل عمیق بازارهای پرحجم هر گروه
TOP_PER_QUOTE = 15
MAX_WATCH = 5
MIN_WATCH_SCORE = 60
BUY_SETUP_SCORE = 80

SCAN_EVERY_SECONDS = 4 * 60 * 60
TIMEFRAMES = (15, 60, 240)
CANDLE_COUNT = 350
REQUEST_PAUSE = 0.45

S = requests.Session()
S.headers.update({
    "User-Agent": "NobitexSmartScanner/3.0",
    "Accept": "application/json"
})


def now_ts():
    return int(time.time())


def num(value, default=float("nan")):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def fmt(value):
    value = num(value)
    if not math.isfinite(value):
        return "نامشخص"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.5f}"
    return f"{value:.8f}"


def api_get(path, params=None, attempts=4):
    for attempt in range(attempts):
        try:
            response = S.get(BASE + path, params=params, timeout=25)

            if response.status_code == 429:
                time.sleep(min(20, 3 * (attempt + 1)))
                continue

            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict) and payload.get("status") == "failed":
                time.sleep(2 * (attempt + 1))
                continue

            return payload

        except Exception as error:
            print("API error:", path, str(error))
            time.sleep(2 * (attempt + 1))

    return None


def telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    for start in range(0, len(message), 3800):
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message[start:start + 3800]
            },
            timeout=25
        )
        response.raise_for_status()

        result = response.json()
        if not result.get("ok"):
            raise RuntimeError("Telegram send failed: " + str(result))

        time.sleep(1)


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as file:
            state = json.load(file)

        if isinstance(state, dict):
            state.setdefault("last_scan", 0)
            state.setdefault("watchlist", [])
            state.setdefault("last_monitor", 0)
            return state

    except Exception as error:
        print("State load note:", str(error))

    return {
        "version": STATE_VERSION,
        "last_scan": 0,
        "watchlist": [],
        "last_monitor": 0
    }


def save_state(state):
    state["version"] = STATE_VERSION
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def symbol_to_market_key(symbol):
    symbol = symbol.upper()

    if symbol.endswith("IRT"):
        return symbol[:-3].lower() + "-rls"

    if symbol.endswith("USDT"):
        return symbol[:-4].lower() + "-usdt"

    return ""


def get_markets():
    payload = api_get("/market/stats")

    if not payload or not isinstance(payload.get("stats"), dict):
        raise RuntimeError("دریافت فهرست بازارهای نوبیتکس ناموفق بود")

    markets = []

    for key, item in payload["stats"].items():
        if not isinstance(item, dict):
            continue

        parts = key.lower().split("-")
        if len(parts) != 2:
            continue

        base, quote = parts
        if quote not in ("rls", "usdt"):
            continue

        if str(item.get("isClosed", False)).lower() == "true":
            continue

        price = num(item.get("latest"), 0)
        if price <= 0:
            continue

        symbol = base.upper() + ("IRT" if quote == "rls" else "USDT")

        markets.append({
            "symbol": symbol,
            "market_key": key.lower(),
            "quote": quote,
            "price": price,
            "volume": num(item.get("volumeDst"), 0),
            "change24": num(item.get("dayChange"), 0)
        })

    return markets


def get_all_prices():
    payload = api_get("/market/stats")

    if not payload or not isinstance(payload.get("stats"), dict):
        return {}

    prices = {}

    for key, item in payload["stats"].items():
        if not isinstance(item, dict):
            continue

        if str(item.get("isClosed", False)).lower() == "true":
            continue

        price = num(item.get("latest"), 0)
        if price > 0:
            prices[key.lower()] = price

    return prices


def candles(symbol, resolution):
    current_time = now_ts()

    payload = api_get(
        "/market/udf/history",
        {
            "symbol": symbol,
            "resolution": str(resolution),
            "to": current_time,
            "countback": CANDLE_COUNT
        }
    )

    if not payload or payload.get("s") != "ok":
        return None

    try:
        df = pd.DataFrame({
            "time": payload["t"],
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload.get("v", [0] * len(payload["t"]))
        })

        for column in ("time", "open", "high", "low", "close", "volume"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = (
            df.dropna()
              .sort_values("time")
              .drop_duplicates("time")
        )

        # کندل در حال تشکیل را وارد محاسبات نمی‌کنیم.
        df = df[df.time + resolution * 60 <= current_time]

        if len(df) < 220:
            return None

        last_closed = int(df.time.iloc[-1] + resolution * 60)
        if current_time - last_closed > resolution * 60 * 2:
            return None

        return df.reset_index(drop=True)

    except Exception as error:
        print("Candle parse error:", symbol, resolution, str(error))
        return None


def indicators(df):
    d = df.copy()

    close = d.close
    high = d.high
    low = d.low
    volume = d.volume

    d["ema20"] = close.ewm(span=20, adjust=False).mean()
    d["ema50"] = close.ewm(span=50, adjust=False).mean()
    d["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)
    d.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    d.loc[(avg_loss == 0) & (avg_gain == 0), "rsi"] = 50

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    d["macd"] = ema12 - ema26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d.macd - d.macd_signal

    # ATR و ADX
    previous_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=d.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=d.index
    )

    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_smoothed = plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    minus_smoothed = minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    plus_di = 100 * plus_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    d["atr"] = atr
    d["plus_di"] = plus_di
    d["minus_di"] = minus_di
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # Bollinger
    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()

    d["bb_mid"] = middle
    d["bb_upper"] = middle + 2 * deviation
    d["bb_lower"] = middle - 2 * deviation

    # Ichimoku: خطوط جاری؛ بدون استفاده از مقادیر آینده
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    d["tenkan"] = tenkan
    d["kijun"] = kijun
    d["cloud_a"] = (tenkan + kijun) / 2
    d["cloud_b"] = (high.rolling(52).max() + low.rolling(52).min()) / 2

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    d["obv"] = (direction * volume).cumsum()
    d["obv_ma"] = d.obv.rolling(20).mean()

    return d


def analyze(df):
    d = indicators(df)
    row = d.iloc[-1]

    keys = (
        "close", "ema20", "ema50", "ema200", "rsi",
        "macd", "macd_signal", "macd_hist", "atr", "adx",
        "plus_di", "minus_di", "cloud_a", "cloud_b",
        "obv", "obv_ma", "bb_upper", "bb_mid", "bb_lower",
        "tenkan", "kijun"
    )

    values = {key: num(row[key]) for key in keys}

    required = (
        "close", "ema20", "ema50", "ema200",
        "rsi", "macd", "macd_signal", "atr"
    )

    if not all(math.isfinite(values[key]) for key in required):
        return None

    # امتیازدهی پیوسته؛ هیچ شرطی به‌تنهایی بازار را حذف نمی‌کند.
    score = 50.0
    reasons = []

    if values["close"] > values["ema20"]:
        score += 5
        reasons.append("قیمت بالای EMA20")
    else:
        score -= 5

    if values["ema20"] > values["ema50"]:
        score += 8
        reasons.append("EMA20 بالای EMA50")
    else:
        score -= 8

    if values["ema50"] > values["ema200"]:
        score += 8
        reasons.append("روند میان‌مدت صعودی")
    else:
        score -= 8

    if values["macd"] > values["macd_signal"]:
        score += 8
        reasons.append("MACD صعودی")
    else:
        score -= 6

    rsi = values["rsi"]
    if 50 <= rsi <= 68:
        score += 8
        reasons.append("RSI در محدوده مناسب")
    elif 45 <= rsi < 50 or 68 < rsi <= 74:
        score += 3
    elif rsi > 78:
        score -= 8
        reasons.append("RSI بیش‌ازحد بالا")
    elif rsi < 40:
        score -= 7
        reasons.append("مومنتوم ضعیف")

    if math.isfinite(values["adx"]):
        if values["adx"] >= 18 and values["plus_di"] > values["minus_di"]:
            score += 6
            reasons.append("قدرت روند خریداران")
        elif values["adx"] >= 25 and values["minus_di"] > values["plus_di"]:
            score -= 5

    if math.isfinite(values["cloud_a"]) and math.isfinite(values["cloud_b"]):
        cloud_top = max(values["cloud_a"], values["cloud_b"])
        cloud_bottom = min(values["cloud_a"], values["cloud_b"])

        if values["close"] > cloud_top:
            score += 4
            reasons.append("قیمت بالای ابر ایچیموکو")
        elif values["close"] < cloud_bottom:
            score -= 4

    if math.isfinite(values["obv_ma"]):
        if values["obv"] > values["obv_ma"]:
            score += 3
            reasons.append("حجم به نفع خریداران")
        else:
            score -= 2

    values["score"] = round(max(0, min(100, score)), 1)
    values["reasons"] = reasons
    values["last_candle_time"] = int(df.time.iloc[-1])

    # سطوح فیبوناچی از دامنه 120 کندل اخیر
    recent = df.tail(120)
    swing_high = float(recent.high.max())
    swing_low = float(recent.low.min())
    swing = swing_high - swing_low

    values["fib_382"] = swing_low + 0.382 * swing if swing > 0 else float("nan")
    values["fib_500"] = swing_low + 0.500 * swing if swing > 0 else float("nan")
    values["fib_618"] = swing_low + 0.618 * swing if swing > 0 else float("nan")

    return values


def spread_pct(symbol):
    # اگر قالب دفتر سفارش قابل خواندن نباشد، نماد حذف نمی‌شود؛ اسپرد نامشخص می‌ماند.
    payload = api_get(f"/v3/orderbook/{symbol}", attempts=2)

    try:
        if not isinstance(payload, dict):
            return None

        data = payload.get("data", payload)
        bids = data.get("bids")
        asks = data.get("asks")

        if not bids or not asks:
            return None

        first_bid = bids[0]
        first_ask = asks[0]

        bid = num(first_bid[0] if isinstance(first_bid, (list, tuple)) else first_bid.get("price"))
        ask = num(first_ask[0] if isinstance(first_ask, (list, tuple)) else first_ask.get("price"))

        if bid <= 0 or ask <= bid:
            return None

        return 100 * (ask - bid) / ((ask + bid) / 2)

    except Exception:
        return None


def build_candidate(market, analyses):
    a15 = analyses[15]
    a1 = analyses[60]
    a4 = analyses[240]

    # وزن تایم‌فریم بلندمدت بیشتر است.
    weighted_score = (
        0.25 * a15["score"]
        + 0.35 * a1["score"]
        + 0.40 * a4["score"]
    )

    spread = spread_pct(market["symbol"])
    spread_penalty = 0

    if spread is not None:
        if spread > 2.0:
            spread_penalty = 15
        elif spread > 1.2:
            spread_penalty = 8
        elif spread > 0.8:
            spread_penalty = 4

    final_score = max(0, min(100, weighted_score - spread_penalty))

    # محدوده ورود تقریبی؛ با قیمت روز بازار باید دوباره ارزیابی شود.
    atr15 = a15["atr"]
    entry_low = a15["ema20"] - 0.35 * atr15
    entry_high = a15["ema20"] + 0.15 * atr15

    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    reference_price = (entry_low + entry_high) / 2

    # ریسک بر پایه ATR پانزده‌دقیقه و یک‌ساعته
    risk_distance = max(1.5 * atr15, 0.8 * a1["atr"])
    stop_loss = reference_price - risk_distance
    target1 = reference_price + 2 * risk_distance
    target2 = reference_price + 3 * risk_distance

    trend_alignment = (
        a1["score"] >= 60
        and a4["score"] >= 60
        and a1["macd"] > a1["macd_signal"]
        and a4["macd"] > a4["macd_signal"]
        and a4["rsi"] < 75
    )

    not_overextended = a15["close"] <= a15["ema20"] + atr15
    spread_ok = spread is None or spread <= 1.2

    if final_score >= BUY_SETUP_SCORE and trend_alignment and not_overextended and spread_ok:
        classification = "آماده بررسی"
    elif final_score >= 70:
        classification = "قوی؛ منتظر تأیید"
    elif final_score >= MIN_WATCH_SCORE:
        classification = "متوسط؛ فقط پایش"
    else:
        classification = "ضعیف؛ عدم ورود"

    return {
        "symbol": market["symbol"],
        "quote": market["quote"],
        "volume": market["volume"],
        "change24": market["change24"],
        "score": round(final_score, 1),
        "rank": final_score,
        "classification": classification,
        "spread": spread,
        "price_at_scan": a15["close"],
        "entry_low": entry_low,
        "entry_high": entry_high,
        "sl": stop_loss,
        "tp1": target1,
        "tp2": target2,
        "scores": {
            "15m": a15["score"],
            "1h": a1["score"],
            "4h": a4["score"]
        },
        "rsi": {
            "15m": a15["rsi"],
            "1h": a1["rsi"],
            "4h": a4["rsi"]
        },
        "adx": {
            "15m": a15["adx"],
            "1h": a1["adx"],
            "4h": a4["adx"]
        },
        "reasons": {
            "15m": a15["reasons"],
            "1h": a1["reasons"],
            "4h": a4["reasons"]
        },
        "indicators": {
            str(tf): {
                key: analyses[tf].get(key)
                for key in (
                    "macd_hist", "ema20", "ema50", "ema200", "atr",
                    "bb_upper", "bb_mid", "bb_lower",
                    "tenkan", "kijun", "cloud_a", "cloud_b",
                    "fib_382", "fib_500", "fib_618"
                )
            }
            for tf in TIMEFRAMES
        },
        "created": now_ts(),
        "status": "WAITING",
        "alerts": []
    }


def analyze_market(market):
    analyses = {}

    for timeframe in TIMEFRAMES:
        df = candles(market["symbol"], timeframe)

        if df is None:
            return None, f"داده کندل {timeframe} دقیقه‌ای موجود نبود"

        result = analyze(df)
        if result is None:
            return None, f"اندیکاتورهای {timeframe} دقیقه‌ای کامل نبود"

        analyses[timeframe] = result
        time.sleep(REQUEST_PAUSE)

    return build_candidate(market, analyses), None


def run_full_scan(state):
    print("Starting v3 full scan")

    markets = get_markets()
    print("Active markets:", len(markets))

    # مرحله اول: غربالگری یک‌ساعته همه بازارهای فعال
    screened = []
    failures_screen = 0

    for index, market in enumerate(markets, 1):
        print(f"1h screen {index}/{len(markets)}: {market['symbol']}")

        df = candles(market["symbol"], 60)
        if df is not None:
            result = analyze(df)
            if result is not None:
                market["screen_score"] = result["score"]
                screened.append(market)
        else:
            failures_screen += 1

        time.sleep(REQUEST_PAUSE)

    # مرحله دوم: 15 بازار پرحجم از هر گروه
    selected = []

    for quote in ("rls", "usdt"):
        group = [market for market in screened if market["quote"] == quote]
        group.sort(key=lambda item: item["volume"], reverse=True)
        selected.extend(group[:TOP_PER_QUOTE])

    # مرحله سوم: تحلیل سه تایم‌فریم برای بازارهای انتخاب‌شده
    candidates = []
    failures_deep = 0

    for index, market in enumerate(selected, 1):
        print(f"Deep analysis {index}/{len(selected)}: {market['symbol']}")

        try:
            candidate, reason = analyze_market(market)
            if candidate is not None:
                candidates.append(candidate)
            else:
                failures_deep += 1
                print("Skipped:", market["symbol"], reason)
        except Exception as error:
            failures_deep += 1
            print("Deep analysis failed:", market["symbol"], str(error))

    candidates.sort(key=lambda item: item["rank"], reverse=True)

    # پنج رتبه برتر نمایش داده می‌شوند؛ فقط امتیاز >= 60 پایش می‌شود.
    top_five = candidates[:MAX_WATCH]
    watchlist = [item for item in top_five if item["score"] >= MIN_WATCH_SCORE]

    state["watchlist"] = watchlist
    state["last_scan"] = now_ts()
    state["last_scan_markets"] = len(markets)
    state["last_scan_screened"] = len(screened)
    state["last_scan_selected"] = len(selected)
    state["last_scan_analyzed"] = len(candidates)
    state["last_scan_failures"] = failures_screen + failures_deep
    state["top_results"] = top_five

    message = [
        "🔎 گزارش اسکن هوشمند نوبیتکس | نسخه ۳",
        f"بازارهای فعال شناسایی‌شده: {len(markets)}",
        f"بازارهای دارای داده یک‌ساعته: {len(screened)}",
        f"بازارهای تحلیل عمیق: {len(selected)}",
        f"تحلیل عمیق موفق: {len(candidates)} | ناموفق/داده ناکافی: {failures_deep}",
        f"نمادهای واردشده به فهرست پایش: {len(watchlist)}",
        "",
        "امتیاز، احتمال موفقیت نیست و ربات سفارش خرید یا فروش ثبت نمی‌کند."
    ]

    if top_five:
        message.append("\n🏆 پنج رتبه برتر این اسکن:")

        for index, item in enumerate(top_five, 1):
            spread_text = f"{item['spread']:.3f}%" if item["spread"] is not None else "نامشخص"

            message.extend([
                "",
                f"{index}) {item['symbol']} | امتیاز {item['score']}/100",
                f"ارزیابی: {item['classification']}",
                f"امتیاز ۱۵د/۱س/۴س: {item['scores']['15m']}/{item['scores']['1h']}/{item['scores']['4h']}",
                f"قیمت آخرین کندل: {fmt(item['price_at_scan'])} | اسپرد: {spread_text}",
                f"محدوده ورود احتمالی: {fmt(item['entry_low'])} تا {fmt(item['entry_high'])}",
                f"حد ضرر محاسباتی: {fmt(item['sl'])}",
                f"هدف ۱: {fmt(item['tp1'])} | هدف ۲: {fmt(item['tp2'])}",
                f"RSI یک‌ساعته/چهارساعته: {fmt(item['rsi']['1h'])}/{fmt(item['rsi']['4h'])}"
            ])

            reasons = item["reasons"]["4h"][:3]
            if reasons:
                message.append("دلایل مثبت ۴ساعته: " + "، ".join(reasons))

    else:
        message.extend([
            "",
            "هیچ نمادی با داده کامل سه تایم‌فریم پیدا نشد.",
            "ممکن است علت، نبود داده کافی یا خطای API باشد؛ این گزارش به‌تنهایی نبود فرصت را ثابت نمی‌کند."
        ])

    if not watchlist:
        message.extend([
            "",
            "هیچ نمادی امتیاز حداقل ۶۰ نگرفت؛ فعلاً ورود پیشنهاد نمی‌شود."
        ])

    message.extend([
        "",
        "«آماده بررسی» هم دستور خرید قطعی نیست. ابتدا باید استراتژی بک‌تست و در حساب آزمایشی ارزیابی شود."
    ])

    telegram("\n".join(message))


def monitor(state):
    watchlist = state.get("watchlist", [])

    if not watchlist:
        print("No watchlist to monitor")
        return

    # یک درخواست برای قیمت همه نمادها، نه یک درخواست برای هر نماد
    prices = get_all_prices()
    if not prices:
        print("Could not refresh market prices")
        return

    for item in watchlist:
        symbol = item["symbol"]
        key = symbol_to_market_key(symbol)
        price = num(prices.get(key), 0)

        if price <= 0:
            continue

        status = item.get("status", "WAITING")
        low = num(item.get("entry_low"))
        high = num(item.get("entry_high"))
        stop = num(item.get("sl"))
        target1 = num(item.get("tp1"))
        target2 = num(item.get("tp2"))

        alert = None

        if status == "WAITING":
            if price <= stop:
                item["status"] = "INVALIDATED"
                alert = (
                    f"⛔ فرصت {symbol} پیش از رسیدن به محدوده ورود نامعتبر شد.\n"
                    f"قیمت فعلی: {fmt(price)} | حد ابطال محاسباتی: {fmt(stop)}"
                )

            elif low <= price <= high:
                # عمداً ENTERED نمی‌نامیم؛ ربات نمی‌داند کاربر واقعاً خریده یا نه.
                item["status"] = "ENTRY_ZONE_ALERTED"
                item["entry_alert_price"] = price
                alert = (
                    f"🟡 هشدار محدوده ورود {symbol}\n"
                    f"قیمت فعلی وارد محدوده محاسباتی شده: {fmt(price)}\n"
                    f"محدوده: {fmt(low)} تا {fmt(high)}\n"
                    f"حد ضرر محاسباتی: {fmt(stop)}\n"
                    f"اهداف محاسباتی: {fmt(target1)} و {fmt(target2)}\n\n"
                    "این فقط هشدار رسیدن قیمت است؛ معامله‌ای ثبت نشده و خرید خودکار انجام نمی‌شود."
                )

            elif price > high + 0.75 * max(high - low, 0):
                item["status"] = "MISSED"
                alert = (
                    f"⚠️ محدوده ورود {symbol} فعلاً از دسترس خارج شد.\n"
                    f"قیمت فعلی: {fmt(price)}؛ دنبال قیمت نرو."
                )

        elif status == "ENTRY_ZONE_ALERTED":
            alerts = item.setdefault("alerts", [])

            if price <= stop and "THEORETICAL_SL_TOUCHED" not in alerts:
                alerts.append("THEORETICAL_SL_TOUCHED")
                alert = (
                    f"🔴 قیمت {symbol} به حد ابطال/ضرر محاسباتی رسید: {fmt(price)}.\n"
                    "این گزارش عملکرد معامله واقعی شما نیست؛ ربات از انجام خرید خبر ندارد."
                )

            elif price >= target2 and "THEORETICAL_TP2_TOUCHED" not in alerts:
                alerts.append("THEORETICAL_TP2_TOUCHED")
                alert = (
                    f"🎯 قیمت {symbol} به هدف دوم محاسباتی رسید: {fmt(price)}.\n"
                    "این صرفاً حرکت قیمت است و سود معامله واقعی را نشان نمی‌دهد."
                )

            elif price >= target1 and "THEORETICAL_TP1_TOUCHED" not in alerts:
                alerts.append("THEORETICAL_TP1_TOUCHED")
                alert = (
                    f"🎯 قیمت {symbol} به هدف اول محاسباتی رسید: {fmt(price)}.\n"
                    "این صرفاً حرکت قیمت است و سود معامله واقعی را نشان نمی‌دهد."
                )

        item["last_price"] = price
        item["last_checked"] = now_ts()

        if alert:
            telegram(alert)
            item.setdefault("alerts", []).append("MESSAGE_SENT")
            item["last_alert_time"] = now_ts()

    state["last_monitor"] = now_ts()
    save_state(state)


def main():
    state = load_state()

    # نسخه جدید باید یک‌بار اسکن تازه انجام دهد و وضعیت قدیمی را کنار بگذارد.
    old_version = int(state.get("version", 0))
    if old_version < STATE_VERSION:
        state["last_scan"] = 0
        state["watchlist"] = []

    force = os.environ.get("EVENT_NAME") == "workflow_dispatch"
    due = now_ts() - int(state.get("last_scan", 0)) >= SCAN_EVERY_SECONDS

    if force or due:
        run_full_scan(state)
        save_state(state)

    monitor(state)
    save_state(state)

    print("Bot cycle completed")


if __name__ == "__main__":
    main()
