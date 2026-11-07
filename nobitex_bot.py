import os
import json
import time
import math
import logging
import html

import requests
import numpy as np
import pandas as pd

# ============================================================
# Nobitex Smart Scanner v5
# Smart Market Scanner + Telegram Alerts
# No automatic trading
# ============================================================

BASE = "https://apiv2.nobitex.ir"

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

STATE_FILE = "bot_state.json"
STATE_VERSION = 5

TOP_PER_QUOTE = 15
MAX_WATCH = 5

MIN_WATCH_SCORE = 60
BUY_SETUP_SCORE = 80

SCAN_EVERY_SECONDS = 4 * 60 * 60

TIMEFRAMES = (15, 60, 240)

CANDLE_COUNT = 350
REQUEST_PAUSE = 0.45

HTTP_TIMEOUT = 25

PRICE_MISMATCH_THRESHOLD = 0.15

TELEGRAM_CHUNK_SIZE = 3800

# حداقل امتیاز برای نمایش ارز در بخش جدید گزارش
TIMEFRAME_HIGHLIGHT_SCORE = 70


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("NobitexScanner")

S = requests.Session()

S.headers.update({
    "User-Agent": "NobitexSmartScanner/5.0",
    "Accept": "application/json"
})


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def now_ts():
    return int(time.time())


def num(value, default=float("nan")):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

        return default

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


def esc(value):
    return html.escape(str(value), quote=True)


def score_bar(score):
    score = max(0, min(100, num(score, 0)))

    filled = int(round(score / 10))

    return "🟩" * filled + "⬜" * (10 - filled)


def score_label(score):

    if score >= 80:
        return "قوی"

    if score >= 70:
        return "نسبتاً قوی"

    if score >= 60:
        return "متوسط"

    return "ضعیف"


def quote_label(quote):

    if quote == "rls":
        return "بازار تومانی (IRT)"

    if quote == "usdt":
        return "بازار تتری (USDT)"

    return quote


def line_separator():
    return "━━━━━━━━━━━━━━━━━━━━"


# ============================================================
# TELEGRAM
# ============================================================

def telegram(message):

    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "BOT_TOKEN or CHAT_ID is missing"
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    lines = message.split("\n")

    chunks = []
    current = ""

    for line in lines:

        candidate = (
            current + "\n" + line
            if current
            else line
        )

        if len(candidate) <= TELEGRAM_CHUNK_SIZE:

            current = candidate

        else:

            if current:
                chunks.append(current)

            current = line

    if current:
        chunks.append(current)

    for chunk in chunks:

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=HTTP_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("ok"):

            raise RuntimeError(
                "Telegram error: " + str(result)
            )

        time.sleep(1)


# ============================================================
# API
# ============================================================

def api_get(path, params=None, attempts=4):

    for attempt in range(attempts):

        try:

            response = S.get(
                BASE + path,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 429:

                wait = min(
                    30,
                    3 * (attempt + 1)
                )

                log.warning(
                    "Rate limited. Waiting %s seconds",
                    wait
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            payload = response.json()

            if (
                isinstance(payload, dict)
                and payload.get("status") == "failed"
            ):

                time.sleep(2 * (attempt + 1))
                continue

            return payload

        except Exception as error:

            log.warning(
                "API error %s: %s",
                path,
                error
            )

            time.sleep(2 * (attempt + 1))

    return None


# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state():

    try:

        with open(
            STATE_FILE,
            encoding="utf-8"
        ) as file:

            state = json.load(file)

        if isinstance(state, dict):

            state.setdefault("last_scan", 0)
            state.setdefault("watchlist", [])
            state.setdefault("last_monitor", 0)

            return state

    except Exception as error:

        log.warning(
            "State load error: %s",
            error
        )

    return {
        "version": STATE_VERSION,
        "last_scan": 0,
        "watchlist": [],
        "last_monitor": 0
    }


def save_state(state):

    state["version"] = STATE_VERSION

    temporary_file = STATE_FILE + ".tmp"

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temporary_file,
        STATE_FILE
    )


# ============================================================
# MARKET SYMBOL
# ============================================================

def symbol_to_market_key(symbol):

    symbol = symbol.upper()

    if symbol.endswith("IRT"):

        return (
            symbol[:-3].lower()
            + "-rls"
        )

    if symbol.endswith("USDT"):

        return (
            symbol[:-4].lower()
            + "-usdt"
        )

    return ""


def symbol_base(symbol):

    symbol = symbol.upper()

    if symbol.endswith("IRT"):
        return symbol[:-3]

    if symbol.endswith("USDT"):
        return symbol[:-4]

    return symbol


# ============================================================
# GET MARKETS
# ============================================================

def get_markets():

    payload = api_get("/market/stats")

    if (
        not payload
        or not isinstance(
            payload.get("stats"),
            dict
        )
    ):

        raise RuntimeError(
            "دریافت بازارهای نوبیتکس ناموفق بود"
        )

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

        if str(
            item.get("isClosed", False)
        ).lower() == "true":
            continue

        price = num(
            item.get("latest"),
            0
        )

        if price <= 0:
            continue

        symbol = (
            base.upper()
            + (
                "IRT"
                if quote == "rls"
                else "USDT"
            )
        )

        markets.append({

            "symbol": symbol,

            "market_key": key.lower(),

            "quote": quote,

            "price": price,

            "volume": num(
                item.get("volumeDst"),
                0
            ),

            "change24": num(
                item.get("dayChange"),
                0
            )

        })

    return markets


# ============================================================
# ALL PRICES
# ============================================================

def get_all_prices():

    payload = api_get("/market/stats")

    if (
        not payload
        or not isinstance(
            payload.get("stats"),
            dict
        )
    ):

        return {}

    prices = {}

    for key, item in payload["stats"].items():

        if not isinstance(item, dict):
            continue

        if str(
            item.get("isClosed", False)
        ).lower() == "true":
            continue

        price = num(
            item.get("latest"),
            0
        )

        if price > 0:

            prices[key.lower()] = price

    return prices


# ============================================================
# CANDLE DATA
# ============================================================

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

    if (
        not payload
        or payload.get("s") != "ok"
    ):

        return None

    try:

        df = pd.DataFrame({

            "time": payload["t"],

            "open": payload["o"],

            "high": payload["h"],

            "low": payload["l"],

            "close": payload["c"],

            "volume": payload.get(
                "v",
                [0] * len(payload["t"])
            )

        })

        for column in (
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ):

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = (
            df.dropna()
            .sort_values("time")
            .drop_duplicates("time")
        )

        df = df[
            df.time + resolution * 60
            <= current_time
        ]

        if len(df) < 220:
            return None

        last_closed = int(
            df.time.iloc[-1]
            + resolution * 60
        )

        if (
            current_time - last_closed
            > resolution * 60 * 2
        ):

            return None

        return df.reset_index(drop=True)

    except Exception as error:

        log.warning(
            "Candle error %s: %s",
            symbol,
            error
        )

        return None


# ============================================================
# INDICATORS
# ============================================================

def indicators(df):

    d = df.copy()

    close = d.close
    high = d.high
    low = d.low
    volume = d.volume

    d["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    d["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    d["ema200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    d["rsi"] = (
        100 - 100 / (1 + rs)
    )

    d.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi"
    ] = 100

    d.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "rsi"
    ] = 50

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    d["macd"] = ema12 - ema26

    d["macd_signal"] = d.macd.ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd_hist"] = (
        d.macd - d.macd_signal
    )

    previous_close = close.shift(1)

    true_range = pd.concat([

        high - low,

        (high - previous_close).abs(),

        (low - previous_close).abs()

    ], axis=1).max(axis=1)

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0
        ),
        index=d.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0
        ),
        index=d.index
    )

    atr = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    plus_smoothed = plus_dm.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    plus_di = (
        100 * plus_smoothed
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_smoothed
        / atr.replace(0, np.nan)
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    d["atr"] = atr

    d["plus_di"] = plus_di

    d["minus_di"] = minus_di

    d["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    middle = close.rolling(20).mean()

    deviation = close.rolling(20).std()

    d["bb_mid"] = middle

    d["bb_upper"] = (
        middle + 2 * deviation
    )

    d["bb_lower"] = (
        middle - 2 * deviation
    )

    tenkan = (
        high.rolling(9).max()
        + low.rolling(9).min()
    ) / 2

    kijun = (
        high.rolling(26).max()
        + low.rolling(26).min()
    ) / 2

    d["tenkan"] = tenkan

    d["kijun"] = kijun

    d["cloud_a"] = (
        tenkan + kijun
    ) / 2

    d["cloud_b"] = (
        high.rolling(52).max()
        + low.rolling(52).min()
    ) / 2

    direction = np.sign(
        close.diff()
    ).fillna(0)

    d["obv"] = (
        direction * volume
    ).cumsum()

    d["obv_ma"] = (
        d.obv.rolling(20).mean()
    )

    return d


# ============================================================
# ANALYSIS
# ============================================================

def analyze(df):

    d = indicators(df)

    row = d.iloc[-1]

    keys = (

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

    )

    values = {
        key: num(row[key])
        for key in keys
    }

    required = (

        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "macd",
        "macd_signal",
        "atr"

    )

    if not all(
        math.isfinite(values[key])
        for key in required
    ):

        return None

    score = 50.0

    reasons = []

    if values["close"] > values["ema20"]:

        score += 5

        reasons.append(
            "قیمت بالای EMA20"
        )

    else:

        score -= 5

    if values["ema20"] > values["ema50"]:

        score += 8

        reasons.append(
            "EMA20 بالای EMA50"
        )

    else:

        score -= 8

    if values["ema50"] > values["ema200"]:

        score += 8

        reasons.append(
            "روند میان‌مدت صعودی"
        )

    else:

        score -= 8

    if values["macd"] > values["macd_signal"]:

        score += 8

        reasons.append(
            "MACD صعودی"
        )

    else:

        score -= 6

    rsi = values["rsi"]

    if 50 <= rsi <= 68:

        score += 8

        reasons.append(
            "RSI در محدوده مناسب"
        )

    elif 45 <= rsi < 50 or 68 < rsi <= 74:

        score += 3

    elif rsi > 78:

        score -= 8

        reasons.append(
            "RSI بیش‌ازحد بالا"
        )

    elif rsi < 40:

        score -= 7

        reasons.append(
            "مومنتوم ضعیف"
        )

    if math.isfinite(values["adx"]):

        if (
            values["adx"] >= 18
            and values["plus_di"] > values["minus_di"]
        ):

            score += 6

            reasons.append(
                "قدرت روند خریداران"
            )

        elif (
            values["adx"] >= 25
            and values["minus_di"] > values["plus_di"]
        ):

            score -= 5

    if (
        math.isfinite(values["cloud_a"])
        and math.isfinite(values["cloud_b"])
    ):

        cloud_top = max(
            values["cloud_a"],
            values["cloud_b"]
        )

        cloud_bottom = min(
            values["cloud_a"],
            values["cloud_b"]
        )

        if values["close"] > cloud_top:

            score += 4

            reasons.append(
                "قیمت بالای ابر ایچیموکو"
            )

        elif values["close"] < cloud_bottom:

            score -= 4

    if math.isfinite(values["obv_ma"]):

        if values["obv"] > values["obv_ma"]:

            score += 3

            reasons.append(
                "حجم به نفع خریداران"
            )

        else:

            score -= 2

    values["score"] = round(
        max(0, min(100, score)),
        1
    )

    values["reasons"] = reasons

    values["last_candle_time"] = int(
        df.time.iloc[-1]
    )

    recent = df.tail(120)

    swing_high = float(
        recent.high.max()
    )

    swing_low = float(
        recent.low.min()
    )

    swing = swing_high - swing_low

    values["fib_382"] = (
        swing_low + 0.382 * swing
        if swing > 0
        else float("nan")
    )

    values["fib_500"] = (
        swing_low + 0.500 * swing
        if swing > 0
        else float("nan")
    )

    values["fib_618"] = (
        swing_low + 0.618 * swing
        if swing > 0
        else float("nan")
    )

    return values


# ============================================================
# SPREAD
# ============================================================

def spread_pct(symbol):

    payload = api_get(
        f"/v3/orderbook/{symbol}",
        attempts=2
    )

    try:

        if not isinstance(payload, dict):
            return None

        data = payload.get(
            "data",
            payload
        )

        bids = data.get("bids")

        asks = data.get("asks")

        if not bids or not asks:
            return None

        first_bid = bids[0]

        first_ask = asks[0]

        if isinstance(first_bid, (list, tuple)):

            bid = num(first_bid[0])

        else:

            bid = num(
                first_bid.get("price")
            )

        if isinstance(first_ask, (list, tuple)):

            ask = num(first_ask[0])

        else:

            ask = num(
                first_ask.get("price")
            )

        if bid <= 0 or ask <= bid:
            return None

        return (
            100 * (ask - bid)
            / ((ask + bid) / 2)
        )

    except Exception:

        return None


# ============================================================
# BUILD CANDIDATE
# ============================================================

def build_candidate(market, analyses):

    a15 = analyses[15]

    a1 = analyses[60]

    a4 = analyses[240]

    weighted_score = (

        0.25 * a15["score"]

        + 0.35 * a1["score"]

        + 0.40 * a4["score"]

    )

    spread = spread_pct(
        market["symbol"]
    )

    spread_penalty = 0

    if spread is not None:

        if spread > 2.0:

            spread_penalty = 15

        elif spread > 1.2:

            spread_penalty = 8

        elif spread > 0.8:

            spread_penalty = 4

    final_score = max(
        0,
        min(
            100,
            weighted_score - spread_penalty
        )
    )

    atr15 = a15["atr"]

    entry_low = (
        a15["ema20"] - 0.35 * atr15
    )

    entry_high = (
        a15["ema20"] + 0.15 * atr15
    )

    if entry_low > entry_high:

        entry_low, entry_high = (
            entry_high,
            entry_low
        )

    reference_price = (
        entry_low + entry_high
    ) / 2

    risk_distance = max(
        1.5 * atr15,
        0.8 * a1["atr"]
    )

    stop_loss = (
        reference_price - risk_distance
    )

    target1 = (
        reference_price + 2 * risk_distance
    )

    target2 = (
        reference_price + 3 * risk_distance
    )

    trend_alignment = (

        a1["score"] >= 60

        and a4["score"] >= 60

        and a1["macd"] > a1["macd_signal"]

        and a4["macd"] > a4["macd_signal"]

        and a4["rsi"] < 75

    )

    not_overextended = (
        a15["close"]
        <= a15["ema20"] + atr15
    )

    spread_ok = (
        spread is None
        or spread <= 1.2
    )

    if (

        final_score >= BUY_SETUP_SCORE

        and trend_alignment

        and not_overextended

        and spread_ok

    ):

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

        "created": now_ts(),

        "status": "WAITING",

        "alerts": [],

        "price_mismatch": False

    }


# ============================================================
# ANALYZE MARKET
# ============================================================

def analyze_market(market):

    analyses = {}

    for timeframe in TIMEFRAMES:

        df = candles(
            market["symbol"],
            timeframe
        )

        if df is None:

            return (
                None,
                f"داده کندل {timeframe} دقیقه‌ای موجود نبود"
            )

        result = analyze(df)

        if result is None:

            return (
                None,
                f"اندیکاتورهای {timeframe} دقیقه‌ای کامل نبود"
            )

        analyses[timeframe] = result

        time.sleep(REQUEST_PAUSE)

    return (
        build_candidate(
            market,
            analyses
        ),
        None
    )


# ============================================================
# REMOVE DUPLICATE ASSETS
# Prefer IRT when both IRT and USDT exist.
# ============================================================

def deduplicate_candidates(candidates):

    grouped = {}

    for item in candidates:

        base = symbol_base(
            item["symbol"]
        )

        if base not in grouped:

            grouped[base] = item

            continue

        current = grouped[base]

        if (
            item["quote"] == "rls"
            and current["quote"] != "rls"
        ):

            grouped[base] = item

        elif (
            item["quote"] == current["quote"]
            and item["rank"] > current["rank"]
        ):

            grouped[base] = item

    result = list(grouped.values())

    result.sort(
        key=lambda item: item["rank"],
        reverse=True
    )

    return result


# ============================================================
# TIMEFRAME SCORES >= 70
# نمایش تمام ارزهای واجد شرایط در سه تایم‌فریم
# ============================================================

def timeframe_score_section(candidates):

    labels = {
        "15m": "۱۵ دقیقه",
        "1h": "۱ ساعت",
        "4h": "۴ ساعت"
    }

    qualifying = []

    for item in candidates:

        scores = item.get("scores", {})

        matched = []

        for timeframe in ("15m", "1h", "4h"):

            score = num(
                scores.get(timeframe)
            )

            if (
                math.isfinite(score)
                and score >= TIMEFRAME_HIGHLIGHT_SCORE
            ):

                matched.append(
                    (timeframe, score)
                )

        if matched:

            qualifying.append(
                (item, matched)
            )

    qualifying.sort(
        key=lambda row: max(
            score for _, score in row[1]
        ),
        reverse=True
    )

    lines = [

        "",

        line_separator(),

        "⭐ <b>ارزهای دارای امتیاز ۷۰ و بالاتر در تایم‌فریم‌ها</b>",

        "📌 فقط تایم‌فریم‌هایی نمایش داده می‌شوند که امتیازشان حداقل ۷۰ باشد.",

        ""

    ]

    if not qualifying:

        lines.append(
            "در این اسکن هیچ ارزی در هیچ تایم‌فریمی امتیاز ۷۰ یا بالاتر نگرفت."
        )

        lines.append(
            line_separator()
        )

        return lines

    total_scores = 0

    for item, matched in qualifying:

        lines.append(
            f"🟢 <b>{esc(item['symbol'])}</b>"
        )

        for timeframe, score in matched:

            lines.append(
                f"   ⏱ {labels[timeframe]}: <b>{score:.1f}/100</b>"
            )

            total_scores += 1

        lines.append("")

    lines.extend([

        f"📊 تعداد ارزهای واجد شرایط: <b>{len(qualifying)}</b>",

        f"⭐ تعداد امتیازهای واجد شرایط: <b>{total_scores}</b>",

        line_separator()

    ])

    return lines


# ============================================================
# TELEGRAM REPORT CARD
# ============================================================

def candidate_card(index, item):

    symbol = esc(item["symbol"])

    score = item["score"]

    spread_text = (

        f"{item['spread']:.3f}%"

        if item["spread"] is not None

        else "نامشخص"

    )

    reasons = item["reasons"]["4h"][:3]

    reasons_text = (
        "، ".join(esc(x) for x in reasons)
        if reasons
        else "مورد برجسته‌ای ثبت نشد"
    )

    lines = [

        line_separator(),

        f"🏅 <b>رتبه {index} | {symbol}</b>",

        f"💱 <i>{quote_label(item['quote'])}</i>",

        "",

        f"⭐ <b>امتیاز: {score}/100</b> — {score_label(score)}",

        f"<code>{score_bar(score)}</code>",

        f"📌 ارزیابی: <b>{esc(item['classification'])}</b>",

        "",

        (
            "📊 امتیاز تایم‌فریم‌ها "
            f"۱۵دقیقه / ۱ساعت / ۴ساعت:"
        ),

        (
            f"<code>{item['scores']['15m']:.1f} / "
            f"{item['scores']['1h']:.1f} / "
            f"{item['scores']['4h']:.1f}</code>"
        ),

        "",

        f"💰 قیمت آخرین کندل: <b>{fmt(item['price_at_scan'])}</b>",

        f"↔️ اسپرد: {spread_text}",

        "",

        "🎯 <b>سطوح محاسباتی:</b>",

        (
            f"🟢 محدوده ورود: "
            f"<code>{fmt(item['entry_low'])}</code>"
            f" تا "
            f"<code>{fmt(item['entry_high'])}</code>"
        ),

        f"🔴 حد ضرر: <code>{fmt(item['sl'])}</code>",

        f"🎯 هدف اول: <code>{fmt(item['tp1'])}</code>",

        f"🏆 هدف دوم: <code>{fmt(item['tp2'])}</code>",

        "",

        (
            "📈 RSI یک‌ساعته / چهارساعته: "
            f"{fmt(item['rsi']['1h'])} / "
            f"{fmt(item['rsi']['4h'])}"
        ),

        "",

        f"💡 دلایل مثبت ۴ساعته: {reasons_text}",

        line_separator()

    ]

    return lines


# ============================================================
# FULL SCAN
# ============================================================

def run_full_scan(state):

    log.info("Starting full scan")

    markets = get_markets()

    log.info(
        "Active markets: %s",
        len(markets)
    )

    if not markets:

        raise RuntimeError(
            "هیچ بازار فعالی دریافت نشد"
        )

    # STEP 1: SCREEN ALL MARKETS

    screened = []

    failures_screen = 0

    for index, market in enumerate(
        markets,
        1
    ):

        log.info(
            "Screen %s/%s: %s",
            index,
            len(markets),
            market["symbol"]
        )

        try:

            df = candles(
                market["symbol"],
                60
            )

            if df is not None:

                result = analyze(df)

                if result is not None:

                    market["screen_score"] = (
                        result["score"]
                    )

                    screened.append(market)

                else:

                    failures_screen += 1

            else:

                failures_screen += 1

        except Exception as error:

            failures_screen += 1

            log.warning(
                "Screen error %s: %s",
                market["symbol"],
                error
            )

        time.sleep(REQUEST_PAUSE)

    if not screened:

        raise RuntimeError(
            "هیچ بازاری در مرحله غربالگری داده معتبر نداشت"
        )

    # STEP 2: TOP VOLUME PER QUOTE

    selected = []

    for quote in ("rls", "usdt"):

        group = [

            market

            for market in screened

            if market["quote"] == quote

        ]

        group.sort(
            key=lambda item: item["volume"],
            reverse=True
        )

        selected.extend(
            group[:TOP_PER_QUOTE]
        )

    # STEP 3: DEEP ANALYSIS

    candidates = []

    failures_deep = 0

    for index, market in enumerate(
        selected,
        1
    ):

        log.info(
            "Deep analysis %s/%s: %s",
            index,
            len(selected),
            market["symbol"]
        )

        try:

            candidate, reason = analyze_market(
                market
            )

            if candidate is not None:

                candidates.append(candidate)

            else:

                failures_deep += 1

                log.warning(
                    "Skipped %s: %s",
                    market["symbol"],
                    reason
                )

        except Exception as error:

            failures_deep += 1

            log.warning(
                "Deep analysis error %s: %s",
                market["symbol"],
                error
            )

    if not candidates:

        raise RuntimeError(
            "تحلیل عمیق هیچ نماد معتبری تولید نکرد"
        )

    # STEP 4: REMOVE DUPLICATES

    candidates = deduplicate_candidates(
        candidates
    )

    candidates.sort(
        key=lambda item: item["rank"],
        reverse=True
    )

    # TOP FIVE

    top_five = candidates[:MAX_WATCH]

    watchlist = [

        item

        for item in top_five

        if item["score"] >= MIN_WATCH_SCORE

    ]

    # STATE UPDATE ONLY AFTER SUCCESSFUL SCAN

    state["watchlist"] = watchlist

    state["last_scan"] = now_ts()

    state["last_scan_markets"] = len(markets)

    state["last_scan_screened"] = len(screened)

    state["last_scan_selected"] = len(selected)

    state["last_scan_analyzed"] = len(candidates)

    state["last_scan_failures"] = (
        failures_screen + failures_deep
    )

    state["top_results"] = top_five

    # TELEGRAM REPORT

    message = [

        "🚀 <b>گزارش اسکن هوشمند نوبیتکس | نسخه ۵</b>",

        line_separator(),

        "",

        f"🌐 بازارهای فعال: <b>{len(markets)}</b>",

        f"📊 بازارهای دارای داده یک‌ساعته: <b>{len(screened)}</b>",

        f"🔬 بازارهای تحلیل عمیق: <b>{len(selected)}</b>",

        f"✅ تحلیل عمیق موفق: <b>{len(candidates)}</b>",

        f"⚠️ تحلیل ناموفق: <b>{failures_deep}</b>",

        f"👀 نمادهای واردشده به پایش: <b>{len(watchlist)}</b>",

        "",

        "🏆 <b>پنج رتبه برتر این اسکن:</b>",

        ""

    ]

    if not top_five:

        message.append(
            "در این اسکن نماد مناسبی پیدا نشد."
        )

    for index, item in enumerate(
        top_five,
        1
    ):

        message.extend(
            candidate_card(index, item)
        )

    if not watchlist:

        message.extend([

            "",

            "⛔ هیچ نمادی امتیاز حداقل ۶۰ نگرفت.",

            "فعلاً هیچ نمادی وارد فهرست پایش نشد."

        ])

    # ========================================================
    # NEW SECTION:
    # ALL ANALYZED COINS WITH TIMEFRAME SCORE >= 70
    # ========================================================

    message.extend(
        timeframe_score_section(candidates)
    )

    message.extend([

        "",

        line_separator(),

        "⚠️ <b>نکات مهم:</b>",

        "• امتیاز، احتمال موفقیت معامله نیست.",

        "• محدوده‌ها و اهداف، محاسبات تحلیلی هستند.",

        "• «آماده بررسی» دستور خرید قطعی نیست.",

        "• ربات سفارش خرید یا فروش ثبت نمی‌کند.",

        "• قبل از معامله واقعی، بک‌تست و حساب آزمایشی ضروری است.",

        line_separator()

    ])

    telegram(
        "\n".join(message)
    )

    log.info(
        "Scan completed successfully"
    )


# ============================================================
# PRICE CONSISTENCY CHECK
# ============================================================

def check_price_consistency(item, current_price):

    scan_price = num(
        item.get("price_at_scan"),
        0
    )

    if scan_price <= 0 or current_price <= 0:

        return False, None

    difference = abs(
        current_price - scan_price
    ) / scan_price

    if difference > PRICE_MISMATCH_THRESHOLD:

        return True, difference

    return False, difference


# ============================================================
# MONITOR
# ============================================================

def monitor(state):

    watchlist = state.get(
        "watchlist",
        []
    )

    if not watchlist:

        log.info(
            "No watchlist to monitor"
        )

        return

    prices = get_all_prices()

    if not prices:

        log.warning(
            "Could not refresh market prices"
        )

        return

    for item in watchlist:

        symbol = item["symbol"]

        key = symbol_to_market_key(
            symbol
        )

        price = num(
            prices.get(key),
            0
        )

        if price <= 0:
            continue

        # PRICE MISMATCH PROTECTION

        mismatch, difference = check_price_consistency(
            item,
            price
        )

        if mismatch:

            if not item.get(
                "price_mismatch",
                False
            ):

                item["price_mismatch"] = True

                item["status"] = "DATA_MISMATCH"

                percent = difference * 100

                alert = (

                    f"🚨 <b>هشدار ناهماهنگی قیمت | {esc(symbol)}</b>\n\n"

                    f"قیمت زمان اسکن: <code>{fmt(item['price_at_scan'])}</code>\n"

                    f"قیمت فعلی API: <code>{fmt(price)}</code>\n"

                    f"اختلاف: <b>{percent:.1f}%</b>\n\n"

                    "⛔ برای جلوگیری از هشدار اشتباه، پایش معاملاتی این نماد متوقف شد.\n"

                    "تا بررسی واحد قیمت و داده‌های نوبیتکس، به این نماد اتکا نکن."

                )

                try:

                    telegram(alert)

                except Exception as error:

                    log.error(
                        "Mismatch Telegram error: %s",
                        error
                    )

            item["last_price"] = price

            item["last_checked"] = now_ts()

            continue

        if item.get("price_mismatch", False):

            continue

        status = item.get(
            "status",
            "WAITING"
        )

        low = num(
            item.get("entry_low")
        )

        high = num(
            item.get("entry_high")
        )

        stop = num(
            item.get("sl")
        )

        target1 = num(
            item.get("tp1")
        )

        target2 = num(
            item.get("tp2")
        )

        alert = None

        # WAITING

        if status == "WAITING":

            if price <= stop:

                item["status"] = "INVALIDATED"

                alert = (

                    f"⛔ <b>فرصت {esc(symbol)} نامعتبر شد.</b>\n\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n"

                    f"حد ابطال محاسباتی: <code>{fmt(stop)}</code>"

                )

            elif low <= price <= high:

                item["status"] = (
                    "ENTRY_ZONE_ALERTED"
                )

                item["entry_alert_price"] = price

                alert = (

                    f"🟡 <b>قیمت {esc(symbol)} وارد محدوده موردنظر شد.</b>\n\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n"

                    f"محدوده محاسباتی: <code>{fmt(low)}</code> تا <code>{fmt(high)}</code>\n"

                    f"حد ضرر محاسباتی: <code>{fmt(stop)}</code>\n"

                    f"هدف اول: <code>{fmt(target1)}</code>\n"

                    f"هدف دوم: <code>{fmt(target2)}</code>\n\n"

                    "⚠️ این هشدار به معنی توصیه یا دستور خرید نیست.\n"

                    "قبل از هر تصمیم، وضعیت بازار و حجم معاملات را بررسی کن."

                )

            elif (

                price > high
                + 0.75 * max(high - low, 0)

            ):

                item["status"] = "MISSED"

                alert = (

                    f"⚠️ <b>محدوده مناسب {esc(symbol)} فعلاً از دسترس خارج شد.</b>\n\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n\n"

                    "دنبال قیمت نرو؛ منتظر موقعیت جدید بمان."

                )

        # AFTER ENTRY ALERT

        elif status == "ENTRY_ZONE_ALERTED":

            alerts = item.setdefault(
                "alerts",
                []
            )

            if (

                price <= stop

                and "THEORETICAL_SL_TOUCHED"
                not in alerts

            ):

                alerts.append(
                    "THEORETICAL_SL_TOUCHED"
                )

                alert = (

                    f"🔴 <b>قیمت {esc(symbol)} به حد ضرر محاسباتی رسید.</b>\n\n"

                    f"قیمت: <code>{fmt(price)}</code>\n\n"

                    "این گزارش معامله واقعی شما نیست."

                )

            elif (

                price >= target2

                and "THEORETICAL_TP2_TOUCHED"
                not in alerts

            ):

                alerts.append(
                    "THEORETICAL_TP2_TOUCHED"
                )

                alert = (

                    f"🏆 <b>قیمت {esc(symbol)} به هدف دوم رسید.</b>\n\n"

                    f"قیمت: <code>{fmt(price)}</code>\n\n"

                    "این صرفاً حرکت قیمت است."

                )

            elif (

                price >= target1

                and "THEORETICAL_TP1_TOUCHED"
                not in alerts

            ):

                alerts.append(
                    "THEORETICAL_TP1_TOUCHED"
                )

                alert = (

                    f"🎯 <b>قیمت {esc(symbol)} به هدف اول رسید.</b>\n\n"

                    f"قیمت: <code>{fmt(price)}</code>\n\n"

                    "این صرفاً حرکت قیمت است."

                )

        item["last_price"] = price

        item["last_checked"] = now_ts()

        if alert:

            try:

                telegram(alert)

                item["last_alert_time"] = now_ts()

            except Exception as error:

                log.error(
                    "Telegram alert error: %s",
                    error
                )

    state["last_monitor"] = now_ts()

    save_state(state)


# ============================================================
# MAIN
# ============================================================

def main():

    if not BOT_TOKEN or not CHAT_ID:

        raise RuntimeError(
            "BOT_TOKEN و CHAT_ID باید در GitHub Secrets تعریف شوند."
        )

    state = load_state()

    old_version = int(
        state.get("version", 0)
    )

    if old_version < STATE_VERSION:

        state["last_scan"] = 0

        state["watchlist"] = []

        state["last_monitor"] = 0

    event_name = (

        os.getenv("EVENT_NAME", "")

        or os.getenv("GITHUB_EVENT_NAME", "")

    )

    force = (
        event_name == "workflow_dispatch"
    )

    last_scan = int(
        state.get("last_scan", 0)
    )

    due = (
        now_ts() - last_scan
        >= SCAN_EVERY_SECONDS
    )

    log.info("================================")

    log.info(
        "Nobitex Smart Scanner v5"
    )

    log.info(
        "Force scan: %s",
        force
    )

    log.info(
        "Scan due: %s",
        due
    )

    log.info("================================")

    # FULL SCAN

    if force or due:

        try:

            run_full_scan(state)

            save_state(state)

        except Exception as error:

            log.exception(
                "FULL SCAN FAILED"
            )

            try:

                telegram(
                    "⚠️ <b>خطا در اسکن کامل نوبیتکس:</b>\n"
                    + esc(str(error)[:2500])
                )

            except Exception as telegram_error:

                log.error(
                    "Telegram error: %s",
                    telegram_error
                )

    # PRICE MONITOR

    try:

        monitor(state)

    except Exception as error:

        log.exception(
            "MONITOR FAILED"
        )

        try:

            telegram(
                "⚠️ <b>خطا در پایش قیمت‌ها:</b>\n"
                + esc(str(error)[:2500])
            )

        except Exception:

            pass

    save_state(state)

    log.info(
        "Bot cycle completed"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
